from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import torch

from .genome import Genome


def _lut_eval(inputs: torch.Tensor, lut: List[int]) -> torch.Tensor:
    """Single-node LUT eval — kept for tests."""
    B, k = inputs.shape
    addr = torch.zeros(B, dtype=torch.long, device=inputs.device)
    for j in range(k):
        addr = addr | (inputs[:, j].long() << (k - 1 - j))
    lut_t = torch.tensor(lut, dtype=torch.bool, device=inputs.device)
    return lut_t[addr]


class EIONetwork:
    """
    Runtime interpreter for a Typed E/I/O Boolean NEAT genome.

    Update rule (synchronous, every node every timestep):
      data_inputs[port] = x_t[src]         (if src is an input node)
                        = state[t-1][src]   (if src is an active node)
      raw        = LUT(data_inputs)
      inhibition = OR(state[t-1][s] for all inhibitory sources s)
      state[t]   = raw & ~inhibition
    """

    def __init__(self, genome: Genome, device: torch.device | None = None):
        self.genome = genome
        self.device = device or torch.device("cpu")
        self.k = genome.k
        dev = self.device
        k = self.k

        self.active_ids: List[int] = sorted(
            nid for nid, node in genome.nodes.items()
            if node.enabled and node.kind in ("E", "I", "O")
        )
        N = len(self.active_ids)
        self.node_index: Dict[int, int] = {nid: i for i, nid in enumerate(self.active_ids)}

        self.output_ids: List[int] = [oid for oid in genome.output_ids if oid in self.node_index]
        self.output_index: List[int] = [self.node_index[oid] for oid in self.output_ids]

        # Build incoming connection dicts.
        # incoming_data:  dst → {port: (src_id, is_input)}
        # incoming_inhibitory: dst → [src_id, ...]
        self.incoming_data: Dict[int, Dict[int, Tuple[int, bool]]] = {nid: {} for nid in self.active_ids}
        self.incoming_inhibitory: Dict[int, List[int]] = {nid: [] for nid in self.active_ids}

        for c in genome.connections.values():
            if not c.enabled:
                continue
            if c.dst not in self.node_index:
                continue
            src_node = genome.nodes.get(c.src)
            if src_node is None or not src_node.enabled:
                continue
            if c.dst_port < 0:
                if c.src in self.node_index:
                    self.incoming_inhibitory[c.dst].append(c.src)
            else:
                if c.dst_port < k:
                    is_input = src_node.kind == "input"
                    if is_input or c.src in self.node_index:
                        self.incoming_data[c.dst][c.dst_port] = (c.src, is_input)

        # ------------------------------------------------------------------
        # Precompute vectorised gather plan
        # ------------------------------------------------------------------
        inp_dst_n: List[int] = []
        inp_dst_p: List[int] = []
        inp_src_f: List[int] = []
        node_dst_n: List[int] = []
        node_dst_p: List[int] = []
        node_src_n: List[int] = []
        inh_dst_n: List[int] = []
        inh_src_n: List[int] = []

        for local_idx, nid in enumerate(self.active_ids):
            for port, (src_id, is_input) in self.incoming_data[nid].items():
                if is_input:
                    inp_dst_n.append(local_idx)
                    inp_dst_p.append(port)
                    inp_src_f.append(src_id)
                else:
                    node_dst_n.append(local_idx)
                    node_dst_p.append(port)
                    node_src_n.append(self.node_index[src_id])
            for inh_src in self.incoming_inhibitory[nid]:
                inh_dst_n.append(local_idx)
                inh_src_n.append(self.node_index[inh_src])

        if inp_dst_n:
            self._inp_n = torch.tensor(inp_dst_n, dtype=torch.long, device=dev)
            self._inp_p = torch.tensor(inp_dst_p, dtype=torch.long, device=dev)
            self._inp_f = torch.tensor(inp_src_f, dtype=torch.long, device=dev)
        else:
            self._inp_n = self._inp_p = self._inp_f = None

        if node_dst_n:
            self._node_n = torch.tensor(node_dst_n, dtype=torch.long, device=dev)
            self._node_p = torch.tensor(node_dst_p, dtype=torch.long, device=dev)
            self._node_s = torch.tensor(node_src_n, dtype=torch.long, device=dev)
        else:
            self._node_n = self._node_p = self._node_s = None

        if inh_dst_n:
            self._inh_n = torch.tensor(inh_dst_n, dtype=torch.long, device=dev)
            self._inh_s = torch.tensor(inh_src_n, dtype=torch.long, device=dev)
        else:
            self._inh_n = self._inh_s = None

        self._has_inhibition = self._inh_n is not None

        self._lut_table = torch.tensor(
            [(genome.nodes[nid].lut or [0] * (2 ** k)) for nid in self.active_ids],
            dtype=torch.bool, device=dev,
        )  # [N, 2^k]

        self._addr_weights = torch.tensor(
            [1 << (k - 1 - j) for j in range(k)],
            dtype=torch.long, device=dev,
        )  # [k]

        self._n_idx = torch.arange(N, dtype=torch.long, device=dev).unsqueeze(0)
        self._out_t = torch.tensor(self.output_index, dtype=torch.long, device=dev)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _step(self, x_t: torch.Tensor, prev_state: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """One timestep: compute state[t] from x_t and state[t-1]."""
        k = self.k
        data_in = torch.zeros(B, N, k, dtype=torch.bool, device=self.device)

        if self._inp_n is not None:
            data_in[:, self._inp_n, self._inp_p] = x_t[:, self._inp_f]

        if self._node_n is not None:
            data_in[:, self._node_n, self._node_p] = prev_state[:, self._node_s]

        addr = torch.einsum("bnk,k->bn", data_in.to(torch.long), self._addr_weights)
        n_idx = self._n_idx.expand(B, -1)
        raw = self._lut_table[n_idx, addr]  # [B, N] bool

        if self._has_inhibition:
            # OR-reduce: sum inhibitory sources per destination, then threshold at 1.
            inh_sum = torch.zeros(B, N, dtype=torch.float32, device=self.device)
            inh_sum.scatter_add_(
                1,
                self._inh_n.unsqueeze(0).expand(B, -1),
                prev_state[:, self._inh_s].float(),
            )
            raw = raw & ~(inh_sum > 0)

        return raw

    def forward_counts(self, X: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Run a batch of binary spike sequences through the network.

        X       : [B, T, F]  long or bool
        returns : (spike_counts [B, n_outputs], stats dict)
        """
        X = X.to(self.device).bool()
        B, T, F = X.shape
        N = len(self.active_ids)

        if N == 0 or len(self.output_index) == 0:
            return (
                torch.zeros(B, self.genome.n_outputs, device=self.device),
                {"mean_activity": 0.0, "silent_frac": 1.0},
            )

        output_counts = torch.zeros(B, len(self.output_index), dtype=torch.float32, device=self.device)
        total_activity = 0.0
        state = torch.zeros(B, N, dtype=torch.bool, device=self.device)

        for t in range(T):
            state = self._step(X[:, t, :], state, B, N)
            output_counts.add_(state[:, self._out_t].float())
            total_activity += state.float().mean().item()

        silent_frac = float((output_counts.sum(dim=1) == 0).float().mean().item())
        mean_activity = total_activity / max(1, T)
        return output_counts, {"mean_activity": mean_activity, "silent_frac": silent_frac}

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_counts(X)
        pred = logits.argmax(dim=1)
        silent = logits.sum(dim=1) == 0
        if silent.any():
            pred = pred.clone()
            pred[silent] = -1
        return pred


@torch.no_grad()
def evaluate_genome(
    genome: Genome,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device | None = None,
    batch_size: int = 64,
    w_silent: float = 0.20,
    w_activity: float = 0.02,
    w_complexity: float = 0.0005,
) -> Tuple[float, Dict[str, float]]:
    """
    Fitness = accuracy - silence_penalty - activity_penalty - complexity_penalty.
    """
    dev = device or torch.device("cpu")
    net = EIONetwork(genome, dev)

    n = X.shape[0]
    correct = 0
    silent_sum = 0.0
    activity_sum = 0.0

    for start in range(0, n, batch_size):
        xb = X[start:start + batch_size].to(dev)
        yb = y[start:start + batch_size].to(dev)
        logits, stats = net.forward_counts(xb)
        b = xb.shape[0]

        pred = logits.argmax(dim=1)
        silent = logits.sum(dim=1) == 0
        correct += int(((pred == yb) & ~silent).sum().detach().cpu())

        silent_sum += stats["silent_frac"] * b
        activity_sum += stats["mean_activity"] * b

    acc = correct / max(1, n)
    silent_frac = silent_sum / max(1, n)
    mean_activity = activity_sum / max(1, n)

    enabled_e = sum(1 for nd in genome.nodes.values() if nd.kind == "E" and nd.enabled)
    enabled_i = sum(1 for nd in genome.nodes.values() if nd.kind == "I" and nd.enabled)
    enabled_data = sum(1 for c in genome.connections.values() if c.enabled and c.dst_port >= 0)
    enabled_inh = sum(1 for c in genome.connections.values() if c.enabled and c.dst_port < 0)
    complexity = enabled_data + enabled_inh + enabled_e + enabled_i

    fitness = acc - w_silent * silent_frac - w_activity * mean_activity - w_complexity * complexity

    metrics = {
        "acc": float(acc),
        "fitness": float(fitness),
        "silent_frac": float(silent_frac),
        "mean_activity": float(mean_activity),
        "enabled_e": float(enabled_e),
        "enabled_i": float(enabled_i),
        "enabled_data_conns": float(enabled_data),
        "enabled_inh_conns": float(enabled_inh),
    }
    return float(fitness), metrics
