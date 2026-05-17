from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

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
    Runtime interpreter for a H/O LIF Boolean NEAT genome.

    H nodes (hidden): LIF spike-register units with membrane_bits membrane.
    O nodes (output): pure leaky integrators with output_membrane_bits membrane;
                      final membrane value is the class logit.

    Update rule per timestep (synchronous — all nodes read t-1 state):
      data_inputs[port] = x_t[src]            (if src is an input node)
                        = h_state[t-1][src]   (if src is an H node)
      h_state[t]     = LUT_spike_h[addr_h]
      h_membrane[t]  = LUT_mem_h[addr_h]      (when membrane_bits > 0)
      o_membrane[t]  = LUT_mem_o[addr_o]      (when output_membrane_bits > 0)
    """

    def __init__(self, genome: Genome, device: torch.device | None = None):
        self.genome = genome
        self.device = device or torch.device("cpu")
        self.k = genome.k
        dev = self.device
        k = self.k

        mb = genome.membrane_bits
        out_mb = genome.effective_output_mb
        self.membrane_bits = mb
        self.output_membrane_bits = out_mb

        # ── H nodes ────────────────────────────────────────────────────
        self.h_ids: List[int] = sorted(
            nid for nid, node in genome.nodes.items()
            if node.enabled and node.kind == "H"
        )
        Nh = len(self.h_ids)
        self.h_index: Dict[int, int] = {nid: i for i, nid in enumerate(self.h_ids)}

        # ── O nodes ────────────────────────────────────────────────────
        self.o_ids: List[int] = [
            oid for oid in genome.output_ids
            if oid in genome.nodes and genome.nodes[oid].enabled
        ]
        No = len(self.o_ids)
        self.o_index: Dict[int, int] = {nid: i for i, nid in enumerate(self.o_ids)}

        # ── Gather plans ───────────────────────────────────────────────
        # For each node collect port → (src_id, is_input_node).
        # Dict assignment means last connection per port wins (matches repair_duplicate_slots).
        h_incoming: Dict[int, Dict[int, Tuple[int, bool]]] = {nid: {} for nid in self.h_ids}
        o_incoming: Dict[int, Dict[int, Tuple[int, bool]]] = {nid: {} for nid in self.o_ids}

        for c in genome.connections.values():
            if not c.enabled or not (0 <= c.dst_port < k):
                continue
            src_node = genome.nodes.get(c.src)
            if src_node is None or not src_node.enabled:
                continue
            is_input = src_node.kind == "input"
            if c.dst in self.h_index:
                # H dst: accepts input→H and H→H
                if is_input or c.src in self.h_index:
                    h_incoming[c.dst][c.dst_port] = (c.src, is_input)
            elif c.dst in self.o_index:
                # O dst: H→O only (no direct input→O)
                if c.src in self.h_index:
                    o_incoming[c.dst][c.dst_port] = (c.src, False)

        # H gather tensors
        h_inp_n: List[int] = []
        h_inp_p: List[int] = []
        h_inp_f: List[int] = []
        h_node_n: List[int] = []
        h_node_p: List[int] = []
        h_node_s: List[int] = []
        for local_idx, nid in enumerate(self.h_ids):
            for port, (src_id, is_inp) in h_incoming[nid].items():
                if is_inp:
                    h_inp_n.append(local_idx)
                    h_inp_p.append(port)
                    h_inp_f.append(src_id)
                else:
                    h_node_n.append(local_idx)
                    h_node_p.append(port)
                    h_node_s.append(self.h_index[src_id])

        if h_inp_n:
            self._h_inp_n = torch.tensor(h_inp_n, dtype=torch.long, device=dev)
            self._h_inp_p = torch.tensor(h_inp_p, dtype=torch.long, device=dev)
            self._h_inp_f = torch.tensor(h_inp_f, dtype=torch.long, device=dev)
        else:
            self._h_inp_n = self._h_inp_p = self._h_inp_f = None

        if h_node_n:
            self._h_node_n = torch.tensor(h_node_n, dtype=torch.long, device=dev)
            self._h_node_p = torch.tensor(h_node_p, dtype=torch.long, device=dev)
            self._h_node_s = torch.tensor(h_node_s, dtype=torch.long, device=dev)
        else:
            self._h_node_n = self._h_node_p = self._h_node_s = None

        # O gather tensors
        o_inp_n: List[int] = []
        o_inp_p: List[int] = []
        o_inp_f: List[int] = []
        o_node_n: List[int] = []
        o_node_p: List[int] = []
        o_node_s: List[int] = []
        for local_idx, nid in enumerate(self.o_ids):
            for port, (src_id, is_inp) in o_incoming[nid].items():
                if is_inp:
                    o_inp_n.append(local_idx)
                    o_inp_p.append(port)
                    o_inp_f.append(src_id)
                else:
                    o_node_n.append(local_idx)
                    o_node_p.append(port)
                    o_node_s.append(self.h_index[src_id])

        if o_inp_n:
            self._o_inp_n = torch.tensor(o_inp_n, dtype=torch.long, device=dev)
            self._o_inp_p = torch.tensor(o_inp_p, dtype=torch.long, device=dev)
            self._o_inp_f = torch.tensor(o_inp_f, dtype=torch.long, device=dev)
        else:
            self._o_inp_n = self._o_inp_p = self._o_inp_f = None

        if o_node_n:
            self._o_node_n = torch.tensor(o_node_n, dtype=torch.long, device=dev)
            self._o_node_p = torch.tensor(o_node_p, dtype=torch.long, device=dev)
            self._o_node_s = torch.tensor(o_node_s, dtype=torch.long, device=dev)
        else:
            self._o_node_n = self._o_node_p = self._o_node_s = None

        # ── H LUTs ─────────────────────────────────────────────────────
        h_lut_size = 2 ** (k + mb)
        _zeros_h_spike = np.zeros(h_lut_size, dtype=np.uint8)
        _zeros_h_mem   = np.zeros(h_lut_size, dtype=np.int16)
        if Nh > 0:
            self._lut_spike_h = torch.from_numpy(np.stack([
                genome.nodes[nid].lut[:h_lut_size] if genome.nodes[nid].lut is not None
                else _zeros_h_spike for nid in self.h_ids
            ])).to(dtype=torch.bool, device=dev)  # [Nh, 2^(k+mb)]
            self._lut_mem_h = torch.from_numpy(np.stack([
                genome.nodes[nid].lut_mem[:h_lut_size] if genome.nodes[nid].lut_mem is not None
                else _zeros_h_mem for nid in self.h_ids
            ]).astype(np.int16)).to(device=dev)  # [Nh, 2^(k+mb)]
        else:
            self._lut_spike_h = torch.zeros(0, h_lut_size, dtype=torch.bool, device=dev)
            self._lut_mem_h   = torch.zeros(0, h_lut_size, dtype=torch.int16, device=dev)

        # ── O LUTs ─────────────────────────────────────────────────────
        o_lut_size = 2 ** (k + out_mb)
        if out_mb > 0 and No > 0:
            _zeros_o_mem = np.zeros(o_lut_size, dtype=np.int16)
            self._lut_mem_o = torch.from_numpy(np.stack([
                genome.nodes[nid].lut_mem[:o_lut_size] if genome.nodes[nid].lut_mem is not None
                else _zeros_o_mem for nid in self.o_ids
            ]).astype(np.int16)).to(device=dev)  # [No, 2^(k+out_mb)]
        elif out_mb == 0 and No > 0:
            # Legacy mode: O nodes use spike LUTs
            _zeros_o_spike = np.zeros(o_lut_size, dtype=np.uint8)
            self._lut_spike_o = torch.from_numpy(np.stack([
                genome.nodes[nid].lut[:o_lut_size] if genome.nodes[nid].lut is not None
                else _zeros_o_spike for nid in self.o_ids
            ])).to(dtype=torch.bool, device=dev)  # [No, 2^k]
        else:
            if out_mb > 0:
                self._lut_mem_o   = torch.zeros(0, o_lut_size, dtype=torch.int16, device=dev)
            else:
                self._lut_spike_o = torch.zeros(0, o_lut_size, dtype=torch.bool, device=dev)

        # ── Address weights ─────────────────────────────────────────────
        h_total = k + mb
        self._addr_weights_h = torch.tensor(
            [1 << (h_total - 1 - j) for j in range(h_total)],
            dtype=torch.long, device=dev,
        )  # [k+mb]

        o_total = k + out_mb
        self._addr_weights_o = torch.tensor(
            [1 << (o_total - 1 - j) for j in range(o_total)],
            dtype=torch.long, device=dev,
        )  # [k+out_mb]

        self._nh_idx = torch.arange(Nh, dtype=torch.long, device=dev).unsqueeze(0)
        self._no_idx = torch.arange(No, dtype=torch.long, device=dev).unsqueeze(0)

        # Membrane bit-shifts for unpacking
        if mb > 0:
            self._mem_shifts_h = torch.arange(mb, dtype=torch.int16, device=dev)
        if out_mb > 0:
            self._mem_shifts_o = torch.arange(out_mb, dtype=torch.int16, device=dev)

    # ------------------------------------------------------------------
    # Forward steps
    # ------------------------------------------------------------------

    def _step_h(
        self,
        x_t: torch.Tensor,
        prev_h_state: torch.Tensor,
        B: int,
        Nh: int,
        prev_h_membrane: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One timestep for H nodes."""
        k = self.k
        mb = self.membrane_bits
        total = k + mb
        data_in = torch.zeros(B, Nh, total, dtype=torch.bool, device=self.device)

        if self._h_inp_n is not None:
            data_in[:, self._h_inp_n, self._h_inp_p] = x_t[:, self._h_inp_f]
        if self._h_node_n is not None:
            data_in[:, self._h_node_n, self._h_node_p] = prev_h_state[:, self._h_node_s]
        if mb > 0 and prev_h_membrane is not None:
            data_in[:, :, k:] = ((prev_h_membrane.unsqueeze(-1) >> self._mem_shifts_h) & 1).bool()

        addr = torch.einsum("bnj,j->bn", data_in.to(torch.long), self._addr_weights_h)
        n_idx = self._nh_idx.expand(B, -1)
        new_state = self._lut_spike_h[n_idx, addr]  # [B, Nh] bool

        new_membrane: Optional[torch.Tensor] = None
        if mb > 0:
            new_membrane = self._lut_mem_h[n_idx, addr]  # [B, Nh] int16

        return new_state, new_membrane

    def _step_o(
        self,
        x_t: torch.Tensor,
        prev_h_state: torch.Tensor,
        B: int,
        No: int,
        prev_o_membrane: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One timestep for O nodes.

        Returns new O membrane (int16) when output_membrane_bits > 0,
        or new O spike state (bool) in legacy mode.
        """
        k = self.k
        out_mb = self.output_membrane_bits
        total = k + out_mb
        data_in = torch.zeros(B, No, total, dtype=torch.bool, device=self.device)

        if self._o_inp_n is not None:
            data_in[:, self._o_inp_n, self._o_inp_p] = x_t[:, self._o_inp_f]
        if self._o_node_n is not None:
            data_in[:, self._o_node_n, self._o_node_p] = prev_h_state[:, self._o_node_s]
        if out_mb > 0 and prev_o_membrane is not None:
            data_in[:, :, k:] = ((prev_o_membrane.unsqueeze(-1) >> self._mem_shifts_o) & 1).bool()

        addr = torch.einsum("bnj,j->bn", data_in.to(torch.long), self._addr_weights_o)
        n_idx = self._no_idx.expand(B, -1)

        if out_mb > 0:
            return self._lut_mem_o[n_idx, addr]   # [B, No] int16
        else:
            return self._lut_spike_o[n_idx, addr]  # [B, No] bool

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward_counts(self, X: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Run a batch of binary spike sequences through the network.

        X       : [B, T, F]  long or bool
        returns : (output_logits [B, n_outputs], stats dict)
        """
        X = X.to(self.device).bool()
        B, T, F = X.shape
        Nh = len(self.h_ids)
        No = len(self.o_ids)

        if No == 0:
            return (
                torch.zeros(B, self.genome.n_outputs, device=self.device),
                {"silent_frac": 1.0},
            )

        mb = self.membrane_bits
        out_mb = self.output_membrane_bits

        if out_mb > 0:
            # LIF mode: O membrane = logit
            h_state = torch.zeros(B, Nh, dtype=torch.bool, device=self.device)
            h_membrane = torch.zeros(B, Nh, dtype=torch.int16, device=self.device) if mb > 0 else None
            o_membrane = torch.zeros(B, No, dtype=torch.int16, device=self.device)
            for t in range(T):
                x_t = X[:, t, :]
                prev_h_state = h_state
                h_state, h_membrane = self._step_h(x_t, prev_h_state, B, Nh, h_membrane)
                o_membrane = self._step_o(x_t, prev_h_state, B, No, o_membrane)
            output_logits = o_membrane.float()  # [B, No]
        else:
            # Legacy spike-count mode
            h_state = torch.zeros(B, Nh, dtype=torch.bool, device=self.device)
            o_counts = torch.zeros(B, No, dtype=torch.float32, device=self.device)
            for t in range(T):
                x_t = X[:, t, :]
                prev_h_state = h_state
                h_state, _ = self._step_h(x_t, prev_h_state, B, Nh)
                o_spike = self._step_o(x_t, prev_h_state, B, No)
                o_counts.add_(o_spike.float())
            output_logits = o_counts  # [B, No]

        silent_frac = float((output_logits.sum(dim=1) == 0).float().mean().item())
        return output_logits, {"silent_frac": silent_frac}

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
    complexity_coef: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Fitness = -mean CE loss."""
    dev = device or torch.device("cpu")
    net = EIONetwork(genome, dev)

    n = X.shape[0]
    correct = 0
    silent_sum = 0.0
    ce_sum = 0.0

    out_mb = genome.effective_output_mb

    for start in range(0, n, batch_size):
        xb = X[start:start + batch_size].to(dev)
        yb = y[start:start + batch_size].to(dev)
        logits, stats = net.forward_counts(xb)
        b = xb.shape[0]

        pred = logits.argmax(dim=1)
        silent = logits.sum(dim=1) == 0
        correct += int(((pred == yb) & ~silent).sum().detach().cpu())
        silent_sum += stats["silent_frac"] * b
        # Normalise logits before CE.
        # O nodes are pure accumulators: v_final ≈ I_avg × T, so dividing by T
        # gives the mean weighted input rate per timestep — a well-scaled value
        # regardless of output_membrane_bits.  (Normalising by max_v = 2^out_mb-1
        # was wrong: actual values are << max_v, making all normalised logits ≈ 0
        # and killing the CE signal.)
        T_val = xb.shape[1]
        ce_sum += F.cross_entropy(logits / T_val, yb).item() * b

    acc = correct / max(1, n)
    silent_frac = silent_sum / max(1, n)
    mean_ce = ce_sum / max(1, n)
    fitness = -mean_ce

    enabled_h    = sum(1 for nd in genome.nodes.values() if nd.kind == "H" and nd.enabled)
    enabled_data = sum(1 for c in genome.connections.values() if c.enabled and c.dst_port >= 0)
    n_nodes      = enabled_h + genome.n_outputs  # hidden + output

    fitness = -mean_ce - complexity_coef * enabled_h

    metrics = {
        "acc": float(acc),
        "fitness": float(fitness),
        "ce": float(mean_ce),
        "silent_frac": float(silent_frac),
        "enabled_h": float(enabled_h),
        "enabled_data_conns": float(enabled_data),
        "n_nodes": float(n_nodes),
    }
    return float(fitness), metrics
