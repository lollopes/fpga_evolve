from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
import copy
import random
import numpy as np

NodeKind = Literal["input", "H", "O"]
ResetMode = Literal["zero", "subtract"]


def lut_or(k: int, membrane_bits: int = 0) -> List[int]:
    size = 2 ** (k + membrane_bits)
    return [1 if i != 0 else 0 for i in range(size)]


def lut_zero(k: int, membrane_bits: int = 0) -> List[int]:
    return [0] * (2 ** (k + membrane_bits))


def random_lut(k: int, rng: random.Random, membrane_bits: int = 0) -> List[int]:
    return [rng.randint(0, 1) for _ in range(2 ** (k + membrane_bits))]


def random_lut_mem(k: int, rng: random.Random, membrane_bits: int) -> List[int]:
    """Random membrane-output LUT: each entry is an integer in [0, 2^membrane_bits - 1]."""
    max_val = (1 << membrane_bits) - 1
    return [rng.randint(0, max_val) for _ in range(2 ** (k + membrane_bits))]


def _bit_reverse(x: int, n_bits: int) -> int:
    """Reverse the n_bits least-significant bits of x."""
    result = 0
    for i in range(n_bits):
        if (x >> i) & 1:
            result |= (1 << (n_bits - 1 - i))
    return result


# ------------------------------------------------------------------
# LIF parameters
# ------------------------------------------------------------------

@dataclass
class LIFParams:
    """Compact LIF neuron parameters (integer/fixed-point).

    All arithmetic is integer; the LUT compiler translates these into
    lut_spike and lut_mem tables that the runtime indexes directly.

    threshold is None for O nodes (pure integrators; compiler forces max_v+1).
    bias is removed — neurons rely solely on weighted inputs.
    """
    leak_shift: int   # membrane decays by v >> leak_shift each step
    reset_mode: ResetMode = "zero"
    threshold: Optional[int] = None  # None for O nodes (pure integrators)

    def clone(self) -> "LIFParams":
        return copy.copy(self)

    def to_dict(self) -> dict:
        d: dict = {"leak_shift": self.leak_shift, "reset_mode": self.reset_mode}
        if self.threshold is not None:
            d["threshold"] = self.threshold
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LIFParams":
        return cls(
            leak_shift=d["leak_shift"],
            reset_mode=d.get("reset_mode", "zero"),
            threshold=d.get("threshold"),
        )


def _random_lif_params(membrane_bits: int, rng: random.Random, k: int = 0) -> LIFParams:
    """Random initialisation of LIF parameters for H nodes.

    Threshold is drawn from [1, k] so it is reachable in a single timestep
    when all k inputs fire with weight=1.  With membrane accumulation and slow
    leaks the neuron can also reach higher thresholds over multiple steps.
    No bias — neurons fire only from weighted inputs.
    """
    max_v = (1 << membrane_bits) - 1
    leak_shift = rng.randint(1, max(1, membrane_bits))
    threshold = rng.randint(1, max(1, min(k, max_v)))
    reset_mode: ResetMode = rng.choice(["zero", "subtract"])  # type: ignore[assignment]
    return LIFParams(leak_shift=leak_shift, threshold=threshold, reset_mode=reset_mode)


# ------------------------------------------------------------------
# LUT compiler
# ------------------------------------------------------------------

def compile_lif_luts_for_node(
    k: int,
    membrane_bits: int,
    lif: LIFParams,
    port_weights: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compile LUT tables for a single LIF node (fully vectorised via NumPy).

    Address encoding matches EIONetwork._step:
      upper k bits  : input spike bits, port 0 = MSB
      lower mb bits : bit-reversed membrane integer (matches _mem_shifts unpack)

    Returns (lut_spike, lut_mem), each a numpy array of length 2^(k+membrane_bits).
    lut_spike dtype: uint8 (0/1).
    lut_mem  dtype: int16, values in [0, 2^membrane_bits - 1].
    """
    mb = membrane_bits
    lut_size = 1 << (k + mb)
    max_v = (1 << mb) - 1

    addr = np.arange(lut_size, dtype=np.int32)

    # ── Decode membrane value from lower mb bits ───────────────────
    if mb > 0:
        addr_mem = addr & max_v                              # lower mb bits, [lut_size]
        # Build bit-reversal table for all 2^mb values in one NumPy pass
        rev_idx = np.arange(1 << mb, dtype=np.int32)
        rev_table = np.zeros(1 << mb, dtype=np.int32)
        for bit in range(mb):
            rev_table |= ((rev_idx >> bit) & 1) << (mb - 1 - bit)
        v_t = rev_table[addr_mem]                            # [lut_size]
    else:
        v_t = np.zeros(lut_size, dtype=np.int32)

    # ── Weighted input current ──────────────────────────────────────
    # Accumulate port-by-port to stay O(lut_size) memory (avoids the
    # [lut_size, k] x_bits matrix that becomes GBs for large k+mb).
    input_current = np.zeros(lut_size, dtype=np.int32)
    for j in range(k):
        w = int(port_weights[j])
        if w != 0:
            bit = int(mb + k - 1 - j)
            input_current += ((addr >> bit) & 1) * w
    v_leak = v_t - (v_t >> int(lif.leak_shift))
    u = np.clip(v_leak + input_current, 0, max_v)

    fired = u >= int(lif.threshold)                          # [lut_size] bool

    if lif.reset_mode == "zero":
        v_next = np.where(fired, np.int32(0), u)
    else:
        v_next = np.where(fired, np.maximum(np.int32(0), u - int(lif.threshold)), u)

    return fired.view(np.uint8), v_next.astype(np.int16)


# ------------------------------------------------------------------
# Gene dataclasses
# ------------------------------------------------------------------

@dataclass
class NodeGene:
    id: int
    kind: NodeKind
    enabled: bool = True
    lut: Optional[List[int]] = None          # spike output table, length 2^(k+membrane_bits)
    lut_mem: Optional[List[int]] = None      # membrane-next-state table; None when membrane_bits==0
    output_class: Optional[int] = None
    lif: Optional[LIFParams] = None          # LIF params (genotype for membrane_bits>0)

    def is_active(self) -> bool:
        return self.kind in ("H", "O")

    def clone(self) -> "NodeGene":
        return copy.deepcopy(self)


@dataclass
class ConnectionGene:
    innovation: int
    src: int
    dst: int
    dst_port: int   # 0..k-1 for data
    enabled: bool = True
    weight: int = 1  # signed integer weight used during LIF LUT compilation

    def key(self) -> Tuple[int, int, int]:
        return (self.src, self.dst, self.dst_port)

    def clone(self) -> "ConnectionGene":
        return copy.deepcopy(self)


@dataclass
class InnovationRegistry:
    next_innovation: int = 0
    next_node_id: int = 0
    connection_innovations: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    split_nodes: Dict[int, int] = field(default_factory=dict)

    def reserve_node_ids(self, n: int) -> None:
        self.next_node_id = max(self.next_node_id, n)

    def new_node_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def get_connection_innovation(self, src: int, dst: int, dst_port: int) -> int:
        key = (src, dst, dst_port)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.next_innovation
            self.next_innovation += 1
        return self.connection_innovations[key]

    def get_split_node_id(self, connection_innovation: int) -> int:
        if connection_innovation not in self.split_nodes:
            self.split_nodes[connection_innovation] = self.new_node_id()
        return self.split_nodes[connection_innovation]


@dataclass
class Genome:
    """
    H/O Boolean NEAT genome with LIF-compiled LUTs.

    Non-input nodes are LIF neuron LUT-register units:
      H — hidden processing node
      O — output/readout node (one per class)

    Connection semantics:
      dst_port 0..k-1 → data connection into LUT input port dst_port
      weight           → signed integer weight used at LUT compile time
    """
    k: int
    input_size: int
    n_outputs: int
    nodes: Dict[int, NodeGene]
    connections: Dict[int, ConnectionGene]
    output_ids: List[int]
    membrane_bits: int = 0          # H-node membrane bits; 0 = legacy spike-count mode
    output_membrane_bits: int = 0   # O-node membrane bits; 0 = use membrane_bits
    fitness: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def effective_output_mb(self) -> int:
        """Effective membrane bits for O nodes."""
        return self.output_membrane_bits if self.output_membrane_bits > 0 else self.membrane_bits

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def minimal(
        cls,
        input_size: int,
        n_outputs: int,
        k: int,
        registry: InnovationRegistry,
        rng: random.Random,
        initial_connections_per_output: Optional[int] = None,
        initial_h_nodes: int = 0,
        membrane_bits: int = 0,
        output_membrane_bits: int = 0,
    ) -> "Genome":
        nodes: Dict[int, NodeGene] = {}
        mb = membrane_bits
        out_mb = output_membrane_bits if output_membrane_bits > 0 else mb

        for i in range(input_size):
            nodes[i] = NodeGene(id=i, kind="input", enabled=True)

        output_ids: List[int] = []
        for c in range(n_outputs):
            nid = input_size + c
            output_ids.append(nid)
            if out_mb > 0:
                nodes[nid] = NodeGene(
                    id=nid, kind="O", enabled=True,
                    lut=None,      # compiled after connections are wired
                    lut_mem=None,
                    output_class=c,
                    # O nodes are pure weighted spike counters: no LIF params.
                    # compile_luts hardcodes no-leak, no-threshold behaviour.
                    lif=None,
                )
            else:
                nodes[nid] = NodeGene(
                    id=nid, kind="O", enabled=True,
                    lut=random_lut(k, rng, 0),
                    lut_mem=None,
                    output_class=c,
                )

        registry.reserve_node_ids(input_size + n_outputs)

        g = cls(
            k=k,
            input_size=input_size,
            n_outputs=n_outputs,
            nodes=nodes,
            connections={},
            output_ids=output_ids,
            membrane_bits=membrane_bits,
            output_membrane_bits=output_membrane_bits,
        )

        # Architecture: input → H only, H → O only (no direct input→O).
        h_ids: List[int] = []
        for _ in range(initial_h_nodes):
            nid = registry.new_node_id()
            if mb > 0:
                nodes[nid] = NodeGene(
                    id=nid, kind="H", enabled=True,
                    lut=None,
                    lut_mem=None,
                    lif=_random_lif_params(mb, rng, k),
                )
            else:
                nodes[nid] = NodeGene(
                    id=nid, kind="H", enabled=True,
                    lut=random_lut(k, rng, mb),
                    lut_mem=random_lut_mem(k, rng, mb) if mb > 0 else None,
                )
            h_ids.append(nid)

        # Wire each H node: k random inputs → H, then H → one O node.
        output_free: Dict[int, List[int]] = {}
        for oid in output_ids:
            ports_list = list(range(k))
            rng.shuffle(ports_list)
            output_free[oid] = ports_list

        for idx, h_id in enumerate(h_ids):
            srcs = rng.sample(range(input_size), k=min(k, input_size))
            for port, src in enumerate(srcs):
                innov = registry.get_connection_innovation(src, h_id, port)
                g.connections[innov] = ConnectionGene(innov, src, h_id, port, True)

            # Connect this H node to one O node (round-robin across outputs).
            base = idx % len(output_ids)
            for try_i in range(len(output_ids)):
                out_id = output_ids[(base + try_i) % len(output_ids)]
                if output_free[out_id]:
                    port = output_free[out_id].pop(0)
                    innov2 = registry.get_connection_innovation(h_id, out_id, port)
                    g.connections[innov2] = ConnectionGene(innov2, h_id, out_id, port, True)
                    break
        # No direct input→O connections.

        if mb > 0:
            g.compile_luts()

        return g

    def clone(self) -> "Genome":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Node/connection queries
    # ------------------------------------------------------------------

    def active_node_ids(self, enabled_only: bool = True) -> List[int]:
        return sorted(
            nid for nid, node in self.nodes.items()
            if node.is_active() and (node.enabled or not enabled_only)
        )

    def micro_node_ids(self, enabled_only: bool = True) -> List[int]:
        return self.active_node_ids(enabled_only)

    def source_ids(self, allow_output_feedback: bool = False) -> List[int]:
        ids = []
        for nid, node in self.nodes.items():
            if not node.enabled:
                continue
            if node.kind == "O" and not allow_output_feedback:
                continue
            ids.append(nid)
        return sorted(ids)

    def target_ids(self) -> List[int]:
        return self.active_node_ids(enabled_only=True)

    def occupied_data_slots(self, enabled_only: bool = True) -> set:
        slots: set = set()
        for c in self.connections.values():
            if enabled_only and not c.enabled:
                continue
            if c.dst_port >= 0:
                slots.add((c.dst, c.dst_port))
        return slots

    def existing_connection_by_key(self, key: Tuple[int, int, int]) -> Optional[ConnectionGene]:
        for c in self.connections.values():
            if c.key() == key:
                return c
        return None

    # ------------------------------------------------------------------
    # LIF compilation
    # ------------------------------------------------------------------

    def compile_luts(self) -> None:
        """Compile lut/lut_mem from LIF params + connection weights for every active node.

        H nodes use membrane_bits; O nodes use effective_output_mb.
        No-op when both are 0 (legacy mode: lut/lut_mem are directly evolved).
        """
        mb = self.membrane_bits
        out_mb = self.effective_output_mb
        if mb == 0 and out_mb == 0:
            return

        for nid in self.active_node_ids(enabled_only=True):
            node = self.nodes[nid]
            eff_mb = out_mb if node.kind == "O" else mb
            if eff_mb == 0:
                continue  # legacy node, LUT evolved directly

            max_v = (1 << eff_mb) - 1
            if node.lif is None:
                node.lif = _random_lif_params(eff_mb, random.Random(nid), self.k)

            port_weights = [0] * self.k
            for c in self.connections.values():
                if c.enabled and c.dst == nid and 0 <= c.dst_port < self.k:
                    port_weights[c.dst_port] = c.weight

            if node.kind == "O":
                # Output nodes are pure weighted spike counters:
                # no leak (leak_shift=out_mb → v>>out_mb==0), no threshold (max_v+1).
                integrator_lif = LIFParams(
                    threshold=max_v + 1,
                    leak_shift=eff_mb,
                    reset_mode="zero",
                )
                node.lut, node.lut_mem = compile_lif_luts_for_node(
                    self.k, eff_mb, integrator_lif, port_weights
                )
            else:
                node.lut, node.lut_mem = compile_lif_luts_for_node(
                    self.k, eff_mb, node.lif, port_weights
                )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_connection_mutation(
        self,
        registry: InnovationRegistry,
        rng: random.Random,
        allow_output_feedback: bool = False,
        max_tries: int = 64,
    ) -> bool:
        sources = self.source_ids(allow_output_feedback)
        targets = self.target_ids()
        if not sources or not targets:
            return False

        occupied_data = self.occupied_data_slots(enabled_only=True)

        for _ in range(max_tries):
            src = rng.choice(sources)
            dst = rng.choice(targets)

            # Architecture constraint: input nodes may only connect to H nodes.
            if (self.nodes[src].kind == "input" and self.nodes[dst].kind == "O"):
                continue

            free_ports = [p for p in range(self.k) if (dst, p) not in occupied_data]
            if not free_ports:
                continue
            port = rng.choice(free_ports)
            key = (src, dst, port)
            old = self.existing_connection_by_key(key)
            if old is not None:
                if not old.enabled:
                    old.enabled = True
                    occupied_data.add((dst, port))
                    return True
                continue
            innov = registry.get_connection_innovation(src, dst, port)
            self.connections[innov] = ConnectionGene(innov, src, dst, port, True)
            return True

        return False

    def add_node_mutation(
        self,
        registry: InnovationRegistry,
        rng: random.Random,
    ) -> bool:
        enabled_data = [c for c in self.connections.values() if c.enabled and c.dst_port >= 0]
        if not enabled_data:
            return False

        c = rng.choice(enabled_data)
        c.enabled = False

        new_id = registry.get_split_node_id(c.innovation)
        mb = self.membrane_bits

        if new_id not in self.nodes:
            if mb > 0:
                self.nodes[new_id] = NodeGene(
                    id=new_id, kind="H", enabled=True,
                    lut=None,
                    lut_mem=None,
                    lif=_random_lif_params(mb, rng, self.k),
                )
            else:
                self.nodes[new_id] = NodeGene(
                    id=new_id, kind="H", enabled=True,
                    lut=lut_or(self.k, mb),
                    lut_mem=None,
                )
        else:
            self.nodes[new_id].enabled = True

        innov1 = registry.get_connection_innovation(c.src, new_id, 0)
        self.connections[innov1] = ConnectionGene(innov1, c.src, new_id, 0, True)

        innov2 = registry.get_connection_innovation(new_id, c.dst, c.dst_port)
        self.connections[innov2] = ConnectionGene(innov2, new_id, c.dst, c.dst_port, True)

        self.repair_duplicate_slots(rng)
        return True

    def mutate_luts(self, rng: random.Random, bit_rate: float) -> None:
        """Legacy LUT bit-flip mutation (used only when membrane_bits == 0)."""
        if bit_rate <= 0.0:
            return
        mb = self.membrane_bits
        lut_size = 2 ** (self.k + mb)
        per_bit_rate = bit_rate / lut_size
        rng_np = np.random.default_rng(rng.randint(0, 2**31 - 1))
        for node in self.nodes.values():
            if not node.enabled or not node.is_active() or node.lut is None:
                continue
            arr = np.array(node.lut, dtype=np.uint8)
            arr[rng_np.random(lut_size) < per_bit_rate] ^= 1
            node.lut = arr.tolist()
            if mb > 0 and node.lut_mem is not None:
                bit_flips = rng_np.random((lut_size, mb)) < per_bit_rate
                bit_vals  = np.int32(1) << np.arange(mb, dtype=np.int32)
                xor_mask  = (bit_flips.astype(np.int32) * bit_vals).sum(axis=1)
                arr_mem   = np.array(node.lut_mem, dtype=np.int32)
                arr_mem   = np.clip(arr_mem ^ xor_mask, 0, (1 << mb) - 1)
                node.lut_mem = arr_mem.tolist()

    def mutate_lif_params(self, rng: random.Random, p: float = 0.1) -> None:
        """Perturb LIF parameters of every active node.

        H nodes use membrane_bits; O nodes use effective_output_mb.
        """
        mb = self.membrane_bits
        out_mb = self.effective_output_mb
        for node in self.nodes.values():
            if not node.enabled or not node.is_active():
                continue
            eff_mb = out_mb if node.kind == "O" else mb
            if eff_mb == 0:
                continue
            max_v = (1 << eff_mb) - 1
            if node.lif is None:
                node.lif = _random_lif_params(eff_mb, rng, self.k)
                continue
            lif = node.lif
            if node.kind == "O":
                continue  # O nodes have no LIF params — pure spike counters
            else:
                if rng.random() < p and lif.threshold is not None:
                    lif.threshold = max(1, min(max_v, lif.threshold + rng.choice([-1, 1])))
                if rng.random() < p:
                    lif.leak_shift = max(1, min(mb, lif.leak_shift + rng.choice([-1, 1])))
                if rng.random() < p * 0.2:
                    lif.reset_mode = "subtract" if lif.reset_mode == "zero" else "zero"

    def mutate_connection_weights(
        self, rng: random.Random, p: float, w_min: int, w_max: int
    ) -> None:
        """Perturb integer weights on enabled data connections."""
        for c in self.connections.values():
            if not c.enabled or c.dst_port < 0:
                continue
            if rng.random() < p:
                c.weight = max(w_min, min(w_max, c.weight + rng.choice([-1, 1])))

    def toggle_connection_mutation(self, rng: random.Random) -> bool:
        if not self.connections:
            return False
        c = rng.choice(list(self.connections.values()))
        c.enabled = not c.enabled
        if c.enabled:
            self.repair_duplicate_slots(rng)
        return True

    def repair_duplicate_slots(self, rng: random.Random) -> None:
        by_slot: Dict[Tuple[int, int], List[ConnectionGene]] = {}
        for c in self.connections.values():
            if not c.enabled or c.dst_port < 0:
                continue
            by_slot.setdefault((c.dst, c.dst_port), []).append(c)

        for genes in by_slot.values():
            if len(genes) <= 1:
                continue
            keep = rng.choice(genes)
            for g in genes:
                if g is not keep:
                    g.enabled = False

    def mutate(
        self,
        registry: InnovationRegistry,
        rng: random.Random,
        lut_bit_rate: float = 0.01,
        p_add_connection: float = 0.20,
        p_add_node: float = 0.05,
        p_toggle_connection: float = 0.02,
        allow_output_feedback: bool = False,
        p_mutate_lif_param: float = 0.1,
        p_mutate_conn_weight: float = 0.1,
        weight_min: int = -3,
        weight_max: int = 3,
    ) -> None:
        use_lif = self.membrane_bits > 0 or self.effective_output_mb > 0
        if use_lif:
            self.mutate_lif_params(rng, p_mutate_lif_param)
            self.mutate_connection_weights(rng, p_mutate_conn_weight, weight_min, weight_max)
        else:
            self.mutate_luts(rng, lut_bit_rate)

        if rng.random() < p_add_connection:
            self.add_connection_mutation(registry, rng, allow_output_feedback=allow_output_feedback)
        if rng.random() < p_add_node:
            self.add_node_mutation(registry, rng)
        if rng.random() < p_toggle_connection:
            self.toggle_connection_mutation(rng)
        self.repair_duplicate_slots(rng)

        if use_lif:
            self.compile_luts()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "input_size": self.input_size,
            "n_outputs": self.n_outputs,
            "output_ids": self.output_ids,
            "membrane_bits": self.membrane_bits,
            "output_membrane_bits": self.output_membrane_bits,
            "nodes": {
                str(nid): {
                    "id": n.id,
                    "kind": n.kind,
                    "enabled": n.enabled,
                    "lut": n.lut.tolist() if isinstance(n.lut, np.ndarray) else n.lut,
                    "lut_mem": n.lut_mem.tolist() if isinstance(n.lut_mem, np.ndarray) else n.lut_mem,
                    "output_class": n.output_class,
                    "lif": n.lif.to_dict() if n.lif is not None else None,
                }
                for nid, n in self.nodes.items()
            },
            "connections": {
                str(innov): {
                    "innovation": c.innovation,
                    "src": c.src,
                    "dst": c.dst,
                    "dst_port": c.dst_port,
                    "enabled": c.enabled,
                    "weight": c.weight,
                }
                for innov, c in self.connections.items()
            },
            "fitness": self.fitness,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        nodes: Dict[int, NodeGene] = {}
        for k, v in d["nodes"].items():
            lif_d = v.get("lif")
            nodes[int(k)] = NodeGene(
                id=v["id"],
                kind=v["kind"],
                enabled=v.get("enabled", True),
                lut=v.get("lut"),
                lut_mem=v.get("lut_mem"),
                output_class=v.get("output_class"),
                lif=LIFParams.from_dict(lif_d) if lif_d is not None else None,
            )
        conns: Dict[int, ConnectionGene] = {}
        for k, v in d["connections"].items():
            conns[int(k)] = ConnectionGene(
                innovation=v["innovation"],
                src=v["src"],
                dst=v["dst"],
                dst_port=v["dst_port"],
                enabled=v.get("enabled", True),
                weight=v.get("weight", 1),
            )
        g = cls(
            k=d["k"],
            input_size=d["input_size"],
            n_outputs=d["n_outputs"],
            nodes=nodes,
            connections=conns,
            output_ids=list(d["output_ids"]),
            membrane_bits=d.get("membrane_bits", 0),
            output_membrane_bits=d.get("output_membrane_bits", 0),
            fitness=d.get("fitness"),
            metrics=d.get("metrics", {}),
        )
        # Recompile if any LIF mode and any node is missing lut (e.g. loaded from old checkpoint)
        if g.membrane_bits > 0 or g.effective_output_mb > 0:
            needs_compile = any(
                n.lut is None for n in g.nodes.values() if n.is_active() and n.enabled
            )
            if needs_compile:
                g.compile_luts()
        return g


# ------------------------------------------------------------------
# Distance and crossover
# ------------------------------------------------------------------

def lut_hamming(a, b) -> float:
    if a is None or b is None:
        return 0.0 if a is b else 1.0
    if len(a) != len(b):
        return 1.0
    a_arr = a if isinstance(a, np.ndarray) else np.asarray(a)
    b_arr = b if isinstance(b, np.ndarray) else np.asarray(b)
    return float(np.mean(a_arr != b_arr))


def lif_distance(
    a: Optional[LIFParams],
    b: Optional[LIFParams],
    membrane_bits: int,
) -> float:
    """Normalised [0,1] distance between two LIFParams."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    max_v = max(1, (1 << membrane_bits) - 1)
    terms: List[float] = []
    if a.threshold is not None and b.threshold is not None:
        terms.append(abs(a.threshold - b.threshold) / max_v)
    terms.append(abs(a.leak_shift - b.leak_shift) / max(1, membrane_bits))
    terms.append(0.0 if a.reset_mode == b.reset_mode else 1.0)
    return float(np.mean(terms))


def genome_distance(
    a: Genome,
    b: Genome,
    c_disjoint: float = 1.0,
    c_lut: float = 0.4,
    c_type: float = 0.2,  # kept for API compatibility, unused
    weight_range: int = 3,
) -> float:
    ia = set(a.connections.keys())
    ib = set(b.connections.keys())
    norm = max(1, max(len(ia), len(ib)))
    d_conn = len(ia ^ ib) / norm

    mb = a.membrane_bits
    common_nodes = set(a.nodes.keys()) & set(b.nodes.keys())
    content_terms: List[float] = []
    for nid in common_nodes:
        na, nb = a.nodes[nid], b.nodes[nid]
        if not (na.is_active() and nb.is_active()):
            continue
        if mb > 0:
            content_terms.append(lif_distance(na.lif, nb.lif, mb))
        else:
            content_terms.append(lut_hamming(na.lut, nb.lut))

    # Average weight distance on common data connections (mb>0 only)
    if mb > 0:
        weight_diffs: List[float] = []
        w_range = max(1, 2 * weight_range)
        for innov in ia & ib:
            ca, cb = a.connections[innov], b.connections[innov]
            if ca.dst_port >= 0:
                weight_diffs.append(abs(ca.weight - cb.weight) / w_range)
        if weight_diffs:
            content_terms.append(float(np.mean(weight_diffs)))

    d_content = float(np.mean(content_terms)) if content_terms else 0.0

    return c_disjoint * d_conn + c_lut * d_content


def crossover(fitter: Genome, other: Genome, rng: random.Random) -> Genome:
    child = fitter.clone()
    child.connections = {}

    all_innov = set(fitter.connections.keys()) | set(other.connections.keys())
    for innov in all_innov:
        if innov in fitter.connections and innov in other.connections:
            gene = rng.choice([fitter.connections[innov], other.connections[innov]]).clone()
            if (not fitter.connections[innov].enabled or not other.connections[innov].enabled) and rng.random() < 0.75:
                gene.enabled = False
            child.connections[innov] = gene
        elif innov in fitter.connections:
            child.connections[innov] = fitter.connections[innov].clone()

    needed = set(range(fitter.input_size)) | set(fitter.output_ids)
    for c in child.connections.values():
        needed.add(c.src)
        needed.add(c.dst)

    mb = fitter.membrane_bits
    out_mb = fitter.effective_output_mb
    child.nodes = {}
    for nid in needed:
        in_fitter = nid in fitter.nodes
        in_other  = nid in other.nodes
        if in_fitter and in_other and fitter.nodes[nid].is_active():
            fn, on = fitter.nodes[nid], other.nodes[nid]
            node = rng.choice([fn, on]).clone()
            eff_mb = out_mb if node.kind == "O" else mb
            if eff_mb > 0:
                # Mix LIF params field-wise; lut/lut_mem will be recompiled
                if fn.lif is not None and on.lif is not None:
                    thresh = (
                        rng.choice([fn.lif.threshold, on.lif.threshold])
                        if fn.lif.threshold is not None and on.lif.threshold is not None
                        else None
                    )
                    node.lif = LIFParams(
                        leak_shift = rng.choice([fn.lif.leak_shift,  on.lif.leak_shift]),
                        reset_mode = rng.choice([fn.lif.reset_mode,  on.lif.reset_mode]),
                        threshold  = thresh,
                    )
                # Don't mix raw LUT entries — they will be recompiled
            else:
                if fn.lut and on.lut and len(fn.lut) == len(on.lut):
                    node.lut = [rng.choice([fa, oa]) for fa, oa in zip(fn.lut, on.lut)]
                if fn.lut_mem and on.lut_mem and len(fn.lut_mem) == len(on.lut_mem):
                    node.lut_mem = [rng.choice([fa, oa]) for fa, oa in zip(fn.lut_mem, on.lut_mem)]
        elif in_fitter:
            node = fitter.nodes[nid].clone()
        elif in_other:
            node = other.nodes[nid].clone()
        else:
            continue
        child.nodes[nid] = node

    child.fitness = None
    child.metrics = {}
    child.repair_duplicate_slots(rng)

    if mb > 0 or out_mb > 0:
        child.compile_luts()

    return child
