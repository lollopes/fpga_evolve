from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
import copy
import random
import numpy as np

NodeKind = Literal["input", "E", "I", "O"]


def lut_or(k: int) -> List[int]:
    return [1 if i != 0 else 0 for i in range(2 ** k)]


def lut_zero(k: int) -> List[int]:
    return [0] * (2 ** k)


def random_lut(k: int, rng: random.Random) -> List[int]:
    return [rng.randint(0, 1) for _ in range(2 ** k)]


@dataclass
class NodeGene:
    id: int
    kind: NodeKind
    enabled: bool = True
    lut: Optional[List[int]] = None
    output_class: Optional[int] = None

    def is_active(self) -> bool:
        return self.kind in ("E", "I", "O")

    def clone(self) -> "NodeGene":
        return copy.deepcopy(self)


@dataclass
class ConnectionGene:
    innovation: int
    src: int
    dst: int
    dst_port: int   # -1 for inhibitory; 0..k-1 for data
    enabled: bool = True

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
    Typed E/I/O Boolean NEAT genome.

    Non-input nodes are single typed LUT-register units:
      E — excitatory processing node
      I — inhibitory gating node
      O — output/readout node (one per class)

    Connection semantics:
      dst_port >= 0  →  data connection into LUT input port dst_port
      dst_port == -1 →  inhibitory connection (source inhibits target via Boolean gate)
    """
    k: int
    input_size: int
    n_outputs: int
    nodes: Dict[int, NodeGene]
    connections: Dict[int, ConnectionGene]
    output_ids: List[int]
    fitness: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)

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
        initial_e_nodes: int = 0,
    ) -> "Genome":
        nodes: Dict[int, NodeGene] = {}

        for i in range(input_size):
            nodes[i] = NodeGene(id=i, kind="input", enabled=True)

        output_ids: List[int] = []
        for c in range(n_outputs):
            nid = input_size + c
            output_ids.append(nid)
            nodes[nid] = NodeGene(id=nid, kind="O", enabled=True, lut=lut_zero(k), output_class=c)

        registry.reserve_node_ids(input_size + n_outputs)

        g = cls(
            k=k,
            input_size=input_size,
            n_outputs=n_outputs,
            nodes=nodes,
            connections={},
            output_ids=output_ids,
        )

        n_conn = initial_connections_per_output
        if n_conn is None:
            n_conn = min(k, input_size)

        e_ids: List[int] = []
        for _ in range(initial_e_nodes):
            nid = registry.new_node_id()
            nodes[nid] = NodeGene(id=nid, kind="E", enabled=True, lut=lut_zero(k))
            e_ids.append(nid)

        output_free: Dict[int, List[int]] = {}
        for oid in output_ids:
            ports_list = list(range(k))
            rng.shuffle(ports_list)
            output_free[oid] = ports_list

        for idx, e_id in enumerate(e_ids):
            srcs = rng.sample(range(input_size), k=min(k, input_size))
            for port, src in enumerate(srcs):
                innov = registry.get_connection_innovation(src, e_id, port)
                g.connections[innov] = ConnectionGene(innov, src, e_id, port, True)

            base = idx % len(output_ids)
            for try_i in range(len(output_ids)):
                out_id = output_ids[(base + try_i) % len(output_ids)]
                if output_free[out_id]:
                    port = output_free[out_id].pop(0)
                    innov2 = registry.get_connection_innovation(e_id, out_id, port)
                    g.connections[innov2] = ConnectionGene(innov2, e_id, out_id, port, True)
                    break

        for out_id in output_ids:
            remaining_ports = output_free[out_id]
            n_fill = min(n_conn, len(remaining_ports))
            if n_fill <= 0:
                continue
            srcs = rng.sample(range(input_size), k=min(n_fill, input_size))
            for port, src in zip(remaining_ports[:n_fill], srcs):
                innov = registry.get_connection_innovation(src, out_id, port)
                g.connections[innov] = ConnectionGene(innov, src, out_id, port, True)

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
        existing_inh = {(c.src, c.dst) for c in self.connections.values() if c.enabled and c.dst_port < 0}

        for _ in range(max_tries):
            src = rng.choice(sources)
            dst = rng.choice(targets)
            src_node = self.nodes[src]

            if src_node.kind == "I":
                if (src, dst) in existing_inh:
                    continue
                key = (src, dst, -1)
                old = self.existing_connection_by_key(key)
                if old is not None:
                    if not old.enabled:
                        old.enabled = True
                        return True
                    continue
                innov = registry.get_connection_innovation(src, dst, -1)
                self.connections[innov] = ConnectionGene(innov, src, dst, -1, True)
                return True
            else:
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
        p_new_node_is_inhibitory: float = 0.2,
    ) -> bool:
        enabled_data = [c for c in self.connections.values() if c.enabled and c.dst_port >= 0]
        if not enabled_data:
            return False

        c = rng.choice(enabled_data)
        c.enabled = False

        new_id = registry.get_split_node_id(c.innovation)
        new_type: NodeKind = "I" if rng.random() < p_new_node_is_inhibitory else "E"

        if new_id not in self.nodes:
            self.nodes[new_id] = NodeGene(id=new_id, kind=new_type, enabled=True, lut=lut_or(self.k))
        else:
            self.nodes[new_id].enabled = True

        innov1 = registry.get_connection_innovation(c.src, new_id, 0)
        self.connections[innov1] = ConnectionGene(innov1, c.src, new_id, 0, True)

        new_node = self.nodes[new_id]
        if new_node.kind == "I":
            innov2 = registry.get_connection_innovation(new_id, c.dst, -1)
            self.connections[innov2] = ConnectionGene(innov2, new_id, c.dst, -1, True)
        else:
            innov2 = registry.get_connection_innovation(new_id, c.dst, c.dst_port)
            self.connections[innov2] = ConnectionGene(innov2, new_id, c.dst, c.dst_port, True)

        self.repair_duplicate_slots(rng)
        return True

    def mutate_luts(self, rng: random.Random, bit_rate: float) -> None:
        if bit_rate <= 0.0:
            return
        lut_size = 2 ** self.k
        per_bit_rate = bit_rate / lut_size
        for node in self.nodes.values():
            if not node.enabled or not node.is_active() or node.lut is None:
                continue
            for i in range(lut_size):
                if rng.random() < per_bit_rate:
                    node.lut[i] = 1 - node.lut[i]

    def mutate_node_type(self, rng: random.Random, p: float) -> None:
        if p <= 0.0:
            return
        for node in self.nodes.values():
            if node.kind not in ("E", "I") or not node.enabled:
                continue
            if rng.random() < p:
                node.kind = "I" if node.kind == "E" else "E"

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
        p_new_node_is_inhibitory: float = 0.2,
        p_mutate_node_type: float = 0.01,
        allow_output_feedback: bool = False,
    ) -> None:
        self.mutate_luts(rng, lut_bit_rate)
        if rng.random() < p_add_connection:
            self.add_connection_mutation(registry, rng, allow_output_feedback=allow_output_feedback)
        if rng.random() < p_add_node:
            self.add_node_mutation(registry, rng, p_new_node_is_inhibitory=p_new_node_is_inhibitory)
        if rng.random() < p_toggle_connection:
            self.toggle_connection_mutation(rng)
        self.mutate_node_type(rng, p_mutate_node_type)
        self.repair_duplicate_slots(rng)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "input_size": self.input_size,
            "n_outputs": self.n_outputs,
            "output_ids": self.output_ids,
            "nodes": {
                str(nid): {
                    "id": n.id,
                    "kind": n.kind,
                    "enabled": n.enabled,
                    "lut": n.lut,
                    "output_class": n.output_class,
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
            nodes[int(k)] = NodeGene(
                id=v["id"],
                kind=v["kind"],
                enabled=v.get("enabled", True),
                lut=v.get("lut"),
                output_class=v.get("output_class"),
            )
        conns: Dict[int, ConnectionGene] = {}
        for k, v in d["connections"].items():
            conns[int(k)] = ConnectionGene(
                innovation=v["innovation"],
                src=v["src"],
                dst=v["dst"],
                dst_port=v["dst_port"],
                enabled=v.get("enabled", True),
            )
        return cls(
            k=d["k"],
            input_size=d["input_size"],
            n_outputs=d["n_outputs"],
            nodes=nodes,
            connections=conns,
            output_ids=list(d["output_ids"]),
            fitness=d.get("fitness"),
            metrics=d.get("metrics", {}),
        )


# ------------------------------------------------------------------
# Distance and crossover
# ------------------------------------------------------------------

def lut_hamming(a: Optional[List[int]], b: Optional[List[int]]) -> float:
    if a is None or b is None:
        return 0.0 if a is b else 1.0
    if len(a) != len(b):
        return 1.0
    return sum(int(x != y) for x, y in zip(a, b)) / max(1, len(a))


def genome_distance(
    a: Genome,
    b: Genome,
    c_disjoint: float = 1.0,
    c_lut: float = 0.4,
    c_type: float = 0.2,
) -> float:
    ia = set(a.connections.keys())
    ib = set(b.connections.keys())
    norm = max(1, max(len(ia), len(ib)))
    d_conn = len(ia ^ ib) / norm

    common_nodes = set(a.nodes.keys()) & set(b.nodes.keys())
    lut_terms: List[float] = []
    type_terms: List[float] = []
    for nid in common_nodes:
        na, nb = a.nodes[nid], b.nodes[nid]
        if not (na.is_active() and nb.is_active()):
            continue
        lut_terms.append(lut_hamming(na.lut, nb.lut))
        if na.kind in ("E", "I") and nb.kind in ("E", "I"):
            type_terms.append(0.0 if na.kind == nb.kind else 1.0)

    d_lut = float(np.mean(lut_terms)) if lut_terms else 0.0
    d_type = float(np.mean(type_terms)) if type_terms else 0.0

    return c_disjoint * d_conn + c_lut * d_lut + c_type * d_type


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

    child.nodes = {}
    for nid in needed:
        in_fitter = nid in fitter.nodes
        in_other = nid in other.nodes
        if in_fitter and in_other and fitter.nodes[nid].is_active():
            fn, on = fitter.nodes[nid], other.nodes[nid]
            node = rng.choice([fn, on]).clone()
            if fn.lut and on.lut and len(fn.lut) == len(on.lut):
                node.lut = [rng.choice([fa, oa]) for fa, oa in zip(fn.lut, on.lut)]
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
    return child
