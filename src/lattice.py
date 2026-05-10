"""
Lattice3DNetwork — 3D topographic-laminar Boolean spiking network.

Each lattice site (x, y, z) holds one E/I/O Boolean microcircuit module
(N_NODES=3: E excitatory aux, I inhibitory, O output) operating on a
16-entry signal pool built from spatial neighbours, recurrent self-state,
feedforward input, and fixed distal projections.

Genome layout:
  per module : [B, 3, k + 2**k]  →  sel [B, 3, k] + lut [B, 3, 2**k]
  per layer z: one shared template [B, 3, k + 2**k] applied to all H×W sites
  whole lattice: [B, Z, 3, k + 2**k]

Key design choices vs. the 2D/multi-area Column:
  • Topology is 3-D (x, y, z) instead of separate Area objects.
  • N_NODES = 3 per module: E (excitatory aux), I (inhibitory), O (output).
  • K = 1 output bit per module: output = O * (1 - I).
  • O node sees E's output substituted at P[0] of its pool (E→O coupling).
  • State tensor [B, Z, H, W, 1] replaces per-Area E_state buffers.
  • Synchronous update: pools are built from the *previous* state,
    then all modules fire simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .node import node_forward


class Lattice3DNetwork(nn.Module):
    """
    3D topographic-laminar Boolean network.

    Dimensions
    ----------
    x  ∈ [0, W)   — topographic width  (columns)
    y  ∈ [0, H)   — topographic height (rows)
    z  ∈ [0, Z)   — laminar depth      (layers)

    Genome
    ------
    One template G_z per layer z, shared by every (x, y) site in that layer.
    Stored as registered buffers:
        layer_sel  [Z, B, 3, k]     MUX selectors   (values 0–POOL_SIZE-1)
        layer_lut  [Z, B, 3, 2**k]  LUT truth tables (values 0/1)

    State
    -----
    state  [B, Z, H, W, 1]  — 1 gated output bit per module (K=1)

    Unified Pool (16 entries, same for every node inside one module)
    ----------------------------------------------------------------
    P[0]  left   neighbour activity — module(x-1, y,   z  )
    P[1]  right  neighbour activity — module(x+1, y,   z  )
    P[2]  up     neighbour activity — module(x,   y-1, z  )
    P[3]  down   neighbour activity — module(x,   y+1, z  )
    P[4]  above  neighbour activity — module(x,   y,   z-1)
    P[5]  below  neighbour activity — module(x,   y,   z+1)
    P[6]  previous self output bit (K=1)
    P[7]  feedforward / sensory input (non-zero only at z=0)
    P[8]  distal signal 0  (fixed random source assigned at construction)
    P[9]  distal signal 1  (fixed random source assigned at construction)
    P[10] identity bit 0  (fixed random bit per module location)
    P[11] identity bit 1  (fixed random bit per module location)
    P[12] x positional cue — 1 if x >= W/2, else 0
    P[13] y positional cue — 1 if y >= H/2, else 0
    P[14] reserved zero
    P[15] reserved zero

    Note: O node sees P[0] replaced by E's output (E→O coupling).
    Out-of-bounds neighbours are treated as 0.
    """

    POOL_SIZE = 16
    N_NODES   = 3   # nodes per module: E (aux), I (inhibitory), O (output)
    K         = 1   # output bits per module

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        Z: int,
        H: int,
        W: int,
        layer_genomes: list,               # Z × [B, 3, k + 2**k] packed tensors
        distal_idx: torch.Tensor = None,   # [Z, H, W, 2, 3] long, or None
        k: int = 2,                        # node arity (MUX inputs per node)
        identity_seed: int = 0,            # RNG seed for deterministic identity bits
        use_identity: bool = True,         # if False, P[10:12] forced to zero
        use_positional_cues: bool = False, # if True, P[12:13] carry x/y cues
        distal_seed: int = 0,              # RNG seed for deterministic distal sources
        use_distal: bool = True,           # if False, P[8:10] forced to zero
        # ── Random per-node pool wiring (replaces canonical pool when enabled) ──
        use_random_pool: bool = False,     # if True, each node gets a randomly wired pool
        lambda_lateral: float = 1.0,       # distance-decay for lateral (x,y) offsets
        lambda_depth: float = 0.5,         # distance-decay for laminar (z) offsets
        pool_wiring_seed: int = 0,         # RNG seed for pool wiring
        pool_idx: torch.Tensor = None,     # pre-computed [Z,H,W,N_NODES,POOL_SIZE,3], or None
    ) -> None:
        """
        Parameters
        ----------
        Z, H, W             : lattice dimensions (layers, height, width)
        layer_genomes       : list of Z tensors [B, 3, k + 2**k]; axis-2 layout
                              [sel0..sel_{k-1}, lut0..lut_{2**k-1}]
        distal_idx          : pre-computed distal source coordinates [Z,H,W,2,3]
                              where the last dim is (z,y,x). If None, random.
        k                   : node arity — number of MUX inputs per node (default 3)
        identity_seed       : integer seed for deterministic identity_bits (default 0)
        use_identity        : if False, P[10:12] are forced to zero (default True)
        use_positional_cues : if True, P[12:13] carry x/y half-plane cues (default False)
        distal_seed         : integer seed for deterministic distal sources (default 0)
        use_distal          : if False, P[8:10] are forced to zero (default True)
        use_random_pool     : if True, each node at each site gets a unique randomly wired
                              pool of POOL_SIZE sources sampled from a distance-weighted
                              direction set; replaces the canonical P[0..15] pool entirely.
                              Slot 0 = self-recurrence, slot 1 = feedforward input; the
                              remaining POOL_SIZE-2 slots are sampled without replacement.
        lambda_lateral      : exponential decay rate for lateral (|dx|+|dy|) distance;
                              larger values favour immediate neighbours (default 1.0)
        lambda_depth        : exponential decay rate for laminar |dz| distance (default 0.5)
        pool_wiring_seed    : integer seed for pool wiring RNG (default 0)
        pool_idx            : pre-computed pool wiring [Z,H,W,N_NODES,POOL_SIZE,3]; if
                              provided, skips regeneration (useful in expand())
        """
        super().__init__()

        assert len(layer_genomes) == Z, \
            f"Expected {Z} layer genomes, got {len(layer_genomes)}"

        self.Z = Z
        self.H = H
        self.W = W
        self.k = k
        self.identity_seed = identity_seed
        self.use_identity = use_identity
        self.use_positional_cues = use_positional_cues
        self.distal_seed = distal_seed
        self.use_distal = use_distal
        self.use_random_pool = use_random_pool
        self.lambda_lateral = lambda_lateral
        self.lambda_depth = lambda_depth
        self.pool_wiring_seed = pool_wiring_seed
        self.LUT_SIZE = 2 ** k
        self.NODE_GENOME_SIZE = k + self.LUT_SIZE

        B = layer_genomes[0].shape[0]
        self.B = B

        # Validate genome shapes
        for i, g in enumerate(layer_genomes):
            assert g.shape == (B, self.N_NODES, self.NODE_GENOME_SIZE), (
                f"layer_genomes[{i}] shape {tuple(g.shape)} != "
                f"({B}, {self.N_NODES}, {self.NODE_GENOME_SIZE})"
            )

        # Unpack and register genome buffers
        sels = [g[:, :, :k].to(torch.uint8) for g in layer_genomes]  # Z × [B, 3, k]
        luts = [g[:, :, k:].bool() for g in layer_genomes]           # Z × [B, 3, 2**k]
        self.register_buffer("layer_sel", torch.stack(sels, dim=0))  # [Z, B, 3, k]
        self.register_buffer("layer_lut", torch.stack(luts, dim=0))  # [Z, B, 3, 2**k]

        assert self.layer_sel.shape == (Z, B, self.N_NODES, k), (
            f"layer_sel shape {tuple(self.layer_sel.shape)} != "
            f"({Z}, {B}, {self.N_NODES}, {k})"
        )
        assert self.layer_lut.shape == (Z, B, self.N_NODES, self.LUT_SIZE), (
            f"layer_lut shape {tuple(self.layer_lut.shape)} != "
            f"({Z}, {B}, {self.N_NODES}, {self.LUT_SIZE})"
        )

        # Fixed random distal projections: 2 sources per module
        if distal_idx is None:
            distal_idx = self._init_distal_sources(
                Z, H, W, seed=distal_seed, device=layer_genomes[0].device
            )
        self.register_buffer("distal_idx", distal_idx.long())  # [Z, H, W, 2, 3]

        # Deterministic identity bits: 2 binary bits per module, seeded once, never evolved
        _g = torch.Generator()
        _g.manual_seed(identity_seed)
        identity_bits = torch.randint(0, 2, (Z, H, W, 2), dtype=torch.uint8, generator=_g).bool()
        self.register_buffer("identity_bits", identity_bits)  # [Z, H, W, 2]

        # Binary positional cues: half-plane indicator along x (P[12]) and y (P[13])
        pos_x = torch.arange(W) >= W // 2  # [W]
        pos_y = torch.arange(H) >= H // 2  # [H]
        self.register_buffer(
            "pos_cue",
            torch.stack([
                pos_x.unsqueeze(0).expand(H, W),  # [H, W]
                pos_y.unsqueeze(1).expand(H, W),  # [H, W]
            ], dim=-1).contiguous()                # [H, W, 2]
        )

        # Recurrent state, zeroed at construction (reset before each sequence)
        self.register_buffer(
            "state", torch.zeros(B, Z, H, W, self.K, dtype=torch.bool)
        )
        assert self.state.shape == (B, Z, H, W, self.K), (
            f"state shape {tuple(self.state.shape)} != "
            f"({B}, {Z}, {H}, {W}, {self.K})"
        )

        # Random per-node pool wiring (only when use_random_pool=True)
        if use_random_pool:
            dev = layer_genomes[0].device
            if pool_idx is None:
                pool_idx = self._generate_pool_wiring(
                    Z, H, W, self.N_NODES, self.POOL_SIZE,
                    lambda_lateral, lambda_depth, pool_wiring_seed, dev,
                )
            # pool_idx: [Z, H, W, N_NODES, POOL_SIZE, 3]  int32
            self.register_buffer("pool_idx", pool_idx.to(torch.int32).to(dev))
            # Precompute flat linear indices into the (Z+1)*H*W extended act tensor
            # so that step() only needs one gather, not index arithmetic per call.
            pz = pool_idx[..., 0].long()   # [Z, H, W, N_NODES, POOL_SIZE]
            py = pool_idx[..., 1].long()
            px = pool_idx[..., 2].long()
            pool_lin = (pz * H * W + py * W + px).to(torch.int32)
            # Reshape to [Z*H*W, N_NODES, POOL_SIZE] — matches the (B)*Z*H*W flattening
            # order used in step() so that reshape(N, N_NODES, POOL_SIZE) aligns with sel.
            self.register_buffer(
                "pool_lin",
                pool_lin.reshape(Z * H * W, self.N_NODES, self.POOL_SIZE).to(dev),
            )

    # ------------------------------------------------------------------
    # Pool helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_activity(output_vector: torch.Tensor) -> torch.Tensor:
        """
        Compress a K-bit output vector to a single activity bit via OR.

        output_vector : [..., K] long binary
        Returns       : [...] long binary  (1 if any bit is 1)
        """
        return output_vector.any(dim=-1)

    def build_pool(
        self,
        x: int,
        y: int,
        z: int,
        t: int,
        state: torch.Tensor,          # [B, Z, H, W, 1]
        input_sequence: torch.Tensor, # [B, T, H, W] long
    ) -> torch.Tensor:                # [B, 16]
        """
        Build the 16-entry signal pool for module at (x, y, z) at time t.

        Intended for debugging / unit tests of individual modules.
        The vectorised step() method builds all pools at once without
        calling this function (to avoid Python-level loops over sites).

        Boundary rule: any neighbour outside the lattice contributes 0.
        """
        B   = state.shape[0]
        dev = state.device

        def nbr(nx: int, ny: int, nz: int) -> torch.Tensor:
            """Return compressed activity of neighbour (nx,ny,nz) or zeros."""
            if nx < 0 or nx >= self.W \
                    or ny < 0 or ny >= self.H \
                    or nz < 0 or nz >= self.Z:
                return torch.zeros(B, dtype=torch.bool, device=dev)
            return self.get_activity(state[:, nz, ny, nx, :])

        # Feedforward: only z=0 receives the raw sensory frame
        ff = self.get_feedforward_input(x, y, z, input_sequence, t=t)

        zeros = torch.zeros(B, dtype=torch.bool, device=dev)

        # Distal sources assigned to this module
        if self.use_distal:
            ds = self.distal_idx[z, y, x]           # [2, 3]
            distal = [
                self.get_activity(state[:, ds[s, 0], ds[s, 1], ds[s, 2], :])
                for s in range(2)
            ]
        else:
            distal = [zeros, zeros]

        # Identity bits and positional cues for this module (scalar → [B])
        if self.use_identity:
            ident_b = self.identity_bits[z, y, x].unsqueeze(0).expand(B, -1)  # [B, 2]
        else:
            ident_b = torch.zeros(B, 2, dtype=torch.bool, device=dev)
        if self.use_positional_cues:
            pos_b = self.pos_cue[y, x].unsqueeze(0).expand(B, -1)             # [B, 2]
        else:
            pos_b = torch.zeros(B, 2, dtype=torch.bool, device=dev)

        pool = torch.stack([
            nbr(x - 1, y,     z    ),   # P[0]  left
            nbr(x + 1, y,     z    ),   # P[1]  right
            nbr(x,     y - 1, z    ),   # P[2]  up
            nbr(x,     y + 1, z    ),   # P[3]  down
            nbr(x,     y,     z - 1),   # P[4]  above-layer
            nbr(x,     y,     z + 1),   # P[5]  below-layer
            state[:, z, y, x, 0],       # P[6]  self bit (K=1)
            ff,                          # P[7]  feedforward
            distal[0],                   # P[8]  distal source 0
            distal[1],                   # P[9]  distal source 1
            ident_b[:, 0],               # P[10] identity bit 0
            ident_b[:, 1],               # P[11] identity bit 1
            pos_b[:, 0],                 # P[12] x positional cue
            pos_b[:, 1],                 # P[13] y positional cue
            zeros,                       # P[14] reserved
            zeros,                       # P[15] reserved
        ], dim=1)  # [B, 16]

        return pool

    # ------------------------------------------------------------------
    # Feedforward input helper
    # ------------------------------------------------------------------

    def get_feedforward_input(
        self,
        x: int,
        y: int,
        z: int,
        input_sequence: torch.Tensor,  # [B, T, H, W] or [B, H, W]
        t: int = None,
    ) -> torch.Tensor:                 # [B] binary
        """
        Return the feedforward input for module (x, y, z) at time t.

        Only z=0 receives direct sensory input (the pixel at position y, x).
        Layers z > 0 receive zero; lateral feedforward (e.g. OR-pooled signal
        from z-1) can be added here later without changing step().

        input_sequence : [B, T, H, W] if t is given, else [B, H, W] (single frame)
        """
        if z > 0:
            B = self.state.shape[0]
            return torch.zeros(B, dtype=torch.bool, device=self.state.device)
        if t is not None:
            return input_sequence[:, t, y, x].bool()   # [B]
        return input_sequence[:, y, x].bool()    # [B] (single-frame path)

    # ------------------------------------------------------------------
    # Synchronous step (vectorised over all sites)
    # ------------------------------------------------------------------

    def step(self, input_t: torch.Tensor) -> torch.Tensor:
        """
        Run one synchronous update across the entire lattice.

        All pools are built from self.state (the previous timestep);
        E and I nodes fire in parallel; O node fires after E (E→O coupling);
        I gating produces K=1 output bit; self.state is updated in-place.

        When use_random_pool=True each node uses its own randomly wired pool
        (built by a single gather from the precomputed pool_lin indices).
        When use_random_pool=False the original canonical pool is used.

        input_t : [B, H, W] long — binary sensory frame for z=0.
        Returns : new state [B, Z, H, W, 1]
        """
        input_t = input_t.bool()
        state = self.state  # [B, Z, H, W, 1]
        B, Z, H, W, _ = state.shape
        dev = state.device
        N = B * Z * H * W

        # ---- 1. Compressed activity map --------------------------------
        act = state[..., 0]  # [B, Z, H, W]

        # ---- 2. Expand layer genomes over all H×W sites ----------------
        # layer_sel: [Z, B, N_NODES, k] → sel: [N, N_NODES, k]
        sel = (
            self.layer_sel              # [Z, B, 3, k]
            .permute(1, 0, 2, 3)        # [B, Z, 3, k]
            [:, :, None, None, :, :]    # [B, Z, 1, 1, 3, k]
            .expand(-1, -1, H, W, -1, -1)  # [B, Z, H, W, 3, k]
            .contiguous()
            .reshape(N, self.N_NODES, self.k)
        )
        lut = (
            self.layer_lut              # [Z, B, 3, 2**k]
            .permute(1, 0, 2, 3)        # [B, Z, 3, 2**k]
            [:, :, None, None, :, :]    # [B, Z, 1, 1, 3, 2**k]
            .expand(-1, -1, H, W, -1, -1)  # [B, Z, H, W, 3, 2**k]
            .contiguous()
            .reshape(N, self.N_NODES, self.LUT_SIZE)
        )

        if self.use_random_pool:
            # ---- Random per-node pool path --------------------------------
            # Append feedforward input as virtual layer Z so that pool_idx
            # entries with z_src=Z resolve to the sensory input via one gather.
            # act_ext: [B, Z+1, H, W]  (uint8 for gather compatibility)
            act_ext = torch.cat(
                [act, input_t.unsqueeze(1)], dim=1
            ).to(torch.uint8)
            act_flat = act_ext.reshape(B, (Z + 1) * H * W)  # [B, (Z+1)*HW]

            # pool_lin: [Z*H*W, N_NODES, POOL_SIZE] — precomputed flat indices
            lin   = self.pool_lin.long().reshape(-1)          # [ZHW*N_NODES*PS]
            lin_b = lin.unsqueeze(0).expand(B, -1)            # [B, ZHW*N_NODES*PS]
            pools = act_flat.gather(1, lin_b)                 # [B, ZHW*N_NODES*PS]
            pools = (
                pools
                .reshape(B, Z, H, W, self.N_NODES, self.POOL_SIZE)
                .reshape(N, self.N_NODES, self.POOL_SIZE)
                .bool()
            )  # [N, N_NODES, POOL_SIZE]

            # Fire E and I on their own per-node pools
            out_E = node_forward(sel[:, 0:1], lut[:, 0:1], pools[:, 0])  # [N, 1]
            out_I = node_forward(sel[:, 1:2], lut[:, 1:2], pools[:, 1])  # [N, 1]

            # E→O coupling: override O's slot 0 with E's current output
            O_pool = pools[:, 2].clone()
            O_pool[:, 0] = out_E.squeeze(1)
            O_raw = node_forward(sel[:, 2:3], lut[:, 2:3], O_pool)       # [N, 1]

        else:
            # ---- Original canonical pool path ----------------------------

            # ---- 3. Lateral neighbours via zero-padding ------------------
            padded_hw = F.pad(act.float(), (1, 1, 1, 1))        # [B, Z, H+2, W+2]
            padded_z  = F.pad(act.float(), (0, 0, 0, 0, 1, 1))  # [B, Z+2, H,   W]

            left  = padded_hw[:, :, 1:-1, :-2].bool()
            right = padded_hw[:, :, 1:-1, 2: ].bool()
            up    = padded_hw[:, :, :-2, 1:-1].bool()
            down  = padded_hw[:, :, 2:,  1:-1].bool()
            above = padded_z[:, :-2].bool()
            below = padded_z[:, 2: ].bool()

            # ---- 4. Feedforward: z=0 gets input_t, z>0 gets zero ---------
            ff = torch.zeros(B, Z, H, W, dtype=torch.bool, device=dev)
            ff[:, 0] = input_t

            # ---- 5. Distal signals ----------------------------------------
            if self.use_distal:
                dist = self._gather_distal(act)   # [B, Z, H, W, 2]
            else:
                dist = torch.zeros(B, Z, H, W, 2, dtype=torch.bool, device=dev)

            # ---- 6. Assemble canonical pool [B, Z, H, W, 16] ---------------
            if self.use_identity:
                identity = self.identity_bits.unsqueeze(0).expand(B, -1, -1, -1, -1)
            else:
                identity = torch.zeros(B, Z, H, W, 2, dtype=torch.bool, device=dev)
            if self.use_positional_cues:
                pos = self.pos_cue.unsqueeze(0).unsqueeze(0).expand(B, Z, -1, -1, -1)
            else:
                pos = torch.zeros(B, Z, H, W, 2, dtype=torch.bool, device=dev)
            zeros2 = torch.zeros(B, Z, H, W, 2, dtype=torch.bool, device=dev)
            pool = torch.cat([
                torch.stack([left, right, up, down, above, below,  # P[0-5]
                             state[..., 0],                        # P[6] self bit
                             ff,                                   # P[7]
                             dist[..., 0], dist[..., 1]], dim=-1), # P[8-9]
                identity,                                          # P[10-11]
                pos,                                               # P[12-13]
                zeros2,                                            # P[14-15]
            ], dim=-1)  # [B, Z, H, W, 16]

            flat_pool = pool.reshape(N, self.POOL_SIZE)

            out_E = node_forward(sel[:, 0:1], lut[:, 0:1], flat_pool)  # [N, 1]
            out_I = node_forward(sel[:, 1:2], lut[:, 1:2], flat_pool)  # [N, 1]
            O_pool = flat_pool.clone()
            O_pool[:, 0] = out_E.squeeze(1)
            O_raw = node_forward(sel[:, 2:3], lut[:, 2:3], O_pool)     # [N, 1]

        # ---- I gating → K=1 output bit per module ---------------------
        o = O_raw & (~out_I)  # [N, 1]

        new_state = o.reshape(B, Z, H, W, 1)
        self.state.copy_(new_state)

        self.last_E     = out_E.reshape(B, Z, H, W).detach()
        self.last_I     = out_I.reshape(B, Z, H, W).detach()
        self.last_O_raw = O_raw.reshape(B, Z, H, W).detach()

        return self.state

    # ------------------------------------------------------------------
    # Sequence runner
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Zero the recurrent lattice state (call before each new sequence)."""
        self.state.zero_()

    def run(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Run a T-step input sequence and collect the final-layer readout (K=1).

        input_sequence : [B, T, H, W] long — binary spike frames.
        Returns        : trajectory [B, T, H*W] long
                         Flattened spatial output of layer z=Z-1 at each step.
                         Pass to a separate linear readout for classification.
        """
        B, T, H, W = input_sequence.shape
        self.reset_state()
        outputs = []
        for t in range(T):
            state = self.step(input_sequence[:, t])       # [B, Z, H, W, 1]
            readout = state[:, -1, :, :, 0].reshape(B, -1)  # [B, H*W]
            outputs.append(readout)
        return torch.stack(outputs, dim=1)                # [B, T, H*W]

    def get_features(
        self,
        input_sequence: torch.Tensor,
        mode: str = "sum_final",
    ) -> torch.Tensor:
        """
        Run input_sequence and return a summary feature vector from the final z layer.

        Parameters
        ----------
        input_sequence : [B, T, H, W] long — binary spike frames
        mode           : "sum_final"  → sum over T → [B, H*W]  float
                         "last_final" → last step  → [B, H*W]  float

        Returns
        -------
        features : [B, H*W] float
        """
        trajectory = self.run(input_sequence)   # [B, T, H*W]  (K=1)
        if mode == "sum_final":
            return trajectory.sum(dim=1).float()
        elif mode == "last_final":
            return trajectory[:, -1].float()
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    # ------------------------------------------------------------------
    # Genome helpers
    # ------------------------------------------------------------------

    def decode_layer_genomes(self, genome_pack: torch.Tensor) -> list:
        """
        Split a packed [B, Z, 3, k + 2**k] tensor into a list of Z tensors [B, 3, k + 2**k].

        genome_pack : [B, Z, 3, k + 2**k]  — 3 nodes, variable k
        Returns     : list of Z tensors, each [B, 3, k + 2**k]
        """
        B, Z, N, G = genome_pack.shape
        assert Z == self.Z, \
            f"Expected Z={self.Z} layer slices, got {Z}"
        assert N == self.N_NODES, \
            f"Expected N_NODES={self.N_NODES} (3), got {N}"
        assert G == self.NODE_GENOME_SIZE, \
            f"Expected genome size k+2**k={self.NODE_GENOME_SIZE} (k={self.k}), got {G}"
        return [genome_pack[:, z] for z in range(self.Z)]

    def to_genomes(self) -> torch.Tensor:
        """
        Export all layer genomes as a packed [B, Z, 3, k + 2**k] tensor.
        Inverse of decode_layer_genomes.
        """
        # Permute stored buffers from [Z, B, 3, *] to [B, Z, 3, *], then cat on last dim.
        sel = self.layer_sel.permute(1, 0, 2, 3)  # [B, Z, 3, k]
        lut = self.layer_lut.permute(1, 0, 2, 3).to(torch.uint8)  # [B, Z, 3, 2**k]
        return torch.cat([sel, lut], dim=-1)        # [B, Z, 3, k + 2**k]

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        Z: int,
        H: int,
        W: int,
        B: int,
        k: int = 2,
        seed: int = 0,
        device: torch.device = None,
    ) -> "Lattice3DNetwork":
        """
        Randomly initialise a Lattice3DNetwork with B independent genomes.

        For each layer z (seeded at seed+z):
            sel    [B, 3, k]      uniform in [0, 16)
            lut    [B, 3, 2**k]   uniform in {0, 1}
            genome = cat([sel, lut], dim=-1)  →  [B, 3, k + 2**k]

        genome_pack shape: [B, Z, 3, k + 2**k]
            k=3 → [B, Z, 3, 11]
        """
        dev = device or torch.device("cpu")
        layer_genomes = []
        for z in range(Z):
            g = torch.Generator(device=dev)
            g.manual_seed(seed + z)
            sel = torch.randint(0, 16,      (B, 3, k),      dtype=torch.uint8, device=dev, generator=g)
            lut = torch.randint(0, 2,       (B, 3, 2 ** k), dtype=torch.uint8, device=dev, generator=g)
            layer_genomes.append(torch.cat([sel, lut], dim=-1))  # [B, 3, k + 2**k]
        return cls(Z, H, W, layer_genomes, k=k)

    def expand(self, N: int) -> "Lattice3DNetwork":
        """
        Return a new lattice with each layer genome repeated N times, fresh state.

        Each layer genome [1, 3, k+2**k] is broadcast to [N, 3, k+2**k] (B must be 1).
        Preserves: distal_idx, identity_bits, ablation flags (use_identity /
        use_positional_cues / use_distal), k, K=1.
        """
        new_genomes = [
            torch.cat([
                self.layer_sel[z].expand(N, -1, -1),  # [N, 3, k]
                self.layer_lut[z].expand(N, -1, -1).to(self.layer_sel.dtype),  # [N, 3, 2**k]
            ], dim=-1).contiguous()                    # [N, 3, k + 2**k]
            for z in range(self.Z)
        ]
        net = Lattice3DNetwork(
            self.Z, self.H, self.W, new_genomes,
            distal_idx=self.distal_idx,
            k=self.k,
            identity_seed=self.identity_seed,
            use_identity=self.use_identity,
            use_positional_cues=self.use_positional_cues,
            distal_seed=self.distal_seed,
            use_distal=self.use_distal,
            use_random_pool=self.use_random_pool,
            lambda_lateral=self.lambda_lateral,
            lambda_depth=self.lambda_depth,
            pool_wiring_seed=self.pool_wiring_seed,
            pool_idx=self.pool_idx if self.use_random_pool else None,
        )
        net.identity_bits.copy_(self.identity_bits)  # preserve exact bits, not just seed
        return net

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_pool_wiring(
        Z: int,
        H: int,
        W: int,
        N_NODES: int,
        POOL_SIZE: int,
        lambda_lateral: float,
        lambda_depth: float,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate random per-node pool wiring for the full lattice.

        For each site (z, y, x) and each node n ∈ {E, I, O}, assigns POOL_SIZE
        source locations drawn from a distance-weighted direction set:

          slot 0  — self  (z, y, x) — guaranteed self-recurrence
          slot 1  — feedforward input sentinel (z = Z) — always on the sensory path
          slots 2..POOL_SIZE-1 — sampled without replacement from 40 canonical offsets
                                  spanning Groups B-E (local → diagonal inter-layer),
                                  weighted by exp(-λ_lat*(|dy|+|dx|) − λ_dep*|dz|)

        Toroidal boundary conditions are applied on all three axes so every direction
        resolves to a valid lattice site regardless of position.

        The feedforward sentinel (z = Z) is treated in step() by prepending input_t
        as a virtual layer Z of the activity tensor before the gather.

        Returns
        -------
        pool_idx : [Z, H, W, N_NODES, POOL_SIZE, 3]  int32 on CPU
                   Last dim: (z_src, y_src, x_src).  z_src = Z → feedforward.
        """
        import math

        # ── Direction set ────────────────────────────────────────────────────────
        # Group B: radius-1 Moore neighbourhood, same layer (8 directions)
        directions = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                directions.append((0, dy, dx))

        # Group C: radius-2 extended, same layer — cardinal-2 + knight-like (12)
        for dy, dx in [
            (0, 2), (0, -2), (2, 0), (-2, 0),
            (1, 2), (-1, 2), (1, -2), (-1, -2),
            (2, 1), (-2, 1), (2, -1), (-2, -1),
        ]:
            directions.append((0, dy, dx))

        # Group D: direct inter-layer, no lateral offset (4)
        for dz in (-1, 1, -2, 2):
            directions.append((dz, 0, 0))

        # Group E: diagonal inter-layer — z±1 combined with radius-1 lateral (16)
        for dz in (-1, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    directions.append((dz, dy, dx))

        # n_dirs = 8 + 12 + 4 + 16 = 40
        n_dirs = len(directions)
        n_sample = POOL_SIZE - 2          # slots filled by sampling

        if n_dirs < n_sample:
            raise ValueError(
                f"Direction set ({n_dirs}) is smaller than n_sample ({n_sample}); "
                "reduce POOL_SIZE or extend the direction set."
            )

        # ── Per-direction sampling weights ───────────────────────────────────────
        weights = torch.tensor(
            [
                math.exp(
                    -lambda_lateral * (abs(dy) + abs(dx)) - lambda_depth * abs(dz)
                )
                for dz, dy, dx in directions
            ],
            dtype=torch.float32,
        )
        weights = weights / weights.sum()

        dir_tensor = torch.tensor(directions, dtype=torch.int32)  # [n_dirs, 3]

        # ── Site coordinates [n_sites, 3] ────────────────────────────────────────
        n_sites = Z * H * W * N_NODES
        zs = torch.arange(Z).view(Z, 1, 1, 1).expand(Z, H, W, N_NODES).reshape(-1)
        ys = torch.arange(H).view(1, H, 1, 1).expand(Z, H, W, N_NODES).reshape(-1)
        xs = torch.arange(W).view(1, 1, W, 1).expand(Z, H, W, N_NODES).reshape(-1)

        # ── Sample n_sample directions per site, without replacement ─────────────
        rng = torch.Generator()           # CPU generator for reproducibility
        rng.manual_seed(seed)
        weights_batch = weights.unsqueeze(0).expand(n_sites, -1)  # [n_sites, n_dirs]
        sampled_dir_idx = torch.multinomial(
            weights_batch, n_sample, replacement=False, generator=rng
        )  # [n_sites, n_sample]

        # ── Resolve directions to absolute coords with toroidal wrapping ─────────
        offsets = dir_tensor[sampled_dir_idx.reshape(-1)].reshape(n_sites, n_sample, 3)
        z_src = (zs.unsqueeze(1) + offsets[:, :, 0]) % Z   # [n_sites, n_sample]
        y_src = (ys.unsqueeze(1) + offsets[:, :, 1]) % H
        x_src = (xs.unsqueeze(1) + offsets[:, :, 2]) % W
        sampled_coords = torch.stack([z_src, y_src, x_src], dim=2)  # [n_sites, n_sample, 3]

        # ── Mandatory slots ──────────────────────────────────────────────────────
        site_coords = torch.stack([zs, ys, xs], dim=1)     # [n_sites, 3]
        slot_self = site_coords.unsqueeze(1)                # [n_sites, 1, 3] — self-recurrence
        slot_ff   = site_coords.unsqueeze(1).clone()
        slot_ff[:, 0, 0] = Z                                # sentinel: z=Z → feedforward

        # ── Assemble full pool_idx ───────────────────────────────────────────────
        pool_idx = torch.cat([slot_self, slot_ff, sampled_coords], dim=1)
        # pool_idx: [n_sites, POOL_SIZE, 3]
        assert pool_idx.shape == (n_sites, POOL_SIZE, 3)

        return pool_idx.reshape(Z, H, W, N_NODES, POOL_SIZE, 3).to(torch.int32)

    @staticmethod
    def _init_distal_sources(
        Z: int, H: int, W: int, seed: int, device=None
    ) -> torch.Tensor:
        """
        Assign 2 fixed distal projection sources per module.

        Returns [Z, H, W, 2, 3] where last dim is (z, y, x) coordinates.
        Sources are valid lattice indices; deterministic for the same seed.
        Not evolved, not learned — registered as a buffer by the caller.
        """
        dev = device or torch.device("cpu")
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        n = Z * H * W
        zs = torch.randint(0, max(Z, 1), (n, 2), generator=g, device=dev)
        ys = torch.randint(0, max(H, 1), (n, 2), generator=g, device=dev)
        xs = torch.randint(0, max(W, 1), (n, 2), generator=g, device=dev)
        coords = torch.stack([zs, ys, xs], dim=-1)  # [n, 2, 3]
        return coords.reshape(Z, H, W, 2, 3)

    def _gather_distal(self, act: torch.Tensor) -> torch.Tensor:
        """
        Read compressed activity from the 2 fixed distal sources of every module.

        act        : [B, Z, H, W]  single output bit per module (K=1)
        distal_idx : [Z, H, W, 2, 3]  (z, y, x) coords of the 2 sources
        Returns    : [B, Z, H, W, 2]
        """
        B, Z, H, W = act.shape

        # Flatten act to 1-D spatial index: [B, Z*H*W]
        act_flat = act.reshape(B, Z * H * W)

        # Linear index into the flat spatial dim: z*H*W + y*W + x
        d   = self.distal_idx           # [Z, H, W, 2, 3]
        lin = (
            d[..., 0] * H * W          # z offset
            + d[..., 1] * W            # y offset
            + d[..., 2]                # x offset
        ).reshape(Z * H * W, 2)        # [ZHW, 2]

        dist = []
        for s in range(2):
            idx = lin[:, s].unsqueeze(0).expand(B, -1)  # [B, ZHW]
            dist.append(act_flat.gather(1, idx))          # [B, ZHW]

        return torch.stack(dist, dim=-1).reshape(B, Z, H, W, 2)
