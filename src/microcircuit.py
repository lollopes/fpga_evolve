import torch
import torch.nn as nn

from .node import node_forward


class Microcircuit(nn.Module):
    """Minimal 3-node E/I/O Boolean microcircuit."""

    def __init__(self, sel: torch.Tensor, lut: torch.Tensor) -> None:
        """
        Parameters
        ----------
        sel : [B, 3, k] long
        lut : [B, 3, 2**k] long  (values 0 or 1)
        """
        super().__init__()

        B, n_nodes, k = sel.shape
        lut_size = lut.shape[2]

        if n_nodes != 3 or lut_size != 2 ** k:
            raise ValueError(
                f"Expected sel [B, 3, k] and lut [B, 3, 2**k], "
                f"got sel {tuple(sel.shape)} and lut {tuple(lut.shape)}"
            )

        self.k = k
        self.register_buffer("sel", sel.long())
        self.register_buffer("lut", lut.long())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pool: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pool : [B, P] long

        Returns
        -------
        o : [B, 1] long
        """
        E = node_forward(self.sel[:, 0:1], self.lut[:, 0:1], pool)   # [B, 1]
        I = node_forward(self.sel[:, 1:2], self.lut[:, 1:2], pool)   # [B, 1]

        O_pool = pool.clone()
        O_pool[:, 0] = E.squeeze(1)

        O_raw = node_forward(self.sel[:, 2:3], self.lut[:, 2:3], O_pool)  # [B, 1]

        o = O_raw * (1 - I)
        return o  # [B, 1]

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def random(cls, B: int, k: int = 3, pool_size: int = 16,
               device: torch.device = None) -> "Microcircuit":
        dev = device or torch.device("cpu")
        sel = torch.randint(0, pool_size, (B, 3, k), dtype=torch.long, device=dev)
        lut = torch.randint(0, 2, (B, 3, 2 ** k), dtype=torch.long, device=dev)
        return cls(sel, lut)

    @classmethod
    def from_genome(cls, genome: torch.Tensor, k: int) -> "Microcircuit":
        """
        Parameters
        ----------
        genome : [B, 3, k + 2**k] long
        k      : number of selectors per node
        """
        lut_size = 2 ** k
        if genome.shape[1] != 3 or genome.shape[2] != k + lut_size:
            raise ValueError(
                f"Expected genome [B, 3, {k + lut_size}], got {tuple(genome.shape)}"
            )
        return cls(sel=genome[:, :, :k], lut=genome[:, :, k:])

    def to_genome(self) -> torch.Tensor:
        """Export packed genome [B, 3, k + 2**k]. Inverse of from_genome."""
        return torch.cat([self.sel, self.lut], dim=2)
