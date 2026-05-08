import torch


def node_forward(
    sel: torch.Tensor,   # [B, N, k]    long  — MUX selectors (indices into pool)
    lut: torch.Tensor,   # [B, N, 2**k] long  — LUT truth table (0/1 values)
    pool: torch.Tensor,  # [B, P]       long  — candidate signal pool (binary 0/1)
) -> torch.Tensor:       # [B, N]       long
    """
    Batched Boolean node forward pass with variable arity k.

    Evaluates N nodes across a batch of B genomes simultaneously.
    Each node consists of k MUXes feeding one k-input LUT:
      1. Picks k signals from the pool using its k MUX selectors.
      2. Forms a k-bit address from those signals (MSB-first).
      3. Looks up the 2**k-entry LUT at that address to produce a 0/1 output.

    k is inferred from sel.shape[-1]; lut.shape[-1] must equal 2**k.

    No Python loops over B or N — everything is vectorised via gather.
    A small Python loop over k builds the address.

    Args:
        sel:  [B, N, k]    — for each (batch, node), k pool indices
        lut:  [B, N, 2**k] — truth table; entry i gives the output for address i
        pool: [B, P]       — binary (0/1) signal pool

    Pool indexing contract
    ----------------------
    sel values must be valid indices into pool (0 <= sel[i] < P).
    The caller is responsible for ensuring this; different node groups
    (E0-5, E6-8, I0-2) receive different pools of increasing size, so
    their sel ranges differ — see Microcircuit for details.
    """
    B, N, k = sel.shape
    assert lut.shape[-1] == 2 ** k, (
        f"lut.shape[-1]={lut.shape[-1]} must equal 2**k=2**{k}={2**k}"
    )
    BN = B * N

    # Flatten (B, N) into one dimension so we can use gather directly.
    flat_sel = sel.reshape(BN, k).long()    # [BN, k] gather requires Long indices
    flat_lut = lut.reshape(BN, 2 ** k)     # [BN, 2**k]

    # Expand pool from [B, P] to [BN, P] — each of the N nodes per
    # batch element gets its own copy of that batch element's pool.
    flat_pool = (
        pool.unsqueeze(1)          # [B, 1, P]
            .expand(-1, N, -1)     # [B, N, P]
            .reshape(BN, -1)       # [BN, P]
    )

    # MUX: gather k input bits per node using sel as column indices.
    inputs = flat_pool.gather(1, flat_sel)  # [BN, k]

    # Build k-bit LUT address (MSB = bit 0 of inputs).
    addr = sum(inputs[:, j].long() << (k - 1 - j) for j in range(k))  # [BN]
    addr = addr.unsqueeze(1)  # [BN, 1]

    # LUT lookup: gather the single output bit at the computed address.
    out = flat_lut.gather(1, addr).squeeze(1)  # [BN]

    return out.reshape(B, N)  # [B, N]
