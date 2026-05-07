"""
_parity_test.py — verify the JS port matches the reference torch step().

Loads viewer/lattice_scene.js, reconstructs an equivalent Lattice3DNetwork,
runs both implementations on random inputs, and asserts every output bit and
every E/I/O intermediate matches at every step.

Run from repo root:

    python viewer/_parity_test.py
"""

import json
import re
import sys
from pathlib import Path

import torch

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.append(str(REPO_ROOT))

from src.lattice import Lattice3DNetwork


def load_scene(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"window\.LATTICE_DATA\s*=\s*(\{.*\});\s*$", text, flags=re.DOTALL)
    if not m:
        raise SystemExit(f"could not parse scene from {path}")
    return json.loads(m.group(1))


def js_style_step(scene: dict, state, input_t, last_E, last_I, last_O):
    """Mirror of the JS stepOnce(): per-site loops, no torch."""
    Z, H, W = scene["Z"], scene["H"], scene["W"]
    k        = scene["k"]
    POOL     = scene["pool_size"]
    layer_sel    = scene["layer_sel"]
    layer_lut    = scene["layer_lut"]
    distal       = scene["distal_idx"]
    identity     = scene["identity_bits"]
    use_distal   = scene["use_distal"]
    use_identity = scene["use_identity"]
    use_pos      = scene["use_positional_cues"]

    new_state = [[[0] * W for _ in range(H)] for _ in range(Z)]

    def eval_node(z, n, pool):
        addr = 0
        for s in range(k):
            midx = layer_sel[z][n][s]
            addr |= (pool[midx] & 1) << (k - 1 - s)
        return layer_lut[z][n][addr] & 1

    for z in range(Z):
        for y in range(H):
            for x in range(W):
                pool = [0] * POOL
                pool[0] = state[z][y][x - 1] if x - 1 >= 0 else 0
                pool[1] = state[z][y][x + 1] if x + 1 < W  else 0
                pool[2] = state[z][y - 1][x] if y - 1 >= 0 else 0
                pool[3] = state[z][y + 1][x] if y + 1 < H  else 0
                pool[4] = state[z - 1][y][x] if z - 1 >= 0 else 0
                pool[5] = state[z + 1][y][x] if z + 1 < Z  else 0
                pool[6] = state[z][y][x]
                pool[7] = input_t[y][x] if z == 0 else 0

                if use_distal:
                    d0 = distal[z][y][x][0]   # (z, y, x)
                    d1 = distal[z][y][x][1]
                    pool[8] = state[d0[0]][d0[1]][d0[2]]
                    pool[9] = state[d1[0]][d1[1]][d1[2]]
                if use_identity:
                    pool[10] = identity[z][y][x][0]
                    pool[11] = identity[z][y][x][1]
                if use_pos:
                    pool[12] = 1 if x >= W // 2 else 0
                    pool[13] = 1 if y >= H // 2 else 0

                outE = eval_node(z, 0, pool)
                outI = eval_node(z, 1, pool)
                pool[0] = outE
                outO = eval_node(z, 2, pool)

                last_E[z][y][x] = outE
                last_I[z][y][x] = outI
                last_O[z][y][x] = outO
                new_state[z][y][x] = outO * (1 - outI)

    return new_state


def main() -> None:
    SCENE_PATH = HERE / "lattice_scene.js"
    SEED       = 0
    T          = 8

    scene = load_scene(SCENE_PATH)
    Z, H, W, k = scene["Z"], scene["H"], scene["W"], scene["k"]

    torch.manual_seed(SEED)
    net = Lattice3DNetwork.random(Z=Z, H=H, W=W, B=1, k=k, seed=SEED)
    net.use_identity        = scene["use_identity"]
    net.use_positional_cues = scene["use_positional_cues"]
    net.use_distal          = scene["use_distal"]

    # Sanity check: the genome and indices we exported actually match the rebuilt net.
    assert net.layer_sel[:, 0].tolist() == scene["layer_sel"], "layer_sel mismatch"
    assert net.layer_lut[:, 0].tolist() == scene["layer_lut"], "layer_lut mismatch"
    assert net.distal_idx.tolist()      == scene["distal_idx"], "distal_idx mismatch"
    assert net.identity_bits.tolist()   == scene["identity_bits"], "identity_bits mismatch"
    print("scene <-> net consistency: OK")

    state_js = [[[0] * W for _ in range(H)] for _ in range(Z)]
    last_E_js = [[[0] * W for _ in range(H)] for _ in range(Z)]
    last_I_js = [[[0] * W for _ in range(H)] for _ in range(Z)]
    last_O_js = [[[0] * W for _ in range(H)] for _ in range(Z)]

    rng = torch.Generator(); rng.manual_seed(123)
    net.reset_state()
    for t in range(T):
        inp = torch.randint(0, 2, (1, H, W), generator=rng, dtype=torch.long)
        inp_list = inp[0].tolist()

        net.step(inp)
        state_js = js_style_step(scene, state_js, inp_list, last_E_js, last_I_js, last_O_js)

        ref_state = net.state[0, :, :, :, 0].tolist()
        ref_E     = net.last_E[0].tolist()
        ref_I     = net.last_I[0].tolist()
        ref_O     = net.last_O_raw[0].tolist()

        assert ref_state == state_js,  f"state mismatch at t={t}"
        assert ref_E     == last_E_js, f"E mismatch at t={t}"
        assert ref_I     == last_I_js, f"I mismatch at t={t}"
        assert ref_O     == last_O_js, f"O mismatch at t={t}"
        print(f"  t={t}: ok")

    print(f"all {T} steps match (Z={Z} H={H} W={W} k={k}).")


if __name__ == "__main__":
    main()
