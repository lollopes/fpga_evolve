import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from src.lattice import Lattice3DNetwork

# 1. Create network
net = Lattice3DNetwork.random(Z=3, H=16, W=16, B=2, k=3, seed=0)

# 2. Shape assertions
assert tuple(net.state.shape)     == (2, 3, 16, 16, 1), f"state:     {tuple(net.state.shape)}"
assert tuple(net.layer_sel.shape) == (3, 2, 3, 3),      f"layer_sel: {tuple(net.layer_sel.shape)}"
assert tuple(net.layer_lut.shape) == (3, 2, 3, 8),      f"layer_lut: {tuple(net.layer_lut.shape)}"
print("Shape assertions passed.")

# 3. Random binary input
x = torch.randint(0, 2, (2, 5, 16, 16))

# 4. Run trajectory
traj = net.run(x)

# 5. Trajectory shape
assert tuple(traj.shape) == (2, 5, 16 * 16), f"traj: {tuple(traj.shape)}"
print("Trajectory shape assertion passed.")

# 6. Get features
features = net.get_features(x, mode="sum_final")

# 7. Feature shape
assert tuple(features.shape) == (2, 16 * 16), f"features: {tuple(features.shape)}"
print("Feature shape assertion passed.")

# 8. Diagnostics
mean_activity   = traj.float().mean().item()
silent_fraction = (traj.sum(dim=1) == 0).float().mean().item()

print(f"mean activity:     {mean_activity:.4f}")
print(f"silent fraction:   {silent_fraction:.4f}")
print(f"identity_bits shape: {tuple(net.identity_bits.shape)}")
print(f"distal_idx shape:    {tuple(net.distal_idx.shape)}")
print("All smoke tests passed.")
