fpga_evolve
===========

A research scaffold for evolving 3D lattices of tiny Boolean microcircuits
("FPGA-style" arrays of MUXes + LUTs) and using them as a Boolean spiking
reservoir on event-camera input (N-MNIST). Includes a PyVista + trame
browser viewer for inspecting and live-editing the lattice.

There is NO evolutionary loop in tree yet. The genome encoding, the batched
forward pass, the dataset loader, and the interactive viewer are. The
evolutionary search is the next thing to wire on top of this.


Repo layout
-----------

  pyproject.toml             project metadata, deps: pyvista, tonic, torch
  uv.lock                    uv lockfile
  .python-version            >= 3.12

  src/
    node.py                  Atom: one batched Boolean node (MUXes + LUT)
    microcircuit.py          Module: 3-node E/I/O microcircuit
    lattice.py               Lattice3DNetwork: Z x H x W grid of modules
    dataset.py               N-MNIST loader via tonic
    render_env.py            PyVista/VTK offscreen env setup

  plot_lattice.py            PyVista scene builder for the 3D lattice
  lattice_server.py          trame web app: 3D viewer + live editor + raster
  smoke_test_minimal_3d.py   shape/run smoke test for Lattice3DNetwork


Core data flow
--------------

1. Atom -- src/node.py :: node_forward
   One batched Boolean node = k MUXes that pick k signals from a P-entry
   pool, concatenate them into a k-bit address (MSB first), then look up a
   2**k-entry LUT to emit one output bit. Fully vectorised over batch B and
   node count N (no Python loops over B or N; one short loop over k builds
   the address).

   Inputs:
     sel  [B, N, k]    long  -- pool indices for each MUX
     lut  [B, N, 2**k] long  -- truth tables (0/1)
     pool [B, P]       long  -- candidate signals
   Output:
     [B, N] long

2. Module -- src/microcircuit.py :: Microcircuit
   Three nodes per site:
     E (excitatory aux), I (inhibitory), O (output).
   Coupling rule: O sees E's output substituted at pool[0] (E -> O).
   Gating rule:   final output = O_raw * (1 - I).
   Genome per module: [B, 3, k + 2**k].

3. Lattice -- src/lattice.py :: Lattice3DNetwork
   A Z x H x W grid, one microcircuit at each site (x, y, z).
   One shared genome template per layer z, applied to every (x, y) in that
   layer (topographic-laminar weight sharing).

   Per-module 16-entry pool:
     P[0..5]   six neighbours: left, right, up, down, above, below
               (out-of-bounds = 0)
     P[6]      self bit (previous output, K=1)
     P[7]      feedforward sensory input (only at z=0)
     P[8..9]   two fixed random "distal" sources (deterministic per seed)
     P[10..11] two fixed random identity bits per (z, y, x)
     P[12..13] x/y half-plane positional cues (optional)
     P[14..15] reserved zeros

   Update is synchronous: all pools are built from the previous state, then
   every site fires in parallel inside one vectorised step().

   Genome shapes:
     per layer:    [B, 3, k + 2**k]
     full lattice: [B, Z, 3, k + 2**k]   (this is what gets evolved)

   Recurrent state:
     [B, Z, H, W, 1]    (K = 1 output bit per module)

   Public API:
     Lattice3DNetwork.random(Z, H, W, B, k=2, seed=0)
     net.step(input_t)                  one synchronous timestep
     net.run(input_sequence)            -> trajectory [B, T, H*W]
     net.get_features(seq, mode=...)    -> [B, H*W] readout from layer Z-1
     net.to_genomes()                   pack to [B, Z, 3, k + 2**k]
     net.decode_layer_genomes(pack)     unpack back to per-layer list
     net.expand(N)                      broadcast B=1 -> B=N (fresh state)
     net.reset_state()


Dataset
-------

src/dataset.py :: load_nmnist
  Uses tonic's NMNIST (first_saccade_only=True by default), bins events
  into n_time_bins frames with ToFrame, merges polarities, optionally
  pools to grid_size x grid_size, and binarises.

  Tasks:  7_vs_rest | 0_vs_1 | 0_vs_8 | 10class
  Output: X_train, y_train, X_val, y_val (torch.long tensors)
          X has shape [N, T, F] where F = grid_size**2 (or 34*34).


3D viewer
---------

plot_lattice.py builds the PyVista scene:
  - one sphere per module, three small spheres per module for E/I/O,
  - 13 connection kinds rendered as tubes + arrows:
      lateral, vertical_up, vertical_down, self, feedforward,
      distal0, distal1, identity0, identity1,
      positional_x, positional_y, zero, internal,
  - optional input-neuron grid drawn below z=0.

lattice_server.py wraps that scene in a trame Vue3 web app with side
panels:
  - Simulation panel: Play / Pause / Step / Reset, plus a clickable
    H x W input-spike grid (toggle a pixel to inject a 1 next step).
  - Selectors panel: per (z, node, mux) bit-flip buttons that rewrite
    net.layer_sel in place and redraw the affected connections.
  - LUT bits panel: per (z, node, address) toggle buttons that rewrite
    net.layer_lut in place.
  - Raster panel: pick (z, y, x), see E / I / O spike trains for the
    last 80 steps as a matplotlib raster image.

src/render_env.py sets PYVISTA_OFF_SCREEN, VTK_DEFAULT_RENDER_WINDOW_-
OFFSCREEN, LIBGL_ALWAYS_SOFTWARE, MESA_LOADER_DRIVER_OVERRIDE before
PyVista / VTK is imported, so the same code runs on a server with no GL
display.


Smoke test
----------

smoke_test_minimal_3d.py builds a Z=3, H=W=16, B=2, k=3 random network,
asserts the genome / state / trajectory / feature shapes, runs 5 steps
of random binary input, and prints mean activity and silent fraction.
This is the fastest way to confirm the install works:

    python smoke_test_minimal_3d.py


Install
-------

Python >= 3.12. Dependencies are declared in pyproject.toml:
    pyvista >= 0.48.1
    tonic   >= 1.6.0
    torch   >= 2.11.0

For the web viewer you also need trame:
    pip install trame trame-vuetify trame-vtk

Run the viewer (from repo root):
    python lattice_server.py
    # then open http://127.0.0.1:8080 (or pass --open-browser)


What's missing / next
---------------------

- An actual evolutionary search loop. Building blocks are ready:
  Lattice3DNetwork.random for init, to_genomes / decode_layer_genomes
  for variation operators, get_features for fitness on N-MNIST.
- A linear (or other) readout head trained on get_features outputs.
- Persistence: saving / loading evolved genomes as plain tensors.
