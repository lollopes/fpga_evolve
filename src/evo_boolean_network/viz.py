"""
viz.py — Visualization utilities for the Boolean network simulator.

All functions accept a trajectory array of shape (T+1, n_nodes) as returned
by EvoBooleanNetwork.run(return_trajectory=True).

Each node is treated as a binary "neuron": output=1 is a spike, output=0 is silence.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


def plot_spike_raster(
    trajectory: np.ndarray,
    nodes: Optional[list[int]] = None,
    ax=None,
    title: str = "Spike raster",
    node_labels: Optional[list[str]] = None,
    markersize: float = 6.0,
    color: str = "black",
    show: bool = True,
):
    """
    Draw a spike raster for selected nodes over time.

    Each row is one node. A vertical tick mark is drawn at every timestep
    where that node's output is 1.

    Parameters
    ----------
    trajectory : np.ndarray of shape (T+1, n_nodes)
        As returned by EvoBooleanNetwork.run(return_trajectory=True).
        Row 0 is the initial state, rows 1..T are after each clock step.
    nodes : list[int] | None
        Which node indices to display. Defaults to all nodes.
    ax : matplotlib.axes.Axes | None
        Axes to draw on. If None, a new figure is created.
    title : str
        Plot title.
    node_labels : list[str] | None
        Y-axis tick labels, one per entry in `nodes`.
        Defaults to "node {i}".
    markersize : float
        Size of the spike tick marks.
    color : str
        Color of the spike marks.
    show : bool
        If True, call plt.show() at the end.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    trajectory = np.asarray(trajectory)
    n_steps_plus_one, n_nodes = trajectory.shape

    if nodes is None:
        nodes = list(range(n_nodes))

    if node_labels is None:
        node_labels = [f"node {i}" for i in nodes]

    if len(node_labels) != len(nodes):
        raise ValueError("node_labels must have the same length as nodes")

    if ax is None:
        fig_height = max(2.5, 0.4 * len(nodes))
        _, ax = plt.subplots(figsize=(10, fig_height))

    timesteps = np.arange(n_steps_plus_one)

    for row_idx, node_id in enumerate(nodes):
        spike_times = timesteps[trajectory[:, node_id] == 1]
        # eventplot draws one horizontal row of ticks per neuron
        ax.eventplot(
            spike_times,
            lineoffsets=row_idx,
            linelengths=0.6,
            linewidths=markersize * 0.15,
            colors=color,
        )

    ax.set_xlim(-0.5, n_steps_plus_one - 0.5)
    ax.set_ylim(-0.5, len(nodes) - 0.5)
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(node_labels, fontsize=9)
    ax.set_xlabel("Timestep (clock cycle)")
    ax.set_ylabel("Node")
    ax.set_title(title)
    ax.invert_yaxis()   # top row = first node, matching matrix convention

    # Light vertical grid lines to help read timesteps
    ax.set_xticks(timesteps)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_raster_grid(
    trajectories: list[np.ndarray],
    input_labels: list[str],
    nodes: Optional[list[int]] = None,
    node_labels: Optional[list[str]] = None,
    suptitle: str = "Spike raster per input",
    markersize: float = 6.0,
    show: bool = True,
):
    """
    Draw one raster subplot per input sample, stacked vertically.

    Useful for seeing how the network responds differently to each input
    in a task dataset (e.g. the 4 rows of the XOR truth table).

    Parameters
    ----------
    trajectories : list of np.ndarray
        One trajectory per input sample, each shape (T+1, n_nodes).
    input_labels : list[str]
        One label per trajectory shown as the subplot title.
    nodes : list[int] | None
        Which nodes to display. Defaults to all.
    node_labels : list[str] | None
        Y-axis labels. Defaults to "node {i}".
    suptitle : str
        Figure-level title.
    markersize : float
    show : bool

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt

    n_plots = len(trajectories)
    n_nodes_total = trajectories[0].shape[1]

    if nodes is None:
        nodes = list(range(n_nodes_total))
    if node_labels is None:
        node_labels = [f"node {i}" for i in nodes]

    row_height = max(1.5, 0.35 * len(nodes))
    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(10, row_height * n_plots),
        sharex=True,
        sharey=True,
    )

    # Make axes always iterable even for n_plots=1
    if n_plots == 1:
        axes = [axes]

    for ax, traj, label in zip(axes, trajectories, input_labels):
        plot_spike_raster(
            traj,
            nodes=nodes,
            ax=ax,
            title=label,
            node_labels=node_labels,
            markersize=markersize,
            color="black",
            show=False,
        )

    fig.suptitle(suptitle, fontsize=12, y=1.01)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, axes
