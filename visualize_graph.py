"""
NTTVis - Phase 1: Proof-of-concept static graph renderer
===========================================================

This script is intentionally NOT part of the engine. It just consumes the
`networkx.DiGraph` that `NTTEngine.forward_ntt()` produces and draws it, to
prove the DAG was constructed correctly. Phase 2's interactive GUI will
replace / extend this, but the engine <-> graph contract this depends on
(the 5 standardized node attributes) won't need to change.

Layout strategy: nodes are arranged with x = stage_number and y = position
within the stage, using networkx's multipartite_layout keyed on the
'stage_number' attribute. That gives the classic left-to-right butterfly
diagram look for free.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from ntt_engine import NTTEngine


def draw_ntt_dag(graph: nx.DiGraph, n: int, modulus: int, title_suffix: str = "") -> None:
    """Draw a static butterfly-DAG visualization of an NTT execution graph.

    Parameters
    ----------
    graph : nx.DiGraph
        A graph produced by NTTEngine.forward_ntt() (or any graph following
        the same node-attribute contract: stage_number, butterfly_index,
        twiddle_value, inputs, outputs).
    n, modulus : int
        Just used for the plot title.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty -- run engine.forward_ntt() first")

    # multipartite_layout arranges nodes into vertical columns ("subsets")
    # based on a chosen node attribute -- here, the NTT stage number. This
    # naturally reproduces the classic left-to-right butterfly diagram.
    pos = nx.multipartite_layout(graph, subset_key="stage_number")

    num_stages = max(data["stage_number"] for _, data in graph.nodes(data=True)) + 1
    cmap = plt.get_cmap("viridis")
    node_colors = [
        cmap(data["stage_number"] / max(num_stages - 1, 1))
        for _, data in graph.nodes(data=True)
    ]

    fig, ax = plt.subplots(figsize=(3 + 2.4 * num_stages, 6))

    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        edge_color="gray", arrows=True, arrowsize=14,
        connectionstyle="arc3,rad=0.05", width=1.2,
    )
    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=node_colors, node_size=1100,
        edgecolors="black", linewidths=1.2,
    )

    # Label each node with its butterfly index and twiddle factor -- the two
    # things you'd want to eyeball first when checking the graph is right.
    labels = {
        node: f"B{data['butterfly_index']}\nw={data['twiddle_value']}"
        for node, data in graph.nodes(data=True)
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=8)

    # Edge labels showing which memory address the dependency flows through.
    edge_labels = {
        (u, v): f"addr {d['memory_address']}"
        for u, v, d in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, ax=ax,
        font_size=6, font_color="dimgray",
    )

    ax.set_title(f"NTTVis Phase 1 - Radix-2 CT Butterfly DAG (N={n}, q={modulus}){title_suffix}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("ntt_dag.png", dpi=160)
    print("Saved visualization to ntt_dag.png")
    plt.show()


if __name__ == "__main__":
    # Build the DAG using the Phase 1 engine, then render it.
    N = 8
    Q = 3329

    engine = NTTEngine(n=N, modulus=Q)
    result = engine.forward_ntt([1, 2, 3, 4, 5, 6, 7, 8])

    draw_ntt_dag(result.graph, n=N, modulus=Q)
