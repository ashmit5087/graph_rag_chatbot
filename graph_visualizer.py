"""
graph_visualizer.py
Renders a NetworkX DiGraph as a Matplotlib figure (for Streamlit).
Also produces a pyvis HTML for interactive rendering.
"""

import io
import os
import textwrap
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional


# --------------------------------------------------------------------------- #
#  Colour palette
# --------------------------------------------------------------------------- #
NODE_COLORS = [
    "#4F8EF7", "#F76E4F", "#4FF7A0", "#F7D44F", "#C34FF7",
    "#4FF7E8", "#F74FA8", "#A8F74F", "#F7974F", "#4F6AF7",
]
EDGE_COLOR  = "#888888"
BG_COLOR    = "#0F1117"
TEXT_COLOR  = "#EEEEEE"


def draw_graph(G: nx.DiGraph, title: str = "Knowledge Graph", figsize=(14, 9)) -> Optional[bytes]:
    """Return PNG bytes of the graph figure, or None if graph is empty."""
    if G.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    # Layout
    if G.number_of_nodes() <= 5:
        pos = nx.circular_layout(G)
    elif G.number_of_nodes() <= 20:
        pos = nx.spring_layout(G, seed=42, k=2.5)
    else:
        pos = nx.kamada_kawai_layout(G)

    # Node colouring by degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_colors = [NODE_COLORS[int((d / max_deg) * (len(NODE_COLORS) - 1))] for d in degrees.values()]
    node_sizes  = [800 + 400 * (d / max_deg) for d in degrees.values()]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
    )

    # Draw edges (curved for readability)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=EDGE_COLOR,
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.15",
        width=1.5,
        alpha=0.7,
    )

    # Node labels
    labels = {n: "\n".join(textwrap.wrap(n, 12)) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_color=TEXT_COLOR, font_size=7)

    # Edge labels
    edge_labels = {
        (u, v): data.get("relation", "")
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_color="#AAAAAA", font_size=6,
        bbox=dict(boxstyle="round,pad=0.2", fc=BG_COLOR, alpha=0.6),
    )

    ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=12, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_pyvis_html(G: nx.DiGraph, output_path: str = "graph.html") -> str:
    """Generate an interactive pyvis HTML file. Returns path."""
    try:
        from pyvis.network import Network
    except ImportError:
        return ""

    net = Network(height="500px", width="100%", bgcolor="#0F1117", font_color="#EEEEEE",
                  directed=True)
    net.barnes_hut(spring_length=200)

    for node in G.nodes():
        net.add_node(node, label=node, title=node, color="#4F8EF7")

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        net.add_edge(u, v, title=rel, label=rel, color="#888888")

    net.write_html(output_path)
    return output_path
