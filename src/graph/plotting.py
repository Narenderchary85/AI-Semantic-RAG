import pickle
import networkx as nx
import matplotlib.pyplot as plt

# ✅ YOUR FUNCTION
def visualize_top_k_nodes(G, k=40):
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:k]
    top_nodes = [n for n, d in top_nodes]

    subG = G.subgraph(top_nodes)

    plt.figure(figsize=(16, 16))
    pos = nx.spring_layout(subG, seed=42)

    nx.draw(
        subG, pos,
        node_size=500,
        node_color="lightgreen",
        edge_color="gray",
        with_labels=True,
        font_size=8
    )

    plt.title(f"Top-{k} Entity Knowledge Subgraph")
    plt.show()


# ✅ LOAD YOUR KG
if __name__ == "__main__":
    with open("data/processed/knowledge_graph.pkl", "rb") as f:
        data = pickle.load(f)

    G = data["graph"]

    visualize_top_k_nodes(G, k=30)
