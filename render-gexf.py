# Render output in the format of https://github.com/semantic-graph/semantic-graph.scala

import sys
import networkx as nx

def read_gexf(gexf_path: str) -> nx.DiGraph:
    return nx.readwrite.gexf.read_gexf(gexf_path)

if __name__ == "__main__":
    gexf_path = sys.argv[1]
    g = read_gexf(gexf_path)
    for node, attrs in g.nodes.items():
        if "type" in attrs:
            attrs["label"] += "\\ntype: " + attrs["type"]
        if "tag" in attrs:
            attrs["label"] += "\\ntag: " + attrs["tag"]
    nx.drawing.nx_agraph.write_dot(g, gexf_path.replace(".gexf", ".dot"))