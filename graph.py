import networkx as nx

from typing import List
from copy import deepcopy
from termcolor import colored, cprint
from collections import Counter, OrderedDict
import re

from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.operators import binary as binary_op
from networkx.algorithms.shortest_paths.generic import shortest_path

class Graph(object):
    def __init__(self, agraph=None, contraction=None):
        if agraph is not None:
            assert isinstance(agraph, nx.DiGraph), type(agraph)
            self.g = agraph
        else:
            self.g = nx.DiGraph()

        if contraction is not None:
            self.contraction = deepcopy(contraction)
        else:
            self.contraction = {}

        self.update__()

    def write_gist(self, gist_path: str):
        with open(gist_path, "w") as f:
            for u, v in self.edges:
                f.write("%s -> %s\n" % (u, v))

    def record_contraction(self, to_node: str, from_node: str):
        if to_node not in self.contraction:
            self.contraction[to_node] = set()
        self.contraction[to_node].add(from_node)

    def set_edge_attr(self, u, v, **kwargs):
        for attr in kwargs:
            nx.set_edge_attributes(self.g, { (u, v) : kwargs[attr] }, attr)

    def set_node_attr(self, u, **kwargs):
        for attr in kwargs:
            nx.set_node_attributes(self.g, { u : kwargs[attr] }, attr)

    def count_node_type(self, key='type'):
        c = Counter()
        for n in self.nodes:
            c[self.get_node_attr(n, key)] += 1
        for name, count in c.most_common():
            print('- %s: %s' % (name, count))

    def count_edge_type(self):
        c = Counter()
        for src, tgt in self.edges:
            c[self.get_edge_attr(src, tgt, 'type')] += 1
        for name, count in c.most_common():
            print('- %s: %s' % (name, count))

    def get_nodes_of_type(self, type_name: str):
        return [ n for n in self.nodes if self.get_node_attr(n, 'type') == type_name ]

    def ls(self):
        if self.ls_ is None:
            self.ls_ = dict(all_pairs_shortest_path_length(self.g))
        return self.ls_

    def ls_rev(self):
        if self.ls_rev_ is None:
            self.ls_rev_ = dict(all_pairs_shortest_path_length(self.g.reverse()))
        return self.ls_rev_

    def adj(self):
        if self.adj_ is None:
            self.adj_ = dict(self.g.adj)
        return self.adj_

    def adj_rev(self):
        if self.adj_rev_ is None:
            self.adj_rev_ = dict(self.g.reverse().adj)
        return self.adj_rev_

    def add_node(self, v):
        self.g.add_node(v)
        self.update__()

    def add_edges(self, uvs):
        for u, v in uvs:
            self.g.add_edge(u, v)
        self.update__()

    def add_edge(self, u, v):
        self.g.add_edge(u, v)
        self.update__()

    def update__(self):
        self.g.remove_edges_from(nx.selfloop_edges(self.g))
        self.ls_ = None
        self.ls_rev_ = None
        self.adj_ = None
        self.adj_rev_ = None
        self.nodes = list(sorted(self.g.nodes))
        self.edges = list(sorted(self.g.edges))

    def search_nodes_by_keyword(self, keyword: str):
        nodes = {} # type: Dict[str, Any]
        for n in self.nodes:
            if re.search(keyword, n, re.IGNORECASE):
                nodes[n] = { "pred": {}, "succ": {} }

        for src, dest in self.edges:
            if src in nodes:
                nodes[src]["succ"][dest] = self.get_edge_attr(src, dest)
            if dest in nodes:
                nodes[dest]["pred"][src] =  self.get_edge_attr(src, dest)

        for n in nodes:
            nodes[n]["succ"] = list(nodes[n]["succ"].items())
            nodes[n]["pred"] = list(nodes[n]["pred"].items())

        return nodes

    def print_search_nodes_by_keyword(self, keyword: str):
        results = self.search_nodes_by_keyword(keyword)
        for node, value in results.items():
            print("- %s" % node)
            print("- Preds:")
            for pred, edge_attr in value["pred"]:
                print("    + %s:\t%s" % (colored(pred, "green"), edge_attr))
            print("- Succs:")
            for succ, edge_attr in value["succ"]:
                print("    + %s:\t%s" % (colored(succ, "green"), edge_attr))
            print()

    def search_edges_by_keyword(self, keyword: str):
        edges = []
        for src, dest in self.edges:
            attrs = self.get_edge_attr(src, dest)
            sentence = src + " " + str(attrs) + " " + dest
            if re.search(keyword, sentence, re.IGNORECASE):
                edges.append((src, dest))
        return edges

    def get_node_attr(self, n, attr_name: str = None):
        if attr_name is None:
            return self.g.nodes.get(n)
        attrs = self.g.nodes.get(n)
        if attr_name in attrs:
            return attrs[attr_name]
        return None

    def get_edge_attr(self, u, v, attr_name: str = None):
        attrs = self.g.get_edge_data(u, v)
        if not attrs:
            if attr_name is not None:
                raise Exception("Access '%s' field of (%s, %s) empty attributes" % (attr_name, u, v))
            else:
                return {}
        if attr_name is None:
            return attrs
        if attr_name in attrs:
            return attrs[attr_name]
        if attr_name == "type":
            return self.get_edge_attr(u, v, "edge_type")
        raise NotImplementedError(attr_name, attrs)

    def remove_nodes(self, nodes_to_kill):
        for n in set(nodes_to_kill):
            if n in self.nodes:
                self.g.remove_node(n)
                if n in self.contraction:
                    del self.contraction[n]
        self.update__()

    def remove_nodes_soft(self, nodes_to_kill):
        ls_copy = self.ls()
        for n in set(nodes_to_kill):
            if n in self.nodes:
                self.g.remove_node(n)
                if n in self.contraction:
                    del self.contraction[n]
        for u in self.g.nodes:
            for v in self.g.nodes:
                if u == v:
                    continue
                if v in ls_copy[u]:
                    self.g.add_edge(u, v)
                    continue
                if u in ls_copy[v]:
                    self.g.add_edge(v, u)
                    continue
        self.update__()

    def print_neighbors(self, node: str):
        print("Pred:")
        for pred in self.g.predecessors(node):
            print("- %s: %s" % (colored(pred, "green"), self.get_edge_attr(pred, node)))
        print("Succ:")
        for succ in self.g.successors(node):
            print("- %s: %s" % (colored(succ, "green"), self.get_edge_attr(node, succ)))

    def get_neighbors(self, node: str, dist=1, type_blacklist=None):
        if node not in self.g.nodes:
            return []
        neighbors = set([node])
        for _ in range(dist):
            for n in list(neighbors):
                for m in nx.all_neighbors(self.g, n):
                    if type_blacklist is not None and self.get_node_attr(m, 'type') in type_blacklist:
                        continue
                    neighbors.add(m)
        return neighbors

    def get_cluster(self, nodes: List[str], dist=1):
        neighbors_map = { n : self.get_neighbors(n, dist=dist) for n in nodes }
        ret = set() # type: Set[str]
        for n in nodes:
            in_neighbors = [ m for m in nodes if m != n and n in neighbors_map[m] ]
            if not in_neighbors:
                return None
            for m in in_neighbors:
                ret = ret.union(neighbors_map[n] & neighbors_map[m])
        return ret

    def get_copy(self):
        return Graph(agraph=self.g.copy(), contraction=self.contraction)

    def is_reachable(self, u: str, v: str):
        return u in self.ls() and v in self.ls()[u]

    def get_path(self, u, v):
        nodes = shortest_path(self.g, u, v)
        edges = []
        if len(nodes) > 1:
            for i in range(len(nodes) - 1):
                edges.append(self.get_edge_attr(nodes[i], nodes[i+1]))
        return nodes, edges

    def contract_node(self, src, dest):
        '''ONLY USE INTERNALLY!'''
        self.g = contracted_nodes(self.g, src, dest, add_contraction_field = False, self_loops=False)
        self.record_contraction(src, dest)

    def print_nodes(self, keyword=None):
        for n in self.nodes:
            if keyword is not None:
                if not re.search(keyword, n, re.IGNORECASE):
                    continue
            print("- %s" % n)
            for k, v in self.get_node_attr(n).items():
                print("    + %s: %s" % (k, v))

    def transitive_successors(self, entry: str):
        nodes = set() # type: Set[str]
        succ_fringe = [entry]
        while succ_fringe:
            n = succ_fringe.pop()
            for m in self.g.successors(n):
                if m not in nodes:
                    nodes.add(m)
                    succ_fringe.append(m)
        return nodes

    def transitive_predecessors_map(self, entry: str):
        nodes = {} # type: Dict[str, int]
        pred_fringe = [entry]
        d = 0
        while pred_fringe:
            d += 1
            n = pred_fringe.pop()
            for m in self.g.predecessors(n):
                if m not in nodes:
                    nodes[m] = d
                    pred_fringe.append(m)
        return nodes

    def transitive_predecessors(self, entry: str):
        nodes = set() # type: Set[str]
        pred_fringe = [entry]
        while pred_fringe:
            n = pred_fringe.pop()
            for m in self.g.predecessors(n):
                if m not in nodes:
                    nodes.add(m)
                    pred_fringe.append(m)
        return nodes

    def check_edge_types(self):
        for u, v in self.edges:
            assert self.get_edge_attr(u, v, "type"), (u, v)

    def get_subcontraction(self, nodes):
        ret = {}
        for n in nodes:
            if n in self.contraction:
                ret[n] = deepcopy(self.contraction[n])
        return ret

    def get_subgraph(self, nodes):
        return Graph(agraph=self.g.subgraph(nodes).copy(), contraction=self.get_subcontraction(nodes))

    def weakly_connected_components(self):
        return [ self.get_subgraph(nodes) for nodes in nx.weakly_connected_components(self.g) ]

    def get_composed_contraction(self, other_contraction):
        ret = deepcopy(self.contraction)
        for u, vs in other_contraction.items():
            if u not in ret:
                ret[u] = set()
            for v in vs:
                ret[u].add(v)
        return ret

    def get_composed(self, other):
        # TODO: asssert same config
        return Graph(agraph=binary_op.compose(self.g, other.g), contraction=self.get_composed_contraction(other.contraction))

    def merge_nodes(self, nodes, new_node_name, new_type="merged"):
        if nodes:
            if new_node_name not in nodes:
                self.g.add_node(new_node_name, domain="library", type=new_type)
            for op in nodes:
                self.contract_node(new_node_name, op)
            self.update__()
            self.set_node_attr(new_node_name, type=new_type)
            self.remove_nodes([None])
            self.g.graph['graph'] = {}

    def print_fanin_fanout(self, FAN_N = 50):
        candidates = []
        for n in self.nodes:
            na = len(self.adj()[n])
            nra = len(self.adj_rev()[n])
            if (na > FAN_N or nra > FAN_N):
                candidates.append((na, nra, n.split("\n")[0]))
        for na, nra, node in sorted(candidates, key=lambda x:x[0]+x[1]):
            cprint("Fan-out: %s\tFan-in: %s\tNode: %s" % (na, nra, node), "red")

def contracted_nodes(G, u, v, self_loops=True, add_contraction_field=True):
    from itertools import chain
    H = G.copy()
    # edge code uses G.edges(v) instead of G.adj[v] to handle multiedges
    if H.is_directed():
        in_edges = ((w if w != v else u, u, d)
                    for w, x, d in G.in_edges(v, data=True)
                    if self_loops or w != u)
        out_edges = ((u, w if w != v else u, d)
                     for x, w, d in G.out_edges(v, data=True)
                     if self_loops or w != u)
        new_edges = chain(in_edges, out_edges)
    else:
        new_edges = ((u, w if w != v else u, d)
                     for x, w, d in G.edges(v, data=True)
                     if self_loops or w != u)
    v_data = H.nodes[v]
    H.remove_node(v)
    H.add_edges_from(new_edges)

    if add_contraction_field:
        if 'contraction' in H.nodes[u]:
            H.nodes[u]['contraction'][v] = v_data # type: ignore
        else:
            H.nodes[u]['contraction'] = {v: v_data}
    return H