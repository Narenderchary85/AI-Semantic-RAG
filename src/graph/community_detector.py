import community as community_louvain

def detect_communities(G):
    partition = community_louvain.best_partition(G)
    comm_to_nodes = {}
    for node, cid in partition.items():
        comm_to_nodes.setdefault(cid, []).append(node)
    return partition, comm_to_nodes
