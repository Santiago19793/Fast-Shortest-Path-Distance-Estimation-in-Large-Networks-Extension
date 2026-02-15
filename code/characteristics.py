import time
import numpy as np

import settings

# GRAPH CHARACTERISTICS

def get_characteristics(G):
    """
    Function to obtain some classifying characteristics for graph G.
    """

    print("CHARACTERISTICS:")
    print("Number of nodes: ", settings.nr_nodes)
    print("Number of edges: ", settings.nr_edges)

    if G.is_directed():
        print("Modularity:", G.community_walktrap().as_clustering().modularity)
    else:
        print("Modularity:", G.community_multilevel().modularity)

    sampled_nodes = np.random.choice(settings.nr_nodes, size=settings.BFS_RUNS, replace=False)
    
    tot_time = 0.0
    for node in sampled_nodes:
        start = time.time()
        G.get_shortest_paths(node)
        tot_time += time.time() - start
    print(f"One BFS averaged over {settings.BFS_RUNS} nodes: ", tot_time/float(settings.BFS_RUNS))

    print(f"Average distance computed over {settings.AVG_DIST_SAMPLE_SIZE} nodes: ", average_distance(G))    
    print()

# AVERAGE DISTANCE

def average_distance(G):
    """
    Computes the average distance in graph G.
    """
    
    sampled_nodes = np.random.choice(settings.nr_nodes, size=settings.AVG_DIST_SAMPLE_SIZE, replace=False)
    distances = np.array(G.distances(sampled_nodes, list(range(settings.nr_nodes)), mode='ALL'), dtype=np.int32)

    return np.average(distances)