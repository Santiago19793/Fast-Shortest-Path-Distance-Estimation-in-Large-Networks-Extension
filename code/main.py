import igraph as ig
import numpy as np
import pandas as pd
from pathlib import Path

import settings
import basic_strats
import constrained_strats
import partition_strats
import local_betw_strat
import linear_strat
import characteristics
import plot

# INITIALIZATION

def create_graph(filename, max_graph_size=None):
    """
    Constructs a graph from file <filename> containing an edgelist.
    Returns the largest connected component in this graph as a graph.
    """

    # Create undirected graph from edgelist and extract the largest connected component, DELIMITER?
    df = pd.read_csv(filename, delimiter='\t', header=None)
    edges = list(df.itertuples(index=False, name=None))

    G = ig.Graph(edges=edges, directed=settings.directed)

    if max_graph_size != None:
        rng = np.random.default_rng(seed=123)
        sampled_nodes = rng.choice(G.vcount(), size=max_graph_size, replace=False)
        largest_cc = G.induced_subgraph(sampled_nodes).components().giant()

    else:
        largest_cc = G.components().giant()

    settings.nr_nodes = largest_cc.vcount()
    settings.nr_edges = largest_cc.ecount()

    return largest_cc


# MAIN

def main():
    """
    Main function to control the program.
    """
    
    # Set all environment variables
    settings.init()
    settings.precompute = True
    settings.characteristics = True

    results = [[] for _ in range(settings.NR_DATASETS)]

    for i, dataset in enumerate(settings.DATASETS):

        if dataset == "":
            settings.directed = True
        else:
            settings.directed = False

        print(f"\n{dataset}")
        filename = f"{settings.PATH}/../datasets/{dataset}.tsv"

        input_file = Path(filename)

        if input_file.is_file():
        
            max_graph_size = None
            
            # Create graph
            G = create_graph(filename, max_graph_size)

            if settings.characteristics:
                characteristics.get_characteristics(G)

            results[i].append(basic_strats.basic_strategies(G))
            results[i].append(constrained_strats.constrained_strategies(G))
            results[i].append(partition_strats.partitioning_strategies(G))
            results[i].append(local_betw_strat.local_betweenness_strategy(G))
            results[i].append(linear_strat.linear_combination_strategy(G))

        else:
            print(f"File {filename} does not exist in this location")

    plot.plot(results)

    
if __name__ == "__main__":
    main()
