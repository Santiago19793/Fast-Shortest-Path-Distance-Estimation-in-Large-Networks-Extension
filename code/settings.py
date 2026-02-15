import os

# INITIALIZE/CREATE ALL ENVIRONMENT VARIABLES

def init():
    global nr_nodes
    global nr_edges
    global degrees
    global degrees_sorted
    global closeness_approx
    global closeness_approx_sorted
    global betweenness_approx_sorted
    global nodes_partitions_d_size
    global directed
    global precompute
    global characteristics

    global LANDMARK_SIZES
    global NR_LANDMARKS
    global NR_ITERATIONS
    global NR_REPETITIONS
    global BFS_RUNS
    global AVG_DIST_SAMPLE_SIZE
    global DATASETS
    global NR_DATASETS
    global PATH

    LANDMARK_SIZES = [5, 10, 20, 50, 100]
    NR_LANDMARKS = len(LANDMARK_SIZES)
    NR_ITERATIONS = 1000
    NR_REPETITIONS = 3.0
    BFS_RUNS = 100
    AVG_DIST_SAMPLE_SIZE = 500
    DATASETS = []
    NR_DATASETS = len(DATASETS)
    PATH = os.getcwd()

