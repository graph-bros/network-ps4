import itertools
from collections import defaultdict
from random import shuffle
import numpy as np
from numpy import *
import sbm


def rand_partition(n, j):
    """
    Choose a random(j, n-j) partition

    n = number of vertices
    j = number of vertices of first part

    Example:
    n = 4
    j = 2
    return = [1, 1, 2, 2] or [1, 2, 1, 2] ... (by random)
    """
    j_group = [1 for i in range(j)]
    n_j_group = [2 for i in range(n-j)]
    z = j_group + n_j_group
    shuffle(z, random.random)
    return z


def swap_pairs(z):
    """
    Get possible pairs that can be swapped

    Example:
    z = [1, 1, 2, 2]
    reutrn = [(0, 2), (1, 3), (1, 2), (0, 3)]
    """
    group_names = unique(z)
    a1 = [idx for idx, val in enumerate(z) if val == group_names[0]]
    a2 = [idx for idx, val in enumerate(z) if val == group_names[1]]
    pairs = [zip(x, a2) for x in itertools.permutations(a1, len(a1))]
    return [x for i in pairs for x in i]


def kl_heuristic(A, z):
    """
    optimize any partition score function via
    the Kernigan-Lin(KL) heuristic

    A: adjacency matrix
    z: membership

    return
    bestL: maximum likelihood over all j, n-j partitioning
    bestP: the best partitioning that provides maximum likelihood
    """
    n = len(A)
    n_round = 10
    P = defaultdict(list)
    L = defaultdict(list)
    bestL = 0
    bestP = None
    for j in range(1, (n+1)/2):
        z_init = rand_partition(n, j)
        l_init = sbm.likelihood(A, z_init, log_scale=True)
        z_round = z_init
        for _round in range(n_round):
            z_values = []
            l_values = []
            for index, pair in enumerate(swap_pairs(z_round)):
                z_round[pair[0]], z_round[pair[1]] = \
                z_init[pair[1]], z_init[pair[0]]
                z_values.append(z_round)
                l_values.append(sbm.likelihood(A, z_round, log_scale=True))
            l_max_index = l_values.index(max(l_values))
            z_max_round = z_values[l_max_index]
            z_round = z_max_round
            L[j].append(l_values)
            bestL = max(l_values)
            bestP = z_max_round
    return bestL, bestP, L, P


def read_size(filename):
    with open(filename) as f:
        return len(unique([pair[0] for pair in pair_generator(f)]))


def read_edgelist(filename):
    with open(filename) as f:
        return [edge for edge in pair_generator(f)]


def read_labels(filename):
    with open(filename) as f:
        return [pair[1] for pair in pair_generator(f)]


def pair_generator(f):
    for line in f:
        n1, n2= line.rstrip("\n").split()
        yield int(n1), int(n2)


if __name__ == "__main__":
    """
    Testing with Karate Club Data

    """
    karate_club_edge_path = "karate_club_edges.txt"
    karate_club_labels_path = "karate_labels.txt"
    n = read_size(karate_club_labels_path)
    A = zeros((n, n))

    for edge in read_edgelist(karate_club_edge_path):
        A[edge[0]-1, edge[1]-1] = 1

    z = read_labels(karate_club_labels_path)
    bestL, bestP, L, P = kl_heuristic(A, z)
