import numpy as np
from numpy import *
from scipy.misc import comb


def likelihood(A, z, log_scale=False):
    """
    calculate SBM likelihood

    A: adjacency matrix
    z: group membership
    log_scale: log or not
    """
    z_group = unique(z)
    tri_lower = tril(ones((len(z_group), len(z_group))))
    tril_uv_pairs = transpose(nonzero(tri_lower))
    L = 1.0
    for u, v in tril_uv_pairs:
        if u == v:
            # u eq v: within group
            Nu = sum(z==z_group[u])
            if Nu == 1:
                Nuu = 1
                Euu = 0 # diagonal
            else:
                Nuu = comb(Nu, 2)
                Euu = sum(A[where(z==z_group[u])[0],:]\
                           [:,where(z==z_group[u])[0]])/2
            Nuv = Nuu
            Euv = Euu
        else:
            # u neq v: between groups
            Nu = sum(z==z_group[u])
            Nv = sum(z==z_group[v])

            Nuv = Nu * Nv
            Euv = sum(A[where(z==z_group[u])[0],:][:,where(z==z_group[v])[0]])
        Nuv = float(Nuv)
        Euv = float(Euv)
        L = L * ((Euv/Nuv)**Euv * (1.0-Euv/Nuv)**(Nuv-Euv))

    if log_scale:
        return log(L)
    else:
        return L


def generate(M, z):
    """
    generate a simple SBM with undirected, unweighted,
    no multi or selp-loop

    M: k X k SBM
    z: n X 1 vector where z(i) gives the group index of vertex i
       for example, z(i) can have {1,2,3...,k} values.

    return A: adjacency matrix
    """
    k = unique(z)
    n = len(z)
    A = zeros((n,n))

    # SBM based on k
    for u in range(len(k)):
        for v in range(len(k)):
            temp_A = random.random((sum(z==k[u]), sum(z==k[v])))
            temp_A[temp_A < M[u, v]] = 1
            temp_A[temp_A != 1] = 0
            if u == v:
                # upper triange only without diagonal
                temp_A = triu(temp_A, 1)
            # FIXME: This is just work around
            # Need to verify this
            ku_range = where(z==k[u])[0]
            kv_range = where(z==k[v])[0]
            A[ku_range[0]:ku_range[-1]+1, \
              kv_range[0]:kv_range[-1]+1] = temp_A
    return A
