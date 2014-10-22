import numpy as np
from numpy import *
from scipy.misc import comb


def sbm_likelihood(A, z, log_scale=False):
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
