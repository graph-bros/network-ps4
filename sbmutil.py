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


if __name__ == "__main__":
    """
    For testing purpose only:

    Example 2.4 in Lecture 6.
    Lgood = 0.0433
    lnLgood = -3.1395
    """

    p10_example = """0 1 1 0 0 0;
    1 0 1 0 0 0;
    1 1 0 1 0 0;
    0 0 1 0 1 1;
    0 0 0 1 0 1;
    0 0 0 1 1 0"""
    z = array([1, 1, 1, 2, 2, 2])
    lgood = 0.0433
    lnlgood = -3.1395

    A = matrix(p10_example)
    lresult = sbm_likelihood(A, z, log_scale=False)
    lnlresult = sbm_likelihood(A, z, log_scale=True)

    print "Lgood:", lresult
    print "lnLgood:", lnlresult

    np.testing.assert_almost_equal(lgood, lresult, 4)
    np.testing.assert_almost_equal(lnlgood, lnlresult, 4)
