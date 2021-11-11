from functools import lru_cache
import torch

def irr_repr(order, alpha, beta, gamma, dtype=None, device=None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    abc = [alpha, beta, gamma]
    for i, x in enumerate(abc):
        if torch.is_tensor(x):
            abc[i] = x.item()
            if dtype is None:
                dtype = x.dtype
            if device is None:
                device = x.device
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.tensor(wigner_D_matrix(order, *abc), dtype=dtype, device=device)


################################################################################
# Linear algebra
################################################################################

def get_d_null_space(l1, l2, l3, eps=1e-10):
    import scipy
    import scipy.linalg
    import gc

    def _DxDxD(a, b, c):
        D1 = irr_repr(l1, a, b, c)
        D2 = irr_repr(l2, a, b, c)
        D3 = irr_repr(l3, a, b, c)
        return torch.einsum('il,jm,kn->ijklmn', (D1, D2, D3)).view(n, n)

    n = (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)
    random_angles = [
        [4.41301023, 5.56684102, 4.59384642],
        [4.93325116, 6.12697327, 4.14574096],
        [0.53878964, 4.09050444, 5.36539036],
        [2.16017393, 3.48835314, 5.55174441],
        [2.52385107, 0.29089583, 3.90040975],
    ]

    B = torch.zeros((n, n))                                                                             # preallocate memory
    for abc in random_angles:                                                                           # expand block matrix multiplication with its transpose
        D = _DxDxD(*abc) - torch.eye(n)
        B += torch.matmul(D.t(), D)                                                                     # B = sum_i { D^T_i @ D_i }
        del D
        gc.collect()

    s, v = scipy.linalg.eigh(B.numpy(), eigvals=(0, min(1, n - 1)), overwrite_a=True)                   # ask for one (smallest) eigenvalue/eigenvector pair if there is only one exists, otherwise ask for two
    del B
    gc.collect()

    kernel = v.T[s < eps]
    # if not.select first
    if kernel.shape[0]==0:
        kernel=v.T[0:1]
    return torch.from_numpy(kernel)

# # ################################################################################
# # # Clebsch Gordan
# # ################################################################################
cg_cache=dict()
# @lru_cache()
def clebsch_gordan(l1, l2, l3):
    """
    Computes the Clebsch–Gordan coefficients
    out in filter
    D(l1)_il D(l2)_jm D(l3)_kn Q_lmn == Q_ijk
    """
    if (l1, l2, l3) in cg_cache:
        return cg_cache[(l1, l2, l3)]
    if torch.is_tensor(l1):
        l1 = l1.item()
    if torch.is_tensor(l2):
        l2 = l2.item()
    if torch.is_tensor(l3):
        l3 = l3.item()
    if l1 <= l2 <= l3:
        cg=_clebsch_gordan(l1, l2, l3)
    if l1 <= l3 <= l2:
        cg=_clebsch_gordan(l1, l3, l2).transpose(1, 2).contiguous()
    if l2 <= l1 <= l3:
        cg=_clebsch_gordan(l2, l1, l3).transpose(0, 1).contiguous()
    if l3 <= l2 <= l1:
        cg=_clebsch_gordan(l3, l2, l1).transpose(0, 2).contiguous()
    if l2 <= l3 <= l1:
        cg=_clebsch_gordan(l2, l3, l1).transpose(0, 2).transpose(1, 2).contiguous()
    if l3 <= l1 <= l2:
        cg=_clebsch_gordan(l3, l1, l2).transpose(0, 2).transpose(0, 1).contiguous()
    cg_cache[(l1, l2, l3)]=cg
    return cg

def _clebsch_gordan(l1, l2, l3):
    """
    Computes the Clebsch–Gordan coefficients

    D(l1)_il D(l2)_jm D(l3)_kn Q_lmn == Q_ijk
    """
    # these three propositions are equivalent
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert abs(l3 - l1) <= l2 <= l3 + l1
    assert abs(l1 - l2) <= l3 <= l1 + l2

    null_space = get_d_null_space(l1, l2, l3)
    assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
    Q = null_space[0]
    Q = Q.view(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)

    if Q.sum() < 0:
            Q.neg_()

    return Q  # [m1, m2, m3]


if __name__=='__main__':
    # 0,0,0  0,1,1  1,0,1  1,1,0  1,1,1  1,1,2  0,2,2  1,2,1  1,2,2  2,2,0  2,2,1  2,2,2  2,0,2  2,1,1  2,1,2
    # out in filter
    C=clebsch_gordan(0,2,2)
    print(C.shape)