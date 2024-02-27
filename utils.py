import re
import math
import torch
import numba
import numpy as np
import torch.nn.functional as F


def getAtoms(file):
    atoms = []
    score = 0
    with open(file, 'r') as f:
        for line in f:
            if line[:4] == 'ATOM' or line[:6] == "HETATM":
                # Split the line
                serial, Ele = line[7:11], line[12:16]
                x, y, z = line[30:38], line[38:46], line[46:54]
                # splitted_line = [line[:6], , line[12:16], line[17:20], line[21], line[22:26], ]
                atoms.append({'serial': serial.strip(),
                              'Ele': re.sub(r'[^a-zA-Z]', '', Ele).upper().strip(),
                              'coordinate': [float(x.strip()), float(y.strip()), float(z.strip())]})
            if line[:4] == 'rms ':  # not rms_stem
                score = float(line[4:].strip())
        return atoms, score


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    elif isinstance(x, dict):
        for k in x:
            x[k] = to_device(x[k], device)
    elif isinstance(x, (list, tuple)):
        x = [to_device(xx, device) for xx in x]
    return x


def help(atoms, dim=3, device='cpu'):
    atoms_info = getAtomInfo(atoms, device=device)
    V = embed(atoms, dim, device=device)
    return V, atoms_info


def getAtomInfo(atoms, device, nei_num=50):
    atoms_coord = np.array([atom['coordinate'] for atom in atoms])
    atoms_distance = distance(atoms_coord[None, :], atoms_coord[:, None])
    values, indices = torch.from_numpy(
        atoms_distance).topk(nei_num+1, largest=False)
    values, indices = values[:, 1:], indices[:, 1:]  # exclude self
    atoms_rads, atoms_vecs, atoms_nei_idxs = getInfo(
        atoms_coord, indices, values)
    atoms_rads = torch.from_numpy(atoms_rads).to(device).float()
    atoms_vecs = torch.from_numpy(atoms_vecs).to(device).float()
    atoms_nei_idxs = torch.from_numpy(atoms_nei_idxs).to(device)
    return atoms_rads, atoms_vecs, atoms_nei_idxs


def embed(atoms, dim=3, device='cpu'):
    n = len(atoms)
    zero, one, two = torch.zeros((n, dim, 1)), torch.zeros(
        (n, dim, 3)), torch.zeros((n, dim, 5))
    onehot(zero, atoms)
    return {0: zero.to(device), 1: one.to(device), 2: two.to(device)}


def onehot(V0, atoms):
    n = len(atoms)
    tabel = {
        'C': 0,
        'O': 1,
        'N': 2
    }
    for i in range(n):
        ele = re.sub(r'[^a-zA-Z]', '', atoms[i]['Ele'][0]).upper()
        if ele in tabel:
            V0[i, tabel[ele], 0] = 1


@numba.njit
def distance(x, y):
    return np.sum((x-y)**2, -1)


@numba.njit
def unit_vector(x, y, mod, eps=1e-9):
    return (x-y) / mod + eps


class Radial:
    sigma = 1
    n = 12
    miu = 12/11
    mius = [12/11*i for i in range(12)]
    p = 1/((2*np.pi)**0.5)
    q = -1/2


@numba.njit
def radial_fn(Rab):
    q = -1/2
    p = 1/((2*np.pi)**0.5)
    mius = [12/11*i for i in range(12)]
    G = [p * np.exp(np.square(Rab-miu)*q)
         for miu in mius]
    return G


def getInfo(atoms_coord, indices, values):
    atoms_rads = []
    atoms_vecs = []
    atoms_nei_idxs = []
    n = len(atoms_coord)
    indices = indices.numpy()
    values = values.sqrt().numpy()
    for i in range(n):
        rads, vecs, nei_idxs = [], [], []
        mod = values[i]
        nei_idxs = indices[i]
        rads = radial_fn(mod)
        vecs = unit_vector(atoms_coord[i][None, :],
                           atoms_coord[nei_idxs], mod[:, None])
        atoms_rads.append(rads)  # n 50 12
        atoms_vecs.append(vecs)  # n 50 3
        atoms_nei_idxs.append(nei_idxs)  # n 50
    return np.array(atoms_rads), np.array(atoms_vecs), np.array(atoms_nei_idxs)


def Y2(vecs):
    @numba.njit
    def y2(vecs):
        eps = 1e-9
        r2 = np.sum(vecs**2, -1) + eps
        x, y, z = vecs[..., 0], vecs[..., 1], vecs[..., 2]
        return [x * y / r2,
                y * z / r2,
                (-x**2 - y**2 + 2. * z**2) /
                (2 * 3**0.5 * r2),
                z * x / r2,
                (x**2 - y**2) / (2. * r2)]
    device = vecs.device
    dtype = vecs.dtype
    vecs = np.stack(y2(vecs.cpu().numpy()), axis=-1)
    return torch.from_numpy(vecs).to(device=device, dtype=dtype)


def eta(x):
    return F.softplus(x) - math.log(2.0)
