import torch
import torch.nn.functional as F
from math import sqrt, exp, log, pi
import re


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


def getRMS(file):
    with open(file, 'r') as f:
        for line in f:
            if line[:4] == 'rms ':  # not rms_stem
                return float(line[4:].strip())


def getAtomInfo(atoms):
    atoms_rads, atoms_vecs, atoms_nei_idxs = [], [], []
    for atom in atoms:
        rads, vecs, nei_idxs = [], [], []
        for ato in getNeighbours(atom, atoms):
            mod = distance(atom, ato)
            rads.append(radial_fn(mod))
            vecs.append(unit_vector(atom, ato, mod))
            nei_idxs.append(atoms.index(ato))
        atoms_rads.append(rads)
        atoms_vecs.append(vecs)
        atoms_nei_idxs.append(nei_idxs)
    return atoms_rads, atoms_vecs, atoms_nei_idxs


def help(name):
    atoms, rms = getAtoms(name)
    atom_info = getAtomInfo(atoms)
    return atoms, atom_info, rms


def embed(atoms, dim, device='cpu'):
    n = len(atoms)
    zero, one, two = torch.zeros((n, dim, 1)), torch.zeros(
        (n, dim, 3)), torch.zeros((n, dim, 5))
    onehot(zero, atoms)
    return {0: zero.to(device), 1: one.to(device), 2: two.to(device)}


def distance(x, y):
    eps = 1e-9
    ans = 0
    for i in range(3):
        ans += (x['coordinate'][i]-y['coordinate'][i])**2
    return max(sqrt(ans), eps)


def unit_vector(x, y, mod=None):
    if not mod:
        mod = distance(x, y)
    vec = [(x['coordinate'][i]-y['coordinate'][i])/mod for i in range(3)]
    return vec


def radial_fn(Rab):
    G = []
    sigma, n, miu = 1, 11, 12/11
    for i in range(n+1):
        temp = 1/(sigma*sqrt(2*pi))*exp(-(Rab-miu*i)**2/(2*sigma**2))
        G.append(temp)
    return G


def eta(x):
    return F.softplus(x) - log(2.0)


def onehot(V0, atoms):
    n = len(atoms)
    tabel = {
        'C': 0,
        'O': 1,
        'N': 2
    }
    for i in range(n):
        ele = re.sub(r'[^a-zA-Z]', '', atoms[i]['Ele']).upper()
        if ele in tabel:
            V0[i, tabel[ele], 0] = 1


def getNeighbours(atom, atoms, k=50):
    '''
    50 neighbours
    '''
    # arr_index=[i for i in range(len(atoms))]
    # arr_index.sort(key=lambda x:compare(atom.coordinate,atoms[x].coordinate))
    # return arr_index
    temp = atoms[:]
    temp.sort(key=lambda x: compare(atom['coordinate'], x['coordinate']))
    return temp[:k]


def compare(x, y):
    ans = 0
    for i in range(3):
        ans += (y[i]-x[i])**2
    return ans
