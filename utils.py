import numpy as np
import torch
from math import sqrt, exp, log
from math import pi

eps = 1e-9

def getAtoms(file):
    atoms = []
    with open(file, 'r') as f:
        for line in f:
            if line[:4] == 'ATOM' or line[:6] == "HETATM":
                # Split the line
                serial, Ele = line[7:11], line[12:16]
                x, y, z = line[30:38], line[38:46], line[46:54]
                # splitted_line = [line[:6], , line[12:16], line[17:20], line[21], line[22:26], ]
                atoms.append({'serial': serial.strip(),
                              'Ele': Ele.strip(),
                              'coordinate': [float(x.strip()), float(y.strip()), float(z.strip())]})
        return atoms


def getAtomInfo(atoms):
    atom_data = []
    for atom in atoms:
        temp = []
        for ato in getNeighbours(atom, atoms):
            mod = distance(atom, ato)
            vec = unit_vector(atom, ato, mod)
            nei_idx = atoms.index(ato)
            temp.append([mod, vec, nei_idx])
        atom_data.append(temp)
    return atom_data


def V_like(n, dim):
    zero, one, two = torch.zeros((n, dim, 1)), torch.zeros(
        (n, dim, 3)), torch.zeros((n, dim, 5))
    return {0: zero, 1: one, 2: two}


def distance(x, y):
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


def eta(x: torch.Tensor):
    return torch.log(0.5*torch.exp(x)+0.5)


def onehot(V0, atoms):
    n = len(atoms)
    for i in range(n):
        Ele = atoms[i]['Ele']
        if Ele == 'C':
            V0[i][0][0] = 1
        elif Ele == 'O':
            V0[i][1][0] = 1
        elif Ele == 'N':
            V0[i][2][0] = 1


def getNeighbours(atom, atoms):
    '''
    50 neighbours
    '''
    # arr_index=[i for i in range(len(atoms))]
    # arr_index.sort(key=lambda x:compare(atom.coordinate,atoms[x].coordinate))
    # return arr_index
    temp = atoms[:]
    temp.sort(key=lambda x: compare(atom['coordinate'], x['coordinate']))
    return temp[:50]


def compare(x, y):
    ans = 0
    for i in range(3):
        ans += (y[i]-x[i])**2
    return ans


def isNan(x):
    return torch.any(torch.isnan(x))
