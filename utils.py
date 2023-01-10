import torch
import torch.nn.functional as F
import re
from torch import Tensor
from typing import List

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


def help(atoms,dim=3,device='cpu'):
    atoms_info = getAtomInfo(atoms,device=device)
    V = embed(atoms, dim, device=device)
    return V, atoms_info

def getAtomInfo(atoms,device,nei_num=50):
    atoms_coord=torch.stack([torch.as_tensor(atom['coordinate']) for atom in  atoms])
    atoms_distance=distance(atoms_coord[None,:],atoms_coord[:,None])
    values,indices=atoms_distance.topk(nei_num+1,largest=False)
    values,indices = values[:,1:],indices[:,1:] # exclude self
    atoms_rads, atoms_vecs, atoms_nei_idxs=getInfo(atoms_coord,indices,values,nei_num)
    atoms_rads=torch.stack(atoms_rads).to(device).float()
    atoms_vecs=torch.stack(atoms_vecs).to(device).float()
    atoms_nei_idxs=torch.stack(atoms_nei_idxs).to(device)
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

def distance(x, y):
    return ((x-y)**2).sum(-1)

def unit_vector(x, y, mod, eps=1e-9):
    z = x[None,:] - y
    return z / mod[:,None].clamp_min(eps)

sigma, n, miu = 1, 12, 12/11
d_p,d_q=1/(sigma*(2*torch.pi)**0.5),1/(2*sigma**2)
def radial_fn(Rab):
    G=[d_p*torch.exp(-(Rab-miu*i).square()*d_q) for i in range(n)]
    G = torch.stack(G,dim=-1)
    return G

def eta(x):
    return F.softplus(x) - torch.tensor(2.0).log()

def getInfo(atoms_coord:Tensor,indices:Tensor,values:Tensor,nei_num:int=50):
    atoms_rads = []
    atoms_vecs = []
    atoms_nei_idxs = []
    n=atoms_coord.size(0)
    values = values.sqrt()
    for i in range(n):
        rads, vecs, nei_idxs = [],[],[]
        mod = values[i]
        nei_idxs = indices[i]
        rads = radial_fn(mod)
        vecs = unit_vector(atoms_coord[i], atoms_coord[nei_idxs], mod)
        atoms_rads.append(rads) # n 50 12
        atoms_vecs.append(vecs) # n 50 3
        atoms_nei_idxs.append(nei_idxs) # n 50
    return atoms_rads, atoms_vecs, atoms_nei_idxs