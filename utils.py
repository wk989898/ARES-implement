import torch
import torch.nn.functional as F
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


def getAtomInfo(atoms,device):
    atoms_rads, atoms_vecs, atoms_nei_idxs = [], [], []
    atoms_coord=torch.stack([torch.tensor(atom['coordinate']) for atom in  atoms])
    atoms_distance=distance(atoms_coord[None,:],atoms_coord[:,None])
    values,indices=atoms_distance.topk(51,largest=False)
    for i in range(len(atoms)):
        rads, vecs, nei_idxs = [], [], []
        for j in range(1,51): # exclude self
            mod=values[i][j]
            rads.append(radial_fn(mod))
            vecs.append(unit_vector(atoms_coord[i], atoms_coord[indices[i][j]], mod))
            nei_idxs.append(indices[i][j])
        atoms_rads.append(rads)
        atoms_vecs.append(torch.stack(vecs))
        atoms_nei_idxs.append(nei_idxs)
    atoms_rads=torch.tensor(atoms_rads,device=device).float()
    atoms_vecs=torch.stack(atoms_vecs).to(device).float()
    atoms_nei_idxs=torch.tensor(atoms_nei_idxs,device=device)
    return atoms_rads, atoms_vecs, atoms_nei_idxs


def help(atoms,dim=3,device='cpu'):
    atoms_info = getAtomInfo(atoms,device=device)
    V = embed(atoms, dim, device=device)
    return V, atoms_info


def embed(atoms, dim=3, device='cpu'):
    n = len(atoms)
    zero, one, two = torch.zeros((n, dim, 1)), torch.zeros(
        (n, dim, 3)), torch.zeros((n, dim, 5))
    onehot(zero, atoms)
    return {0: zero.to(device), 1: one.to(device), 2: two.to(device)}


def distance(x, y):
    return torch.sqrt(((x-y)**2).sum(-1))


def unit_vector(x, y, mod):
    return (x-y)/mod.clamp_min(1e-9)

@torch.jit.script
def radial_fn(Rab):
    sigma, n, miu = 1, 11, 12/11
    p,q=sigma*(2*torch.pi)**0.5,2*sigma**2
    G=[1/p*torch.exp(-(Rab-miu*i).square()/q) for i in range(n+1)]
    return G


def eta(x):
    return F.softplus(x) - torch.tensor(2.0).log()


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
