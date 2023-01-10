import os
import torch
from torch.utils.data import Dataset
from utils import getAtoms, help


class ARESdataset(Dataset):
    def __init__(self, pdb_path, device='cpu') -> None:
        super().__init__()
        self.files = []
        for root, dirs, files in os.walk(pdb_path):
            for name in files:
                self.files.append(os.path.join(root, name))
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name=self.files[idx]
        atoms,rms=getAtoms(name)
        V,atoms_info=help(atoms)
        return V, atoms_info, rms, len(atoms)

def pad(x,l):
    return torch.cat((x,torch.zeros((l, *x.shape[1:]))),dim=0)

def collate_fn(batch):
    Vs, atoms_infos, batch_rms, atoms_lens = zip(*batch)
    max_len = max(atoms_lens)
    O={0:[],1:[],2:[]}
    for n,V in zip(atoms_lens,Vs):
        l = max_len - n
        for i in range(3):
            O[i].append(pad(V[i],l))
    # O:b n d 1/3/5 
    for i in range(3):
        O[i]=torch.stack(O[i])
    
    batch_rads,batch_vecs,batch_idxs = [],[],[]
    for atoms_rads, atoms_vecs, atoms_nei_idxs in atoms_infos:
        l = max_len - atoms_rads.size(0)
        # rad:n 50 12  vec:n 50 3  idx:n 50 
        batch_rads.append(pad(atoms_rads, l))
        batch_vecs.append(pad(atoms_vecs, l))
        batch_idxs.append(pad(atoms_nei_idxs, l))
    batch_rads = torch.stack(batch_rads)
    batch_vecs = torch.stack(batch_vecs)
    batch_idxs = torch.stack(batch_idxs).long()
    batch_rms = torch.as_tensor(batch_rms).float()
    atoms_lens = torch.as_tensor(atoms_lens)
    return O,(batch_rads,batch_vecs,batch_idxs),batch_rms,atoms_lens