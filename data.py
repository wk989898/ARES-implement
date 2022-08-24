import os
from torch.utils.data import Dataset
from utils import getAtoms

class ARESdataset(Dataset):
    def __init__(self, pdb_path) -> None:
        super().__init__()
        self.files = []
        for root, dirs, files in os.walk(pdb_path):
            for name in files:
                self.files.append(os.path.join(root, name))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name=self.files[idx]
        atoms,rms=getAtoms(name)
        return atoms, rms
