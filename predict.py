import torch
from model import Net
import argparse
import os
from utils import getAtoms, getAtomInfo


def get_data(pdb_path, device='cpu'):
    res = []
    if os.path.isfile(pdb_path):
        atoms, rms = getAtoms(pdb_path)
        atom_data = getAtomInfo(atoms, device=device)
        res.append([atoms, atom_data, rms])
    else:
        for name in os.listdir(pdb_path):
            if os.path.isfile(f'{pdb_path}/{name}'):
                atoms, rms = getAtoms(f'{pdb_path}/{name}')
                atom_data = getAtomInfo(atoms, device=device)
                res.append([atoms, atom_data, rms])
            else:
                for file in os.listdir(f'{pdb_path}/{name}'):
                    atoms, rms = getAtoms(f'{pdb_path}/{name}/{file}')
                    atom_data = getAtomInfo(atoms, device=device)
                    res.append([atoms, atom_data, rms])
    return res


def main(args):
    data_set = get_data(args.pdb_path, device=args.device)
    net = Net(device=args.device)
    net.load_state_dict(torch.load(args.model_path, map_location=args.device))
    net.eval()
    with torch.no_grad():
        for atoms, atom_data, rms in data_set:
            out = net(atoms, atom_data)
            print(f'out:{out.item()} rms:{rms} gap:{(out-rms).abs().item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, default='data/val')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    main(args)
