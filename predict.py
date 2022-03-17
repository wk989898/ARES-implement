import torch
from model import Net
import argparse
import os
from utils import getAtoms


def get_data(pdb_path):
    res = []
    if os.path.isfile(pdb_path):
        atoms, rms = getAtoms(pdb_path)
        res.append([atoms, rms])
    else:
        for name in os.listdir(pdb_path):
            for file in os.listdir(f'{pdb_path}/{name}'):
                atoms, rms = getAtoms(f'{pdb_path}/{name}/{file}')
                res.append([atoms, rms])
    return res

def main(args):
    dataSet = get_data(args.pdb_path)
    net = Net(device=args.device)
    net.load_state_dict(torch.load(args.model_path,map_location=args.device))
    net.eval()
    with torch.no_grad():
        for atoms, rms in dataSet:
            out = net(atoms)
            print(f'out:{out.item()} rms:{rms} gap:{(out-rms).abs().item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, default='data/val')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    main(args)
