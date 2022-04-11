import torch
from model import Net
import argparse
import os
from utils import help


def get_data_path(pdb_path):
    res = []
    for root, dirs, files in os.walk(pdb_path):
        for name in files:
            res.append(os.path.join(root, name))
    return res


def main(args):
    data_set_path = get_data_path(args.pdb_path)
    net = Net(device=args.device)
    net.load_state_dict(torch.load(args.model_path, map_location=args.device))
    net.eval()
    with torch.no_grad():
        for pdb_path in data_set_path:
            atoms, atom_info, rms = help(pdb_path)
            atom_data = [torch.tensor(info).to(args.device)
                         for info in atom_info]
            out = net(atoms, atom_data)
            print(f'out:{out.item()} rms:{rms} gap:{(out-rms).abs().item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, default='data/val')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    main(args)
