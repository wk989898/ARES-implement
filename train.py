import torch
from model import Net
import argparse
import os
from utils import getAtoms, getAtomInfo
import random


def get_data(pdb_path, device='cpu'):
    res = []
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
    data_set = get_data(args.dir, device=args.device)
    net = Net(device=args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.HuberLoss()
    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint))
    net.train()
    for epoch in range(args.epchos):
        avgloss = 0
        random.shuffle(data_set)
        for i, (atoms, atom_data, rms) in enumerate(data_set):
            out = net(atoms, atom_data)
            rms = torch.tensor([rms], device=args.device)
            loss = loss_fn(out, rms)
            loss.backward()
            avgloss += loss.item()
            if (i+1) % args.accumulation_steps == 0 or (i+1) == len(data_set):
                optimizer.step()
                optimizer.zero_grad()
        print(f'epcho:{epoch} loss:{avgloss/len(data_set)}')
    torch.save(net.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/train')
    parser.add_argument('--epchos', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='ARES.pt')
    args = parser.parse_args()

    main(args)
    print('done')
