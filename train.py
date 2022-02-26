import torch
from model import Net
import argparse
import os
from utils import getAtoms


def get_data(pdb_path):
    res = []
    for name in os.listdir(pdb_path):
        for file in os.listdir(f'{pdb_path}/{name}'):
            atoms, rms = getAtoms(f'{pdb_path}/{name}/{file}')
            res.append([atoms, rms])
    return res


def main(args):
    dataSet = get_data(args.dir)
    net = Net(device=args.device)
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    loss_fn = torch.nn.HuberLoss()
    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint))
    net.train()
    for i in range(args.epchos):
        for atoms, rms in dataSet:
            out = net(atoms)
            rms = torch.tensor([rms], device=args.device)
            loss = loss_fn(out, rms)
            net.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f'epcho:{i} loss:{loss.item()} out:{out.item()} score:{rms.item()}')
    torch.save(net.state_dict(), args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/train')
    parser.add_argument('--epchos', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save', type=str, default='ARES.pt')
    args = parser.parse_args()

    main(args)
