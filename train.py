import torch
from model import Net
import argparse
import os
from utils import getAtoms


def get_data(pdb_path, names):
    res = []
    for name in names:
        for file in os.listdir(f'{pdb_path}/{name}'):
            atoms, score = getAtoms(f'{pdb_path}/{name}/{file}')
            res.append([atoms, score])
    return res


def main(args):
    dataSet = get_data(args.dir, args.files)
    net = Net(device=args.device)
    optimizer = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.HuberLoss()
    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint))
    net.train()
    for i in range(args.epchos):
        for atoms, score in dataSet:
            out = net(atoms)
            score = torch.tensor([score], device=args.device)
            loss = loss_fn(out, score)
            net.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f'epcho:{i} loss:{loss.item()} out:{out.item()} score:{score.item()}')
    torch.save(net.state_dict(), args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+', default=['157D'])
    parser.add_argument('--dir', type=str, default='data')
    parser.add_argument('--epchos', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save', type=str, default='ARES.pt')
    args = parser.parse_args()

    main(args)
