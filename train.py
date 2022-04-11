import torch
from model import Net
import os
import random
import argparse
from utils import help


def get_data_path(pdb_path):
    res = []
    for root, dirs, files in os.walk(pdb_path):
        for name in files:
            res.append(os.path.join(root, name))
    return res


def main(args):
    data_set_path = get_data_path(args.dir)
    net = Net(device=args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.HuberLoss()
    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint))
    net.train()
    for epoch in range(args.epchos):
        avgloss = 0
        random.shuffle(data_set_path)
        for i, name in enumerate(data_set_path):
            atoms, atom_info, rms = help(name)
            atom_data = [torch.tensor(info).to(args.device)
                         for info in atom_info]
            rms = torch.tensor([rms], device=args.device)
            out = net(atoms, atom_data)
            loss = loss_fn(out, rms)
            loss.backward()
            avgloss += loss.item()
            if (i+1) % args.accumulation_steps == 0 or (i+1) == len(data_set_path):
                optimizer.step()
                optimizer.zero_grad()
        print(f'epcho:{epoch} loss:{avgloss/len(data_set_path)}')
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
