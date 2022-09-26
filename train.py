import torch
import argparse
from utils import help
from model import Net
from torch.utils.data import DataLoader
from data import ARESdataset


def main(args):
    dataset=ARESdataset(args.dir)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
    net = Net(device=args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.HuberLoss()
    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint))
    net.train()
    for epoch in range(args.epochs):
        avgloss = 0
        for i,(atoms,rms) in enumerate(dataloader):
            V,atoms_info=help(atoms,device=args.device)
            rms = rms.to(args.device).float()
            out = net(V, atoms_info)
            loss = loss_fn(out.squeeze(-1), rms)
            loss.backward()
            avgloss += loss.item()
            if (i+1) % args.accumulation_steps == 0 or (i+1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
        print(f'epcho:{epoch} loss:{avgloss/len(dataset)}')
    if args.save_path is not None:
        torch.save(net.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='ARES.pt')
    args = parser.parse_args()

    main(args)
    print('done')
