import torch
import argparse
# from model import Net
from batch_model import Net
from utils import to_device
from data import ARESdataset, collate_fn


def main(args):
    print(args)
    dataset=ARESdataset(args.dir)
    # dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    net = Net(device=args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.HuberLoss()
    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint))
    net.train()
    for epoch in range(args.epochs):
        avgloss = 0
        for i,batch in enumerate(dataloader):
            V,atoms_info,rms,atoms_lens = (to_device(x,args.device) for x in batch)
            out = net(V, atoms_info, atoms_lens)
            loss = loss_fn(out, rms)
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
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='ARES.pt')
    args = parser.parse_args()

    main(args)
    print('done')
