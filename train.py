import torch
import argparse
# from model import Net
from batch_model import Net
from utils import to_device
from data import ARESdataset, collate_fn


def main(args):
    print(args)
    train_dataset=ARESdataset(args.train_dir)
    valid_dataset=ARESdataset(args.valid_dir)
    # train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True)
    # valid_dataloader=torch.utils.data.DataLoader(valid_dataset,batch_size=1,shuffle=True)
    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    valid_dataloader=torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    net = Net(device=args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.HuberLoss()
    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint))
    for epoch in range(args.epochs):
        train_loss = 0
        net.train()
        for i,batch in enumerate(train_dataloader):
            V,atoms_info,rms,atoms_lens = (to_device(x,args.device) for x in batch)
            out = net(V, atoms_info, atoms_lens)
            loss = loss_fn(out, rms)
            loss.backward()
            train_loss += loss.item()
            if (i+1) % args.accumulation_steps == 0 or (i+1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
        print(f'epcho:{epoch} train loss:{train_loss/len(train_dataset)}')

        val_loss = 0
        net.eval()
        with torch.no_grad():
            for i,batch in enumerate(train_dataloader):
                V,atoms_info,rms,atoms_lens = (to_device(x,args.device) for x in batch)
                out = net(V, atoms_info, atoms_lens)
                loss = loss_fn(out, rms)
                val_loss+=loss.item()
        print(f'epcho:{epoch} valid loss:{val_loss/len(valid_dataset)}')

    if args.save_path is not None:
        torch.save(net.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--valid_dir', type=str, default='data/val')
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
