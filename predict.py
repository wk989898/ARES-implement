import torch
import argparse
from model import Net
# from batch_model import Net
from utils import to_device
from data import ARESdataset
from torch.utils.data import DataLoader



def main(args):
    dataset = ARESdataset(args.dir)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
    net = Net(device=args.device)
    net.load_state_dict(torch.load(args.model_path, map_location=args.device))
    net.eval()
    with torch.no_grad():
        for batch in dataloader:
            V,atoms_info,rms,atoms_lens = (to_device(x,args.device) for x in batch)
            out = net(V, atoms_info, atoms_lens)
            print(f'out:{out.item()} rms:{rms.item()} gap:{(out-rms).abs().item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/val')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    main(args)
