import torch
import torch.nn as nn
from utils import embed, eta
import torch.nn.functional as F
from collections import defaultdict
from cg import clebsch_gordan, O3_clebsch_gordan
from torch.nn.init import xavier_uniform_, zeros_
from einops import repeat, rearrange


class Net(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dim = 3
        self.model1 = Model(input_dim=3, dimension=24)
        self.model2 = Model(input_dim=24, dimension=12)
        self.model3 = Model(input_dim=12, dimension=4)
        self.channelMean = Channel_mean()
        self.dense = nn.Sequential(
            Dense(4, 4, activation=F.elu),
            Dense(4, 256),
            Dense(256, 1),
        )
        self.to(device)
        self.device = device

    def forward(self, V, atoms_info, atoms_lens):
        '''
        For the input to the first network layer,
         we only have scalar features (angular order 𝑙 = 0) and
          a total of 𝐸 = 3 radial features
        V type:
            l:0-2
                a:atoms
                    c:0-3
                        m: 1->3->5
        E as dimension
        first 3 dimension
        '''
        V = self.model1(V, atoms_info)
        V = self.model2(V, atoms_info)
        V = self.model3(V, atoms_info)
        E = self.channelMean(V, atoms_lens)
        E = self.dense(E)

        return E.squeeze(-1)


class Model(nn.Module):
    def __init__(self, input_dim, dimension) -> None:
        super().__init__()
        self.interaction1 = SelfInteractionLayer(input_dim, dimension)
        self.conv = Convolution(dimension)
        self.norm = Norm()
        self.interaction2 = SelfInteractionLayer(dimension, dimension)
        self.nl = NonLinearity(dimension)

    def forward(self, V, atom_data):
        V = self.interaction1(V)
        V = self.conv(V, atom_data)
        V = self.norm(V)
        V = self.interaction2(V)
        V = self.nl(V)
        return V


class R(nn.Module):
    '''
    distance
    '''

    def __init__(self, output_dim, hide_dim=12) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            Dense(12, hide_dim, activation=F.relu),
            Dense(hide_dim, output_dim)
        )

    def forward(self, r):
        assert r.shape[-1] == 12
        r = self.dense(r)
        return r


def Y(dim, vecs):
    '''
    angular
    '''
    if dim == 0:
        return torch.ones((*vecs.shape[:3], 1),device=vecs.device)
    elif dim == 1:
        return vecs
    elif dim == 2:
        r2 = vecs.square().sum(dim=-1).clamp(min=1e-9)
        x, y, z = vecs[..., 0], vecs[..., 1], vecs[..., 2]
        return torch.stack([x * y / r2,
                            y * z / r2,
                            (-x**2 - y**2 + 2. * z**2) /
                            (2 * 3**0.5 * r2),
                            z * x / r2,
                            (x**2 - y**2) / (2. * r2)],
                           dim=-1)
    else:
        raise ValueError('angular dimension error')


class Convolution(nn.Module):
    def __init__(self, output_dim) -> None:
        super().__init__()
        self.radial = R(output_dim)
        self.angular = Y

        # non-zero only for |𝑙𝑖 − 𝑙𝑓 | ≤ 𝑙𝑜 ≤ 𝑙𝑖 + 𝑙𝑓
        self.C = [[0, 0, 0],  [0, 1, 1],  [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 2], [0, 2, 2], [1, 2, 1], [1, 2, 2],
                  [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 0, 2], [2, 1, 1], [2, 1, 2]]
        for i, f, o in self.C:
            self.register_buffer(f'{(o, i, f)}', O3_clebsch_gordan(o, i, f))

    def forward(self, V, atoms_info):
        O = defaultdict(list)
        atoms_rads, atoms_vecs, atoms_nei_idxs = atoms_info
        r = self.radial(atoms_rads)
        atoms_nei_idxs = rearrange(atoms_nei_idxs,'b n l -> b (n l)')
        order = [rearrange(torch.stack([torch.index_select(vv, dim=0, index=nei_idxs) for vv,nei_idxs in zip(v,atoms_nei_idxs)]),'b (n l) d t -> b n l d t',n=v.size(1)) for v in V.values()]
        for i, f, o in self.C:
            y = self.angular(f, atoms_vecs)
            acif = torch.einsum('balc,balf,balci->bacif',
                                r, y, order[i])
            O[o].append(torch.einsum(
                'oif,bacif->baco', self.get_buffer(f'{(o, i, f)}'), acif))
        for i in range(3):
            O[i] = torch.stack(O[i]).sum(dim=0)
        assert O[0].shape[-1] == 1
        assert O[1].shape[-1] == 3
        assert O[2].shape[-1] == 5
        del V

        return O


class Norm(nn.Module):
    def __init__(self, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, V):
        for key in V:
            V[key] = V[key] / torch.sqrt(V[key].square().sum((-1,-2),keepdim=True)).clamp(min=self.eps)
        return V


class SelfInteractionLayer(nn.Module):
    '''
     bias term is only used when operating on angular order 0
    '''

    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.weight = nn.Parameter(xavier_uniform_(
            torch.Tensor(input_dim, output_dim)))
        self.bias = nn.Parameter(zeros_(torch.Tensor(output_dim)))

    def forward(self, V):
        for key in V:
            if key == 0:  # need bias
                V[key] = (torch.einsum('bacm,cd->bamd',
                                       V[key], self.weight)+self.bias).transpose(-1,-2)
            else:
                V[key] = torch.einsum('bacm,cd->badm',
                                      V[key], self.weight)
        return V


class NonLinearity(nn.Module):
    def __init__(self, output_dim) -> None:
        super().__init__()
        self.b1 = nn.Parameter(zeros_(torch.Tensor(output_dim)))
        self.b2 = nn.Parameter(zeros_(torch.Tensor(output_dim)))
        self.bias = torch.nn.ParameterDict({
            '1': self.b1,
            '2': self.b2
        })
        self.output_dim = output_dim

    def forward(self, V):
        eps = 1e-9
        for key in V:
            if key == 0:
                V[key] = eta(V[key])
            else:
                temp = torch.sqrt(V[key].square().sum(-1)).clamp(min=eps)+self.bias[str(key)]
                V[key] = torch.einsum('bacm,bac->bacm', V[key], eta(temp))
        assert V[0].shape[-1] == 1
        assert V[1].shape[-1] == 3
        assert V[2].shape[-1] == 5
        return V


class Channel_mean(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, V, atoms_lens):
        '''
        only V[0]
        '''
        E = V[0]
        for i,l in enumerate(atoms_lens):
            E[i,l:]=0
        return E.sum(dim=1).squeeze(-1)/atoms_lens.unsqueeze(-1)


def init_wb(m):
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)
        zeros_(m.bias)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim).apply(init_wb)
        self.activation = activation

    def forward(self, E):
        out = self.dense(E)
        if self.activation is not None:
            out = self.activation(out)
        return out
