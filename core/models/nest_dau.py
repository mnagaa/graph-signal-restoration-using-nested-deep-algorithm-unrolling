from __future__ import annotations
import typing as t
from dataclasses import dataclass
import torch
from torch import nn

from core.common import Task
from core.models.denoisers import GraphDAU, GraphDAU_Params


@dataclass(frozen=True)
class NestDAU_Params:
    L: int = 1
    P: int = 1
    K: int | None = None
    task: Task = 'denoising'
    n_channels: int = 1



class NestDAU(nn.Module):
    def __init__(self, dataset, params: NestDAU_Params, gdau_params: GraphDAU_Params):
        super(NestDAU, self).__init__()
        self.dataset = dataset
        self.params = params
        self.gdau_params = gdau_params
        self.N = dataset.G.N
        # self.E = self.G.e
        # self.L_sparse = dataset.G.L  # sparse graph Lapracani
        self.L = torch.tensor(dataset.G.L.toarray())  # graph lapracian
        self.I = torch.eye(self.N)  # Identity matrix
        # self.I_sp = torch.eye(self.N).to_sparse()

        # Define learnable parameter
        self.rhos = nn.ParameterList(
            [nn.Parameter(torch.rand(1)) for _ in range(self.params.P)])

        denoiser_list = []
        for _ in range(self.params.P):
            denoiser = GraphDAU(dataset=self.dataset, params=gdau_params)
            for p in denoiser.parameters():
                p.requires_grad = True
            denoiser_list.append(denoiser)
        self.denoiser_list = nn.ModuleList(denoiser_list)

    def forward(self, y, H=None):
        '''
        Parameters
        ----------
        y : tensor
        batched tensor (noisy signal)
        H: Degradation matrix

        Returns
        -------
        reconst : tensor
        batched tensor (reconstructed signals) (coordinates: x,y,z)
        '''

        batch_size = y.size()[0]
        y = y.double()  # torch.Size([1, 250, 1])
        # import pdb; pdb.set_trace()

        if self.params.task == 'denoising':
            H = torch.eye(self.N).unsqueeze_(0).repeat(batch_size, 1, 1).double()
        elif self.params.task == 'interpolation':
            H = H.double()
        else:
            raise ValueError

        # initialize variables
        s = [torch.rand([self.N, self.params.n_channels]).double() for _ in range(batch_size)]
        t = [torch.rand([self.N, self.params.n_channels]).double() for _ in range(batch_size)]

        for layer in range(self.params.P):
            rho = self.rhos[layer].double()
            denoiser = self.denoiser_list[layer].double()

            x_list = []
            for idx, (y_, H_) in enumerate(zip(y, H)):
                x = self.inverse_module(y_.double(), tx=s[idx]-t[idx], rho=rho, H=H_)
                x_list.append(x)

                x = torch.unsqueeze(x, 0)  # torch.Size([1, 250, 1])
                t[idx] = torch.unsqueeze(t[idx], 0)  # torch.Size([1, 250, 1])
                s[idx] = denoiser((x + t[idx]).double())  # torch.Size([1, 250, 1])

                t[idx] = t[idx] + x - s[idx]
                t[idx] = torch.squeeze(t[idx], 0)
                s[idx] = torch.squeeze(s[idx], 0)

        reconst = torch.stack(x_list, dim=0)
        return reconst

    def inverse_module(self, y_, tx, rho, H):
        inv = torch.inverse(torch.einsum('nm,mk->nk', torch.t(H), H) + rho * self.I)
        right = torch.sparse.mm(torch.t(H), y_) + rho * tx
        return torch.einsum('nm,mk->nk', inv, right)
