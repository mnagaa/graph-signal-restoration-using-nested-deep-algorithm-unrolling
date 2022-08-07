from __future__ import annotations

import math
import typing as t
from dataclasses import dataclass

import numpy as np
import torch
from core.common import Suffix, Task, soft_threshold
from torch import nn


@dataclass(frozen=True)
class GraphDAU_Params:
    L: int = 1
    K: int | None = None
    task: Task = 'denoising'
    suffix: Suffix = 'TV-E'
    n_channels: int = 1


class GraphDAU(nn.Module):
    def __init__(self, dataset, params: GraphDAU_Params):
        super().__init__()
        assert params.suffix in t.get_args(Suffix)
        torch.manual_seed(0)
        np.random.seed(seed=42)
        self.params = params
        self.N, self.E = dataset.N, dataset.E  # Num of nodes and edges
        self.setup(params.suffix, dataset)
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.rand([params.n_channels])) for _ in range(params.L)])
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.rand([params.n_channels])) for _ in range(params.L)])
        if params.suffix in ['EN-E', 'EN-C']:
            self.alphas = nn.ParameterList(
                [nn.Parameter(torch.rand([params.n_channels])) for _ in range(params.L)])

        if params.suffix == 'TV-E':
            self.denoiser = self.TV_E
        elif params.suffix == 'TV-C':
            self.denoiser = self.TV_C
        elif params.suffix == 'EN-E':
            self.denoiser = self.EN_E
        elif params.suffix == 'EN-C':
            self.denoiser = self.EN_C

    def setup(self, suffix: Suffix, dataset):
        M = dataset.M  # incidence matrix
        self.M = torch.tensor(M)
        self.Mt = torch.transpose(self.M, 0, 1)
        # sparse matrix
        self.M_sp = torch.tensor(M).to_sparse()
        self.Mt_sp = torch.transpose(self.M, 0, 1).to_sparse()
        self.I = torch.eye(self.N)  # Identity matrix
        self.I_sp = self.I.to_sparse().double()
        if suffix in ['TV-E', 'EN-E']:
            self.ones = torch.ones(self.N)  # vector of ones
            # eigendecomposition of graph Laplacian
            self.eigenV = torch.tensor(dataset.eigenV)  # vector
            self.U = torch.tensor(dataset.U)
            self.Ut = torch.transpose(self.U, 0, 1)

        elif suffix in ['TV-C', 'EN-C']:
            L = dataset.L
            self.L = torch.tensor(L)
            self.L_sp = self.L.to_sparse().double()  # graph Lapracian
            self.lmax = dataset.lmax

    def TV_E(self, noisy: torch.Tensor, layer: int, z: torch.Tensor, y: torch.Tensor):
        """Graph total variation with Eigendecomposition"""
        gamma = self.gammas[layer]
        beta = self.betas[layer]
        reconst_signals = []
        for idx, noisy_ in enumerate(noisy):  # torch.Size([250, 1])
            tmp_signals = []
            # equation (1)
            for c in range(self.params.n_channels):  # channels
                inv_vec = 1.0 / (self.ones + (1.0 / gamma[c]) * self.eigenV)
                diag = torch.diag(inv_vec).to_sparse()
                a1 = torch.sparse.mm(diag, self.Ut)
                a2 = torch.einsum('nm, mk -> nk', self.U, a1)
                b1 = z[idx][:, c] - y[idx][:, c]
                b2 = torch.sparse.mm(self.Mt_sp, b1.view(-1, 1))  # N, 1

                g_lth = noisy_[:, c].view(-1, 1) + (1.0 / gamma[c]) * b2
                x_ = torch.einsum('nm, mk -> nk', a2, g_lth)
                tmp_signals.append(x_.squeeze())
            x = torch.stack(tmp_signals, dim=1)  # torch.Size([250, 1])
            reconst_signals.append(x)

            # equation (2)
            for c in range(self.params.n_channels):  # channels
                v = torch.sparse.mm(
                    self.M_sp, x[:, c].view(-1, 1)) + y[idx][:, c].view(-1, 1)
                z[idx][:, c] = soft_threshold(v.squeeze(), beta[c])

            # equation (3)
            y[idx] = y[idx] + torch.sparse.mm(self.M_sp, x) - z[idx]
        return reconst_signals

    def TV_C(self, g: torch.Tensor, layer: int, z: torch.Tensor, y: torch.Tensor):
        """Graph total variation with Chebyshev polynomial approximation"""
        gamma = self.gammas[layer]
        beta = self.betas[layer]
        reconst_signals = []
        for idx, g1 in enumerate(g):  # torch.Size([250, 1])
            x_list1 = []
            # equation (1)
            for c in range(self.params.n_channels):  # channels
                g2 = torch.sparse.mm(
                    self.Mt_sp, (z[idx][:, c]-y[idx][:, c]).view(-1, 1)).squeeze()
                g_lth = g1[:, c] + (1/gamma[c]) * g2
                g_lth = g_lth.view(-1, 1)
                x_ = self.filter_H(g_lth, gamma_lth=gamma[c])
                x_list1.append(x_)

            x = torch.stack(x_list1, dim=1)  # torch.Size([250, 1])
            x = torch.squeeze(x, dim=2)
            reconst_signals.append(x)

            # equation (2)
            for c in range(self.params.n_channels):  # channels
                v = torch.sparse.mm(
                    self.M_sp, x[:, c].view(-1, 1)) + y[idx][:, c].view(-1, 1)
                z[idx][:, c] = soft_threshold(
                    v.squeeze(), beta[c])  # soft_thresh

            # equation (3)
            y[idx] = y[idx] + torch.sparse.mm(self.M_sp, x) - z[idx]
        return reconst_signals

    def EN_E(self, g: torch.Tensor, layer: int, z: torch.Tensor, y: torch.Tensor):
        gamma = self.gammas[layer]
        beta = self.betas[layer]
        alpha = self.alphas[layer]
        reconst_signals = []
        for idx, g_ in enumerate(g):  # torch.Size([250, 1])
            x_list1 = []
            # equation (1)
            for c in range(self.params.n_channels):  # channels
                inv_vec = 1.0 / (self.ones + (1.0 / gamma[c]) * self.eigenV)
                diag = torch.diag(inv_vec).to_sparse()
                a1 = torch.sparse.mm(diag, self.Ut)
                a2 = torch.einsum('nm, mk -> nk', self.U, a1)
                b1 = z[idx][:, c] - y[idx][:, c]
                b2 = torch.sparse.mm(self.Mt_sp, b1.view(-1, 1))  # N, 1

                g_lth = g_[:, c].view(-1, 1) + (1.0 / gamma[c]) * b2
                x_ = torch.einsum('nm, mk -> nk', a2, g_lth)
                x_list1.append(x_.squeeze())

            x = torch.stack(x_list1, dim=1)  # torch.Size([250, 1])
            reconst_signals.append(x)

            # equation (2)
            for c in range(self.params.n_channels):  # channels
                v = torch.sparse.mm(
                    self.M_sp, x[:, c].view(-1, 1)) + y[idx][:, c].view(-1, 1)
                z[idx][:, c] = alpha[c] * soft_threshold(v.squeeze(), beta[c])

            # equation (3)
            y[idx] = y[idx] + torch.sparse.mm(self.M_sp, x) - z[idx]
        return reconst_signals

    def EN_C(self, g: torch.Tensor, layer: int, z: torch.Tensor, y: torch.Tensor):
        """ElasticNet-like regularization with Chebyshev polynomial approximation

        Description:
            ADMM for ElasticNet-like regularization
        """
        gamma = self.gammas[layer]
        beta = self.betas[layer]
        alpha = self.alphas[layer]
        reconst_signals = []
        for idx, g1 in enumerate(g):
            x_list1 = []
            # equation (1)
            for c in range(self.params.n_channels):
                g2 = torch.sparse.mm(
                    self.Mt_sp, (z[idx][:, c]-y[idx][:, c]).view(-1, 1)).squeeze()
                g_lth = g1[:, c] + (1/gamma[c]) * g2
                g_lth = g_lth.view(-1, 1)
                x_ = self.filter_H(g_lth, gamma_lth=gamma[c])
                x_list1.append(x_)
            x = torch.stack(x_list1, dim=1)
            x = torch.squeeze(x, dim=2)

            # equation (2)
            reconst_signals.append(x)
            for c in range(self.params.n_channels):
                v = torch.sparse.mm(
                    self.M_sp, x[:, c].view(-1, 1)) + y[idx][:, c].view(-1, 1)
                z[idx][:, c] = alpha[c] * soft_threshold(v.squeeze(), beta[c])

            # equation (3)
            y[idx] = y[idx] + torch.sparse.mm(self.M_sp, x) - z[idx]
        return reconst_signals

    def forward(self, g: torch.Tensor):
        batch_size = g.size()[0]
        # auxiliary variables
        z, y = (
            [torch.rand([self.E, self.params.n_channels]).double()
             for _ in range(batch_size)],
            [torch.rand([self.E, self.params.n_channels]).double()
             for _ in range(batch_size)]
        )
        for layer in range(self.params.L):
            x_list = self.denoiser(g, layer, z, y)
        pred_x = torch.stack(x_list, dim=0)
        return pred_x

    def filter_H(self, g_lth, gamma_lth):
        """
        See for detail.
            "Fast singular value shrinkage with Chebyshev polynomial approximation based on signal sparsity"
        """
        def recur_fn(L_hat_sp, K, g_lth):
            '''
            Parameters
            ----------
            L_hat_sparse: sparse torch.Tensor
                            L's eigenvalues range [-1,1]
            '''
            Zi = []
            for i in range(K):
                if i == 0:
                    Zi_k = torch.sparse.mm(self.I_sp, g_lth)
                    Zi.append(Zi_k)
                elif i == 1:
                    Zi_k = torch.sparse.mm(L_hat_sp, g_lth)
                    Zi.append(Zi_k)
                else:
                    Zi_k = torch.sparse.mm(L_hat_sp, 2 * Zi[-1]) - Zi[-2]
                    Zi.append(Zi_k)
            return Zi

        def coef_fn(K, lmax, gamma_lth):
            """calculate chebyshev coefficients"""
            def h_fn(x):
                return gamma_lth / (gamma_lth + x)

            alpha = K
            coefs = []
            for k in range(alpha):
                sum_part = 0
                for l in range(1, alpha+1):
                    theta_lth = (math.pi * (l - 0.5)) / alpha
                    sum_part = (
                        sum_part + math.cos(k * theta_lth) *
                        h_fn((lmax / 2) * (math.cos(theta_lth) + 1))
                    )
                c = (2 / alpha) * sum_part
                coefs.append(c)
            return coefs

        K = self.params.K
        lmax = self.lmax
        L_hat = (2 / lmax) * self.L - self.I
        L_hat_sp = L_hat.to_sparse().double()

        comps = recur_fn(L_hat_sp, K, g_lth.double())
        coefs = coef_fn(K=K, lmax=lmax, gamma_lth=gamma_lth)

        coefs[0] = 0.5 * coefs[0]
        comps = torch.stack(comps, dim=1).squeeze()
        coefs = torch.stack(coefs)
        coefs = coefs.view(len(coefs), 1)
        H_A = torch.einsum('ep,pc->ec', comps.double(), coefs.double())
        return H_A

    def print_grad(self):
        gamma = ' '.join([f'{i}: {gamma.grad}' for i,
                         gamma in enumerate(self.gammas)])
        beta = ' '.join([f'{i}: {beta.grad}' for i,
                        beta in enumerate(self.betas)])
        return f"gamma: {gamma}\n beta: {beta}"
