from __future__ import annotations
import typing as t
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from dataset.data_loader.base import DataLoaderProps
from dataset.modules.community import create_graph
from params import Params


@dataclass(frozen=True)
class CommunityProps(DataLoaderProps):
    task: t.Literal['denoising', 'interpolation']
    data_name: t.Literal['dataA', 'dataB']
    percent: int = 0
    compute_fourier_basis: bool = False

    @property
    def data_path(self):
        if self.task == 'denoising':
            return f"{Params.DATASET_BASE}/{self.task}_community_{self.data_name}"
        else:
            return f"{Params.DATASET_BASE}/{self.task}_community_{self.data_name}_{self.percent}"

def load_data(props: CommunityProps):
    signal = np.load(f"{props.data_path}/data_{props.partition}.npy")
    noisy_signal = np.load(f"{props.data_path}/data_noise_{props.partition}.npy")
    return signal, noisy_signal

def load_degradation_vector(props: CommunityProps):
    deg_vec = np.load(f"{props.data_path}/missing_vectors_{props.partition}.npy")
    deg_sig = np.load(f"{props.data_path}/missing_signals_{props.partition}.npy")  # load degradation mat
    deg_vecs = np.stack([np.diag(vec) for vec in deg_vec])
    return deg_sig, deg_vecs


class Community(Dataset):
    def __init__(self, props: CommunityProps):
        self.props = props
        self.signals, self.noisy_signals = load_data(props)
        self.G = create_graph(n_vertices=250, n_clusters=3, seed=42)
        self.M = self.G.D.toarray()  # normalized differential operator
        self.L = self.G.L.toarray()  # normalized graph lapracian
        self.N = self.G.N      # num of nodes
        self.E = self.G.Ne     # num of edges
        if props.compute_fourier_basis:
            self.G.compute_fourier_basis()
            self.U = self.G.U              # eigenvector matrix
            self.eigenV = self.G.e         # eigenvalues vector
        else:
            self.G.estimate_lmax()
            self.lmax = self.G.lmax

        if props.task == 'interpolation':
            assert props.percent > 0, 'persent should set larger than 0'
            self.deg_sig, self.deg_vecs = load_degradation_vector(props)

    def __getitem__(self, idx):
        if self.props.task == 'interpolation':
            original_signal = self.signals[idx]
            noisy_signal = self.deg_sig[idx]
            return (
                original_signal[:, np.newaxis],
                noisy_signal[:, np.newaxis],
                self.deg_vecs[idx])
        else:
            original_signal = self.signals[idx]
            noisy_signal = self.noisy_signals[idx]
            return (
                original_signal[:, np.newaxis],
                noisy_signal[:, np.newaxis])

    def __len__(self):
        return len(self.noisy_signals)


if __name__ == '__main__':
    props = CommunityProps(
        partition='train',
        task='denoising',
        data_name='dataA',
        compute_fourier_basis=True
    )
