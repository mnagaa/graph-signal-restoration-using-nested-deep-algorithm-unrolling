from __future__ import annotations
import os
import typing as t

from core.train import Train, TrainingTools
from core.common import Suffix
from params import Params


DataName = t.Literal['dataA', 'dataB']


def execute_denoising_community_GraphDAU(
    L: int = 1,
    K: int | None = 2,
    epochs: int = 1,
    suffix: Suffix = 'TV-E',
    data_name: DataName = 'dataA'
):
    """
    Community graph (fixed) for denoising
    """
    from core.models.denoisers import GraphDAU, GraphDAU_Params
    from dataset.data_loader.community import CommunityProps, Community
    assert suffix in t.get_args(Suffix)
    compute_fourier_basis = suffix in ['TV-E', 'EN-E']
    trainset = Community(CommunityProps(partition='train', task='denoising',
                                        data_name=data_name, compute_fourier_basis=compute_fourier_basis))
    validset = Community(CommunityProps(partition='valid', task='denoising',
                                        data_name=data_name, compute_fourier_basis=compute_fourier_basis))
    params = GraphDAU_Params(L=L, K=K, task='denoising', suffix=suffix)
    print(f"{params!r}")
    model = GraphDAU(dataset=trainset, params=params)
    tools = TrainingTools(trainset=trainset, validset=validset,
                          model=model, optimizer_name='Adam', param_clamp=False)
    if suffix in ['TV-E', 'EN-E']:
        save_dir = f'{Params.RESULT_BASE}/experiments/basic/{suffix}.{L=}.{epochs=}'
    elif suffix in ['TV-C', 'EN-C']:
        save_dir = f'{Params.RESULT_BASE}/experiments/basic/{suffix}.{L=}.{K=}.{epochs=}'
    os.makedirs(save_dir, exist_ok=True)
    training = Train(tools, save_dir=save_dir)
    training.run(epochs=epochs)


def execute_denoising_community_NestDAU(
    P: int = 1,
    L: int = 1,
    K: int | None = 2,
    epochs: int = 1,
    suffix: t.Literal['TV-E', 'TV-C', 'EN-E', 'EN-C'] = 'TV-E',
    data_name: DataName = 'dataA'
):
    from core.models.denoisers import GraphDAU, GraphDAU_Params
    from core.models.nest_dau import NestDAU, NestDAU_Params
    from dataset.data_loader.community import CommunityProps, Community
    assert suffix in t.get_args(Suffix)
    compute_fourier_basis = suffix in ['TV-E', 'EN-E']
    trainset = Community(CommunityProps(partition='train', task='denoising',
                                        data_name=data_name, compute_fourier_basis=compute_fourier_basis))
    validset = Community(CommunityProps(partition='valid', task='denoising',
                                        data_name=data_name, compute_fourier_basis=compute_fourier_basis))
    gdau_params = GraphDAU_Params(L=L, K=K, task='denoising', suffix=suffix)
    params = NestDAU_Params(L=L, P=P, K=K, task='denoising')
    print(f"{params!r}")
    print(f"{gdau_params!r}")
    model = NestDAU(dataset=trainset, params=params, gdau_params=gdau_params)
    tools = TrainingTools(trainset=trainset, validset=validset,
                          model=model, optimizer_name='Adam', param_clamp=False)
    if suffix in ['TV-E', 'EN-E']:
        save_dir = f'{Params.RESULT_BASE}/experiments/basic/nestdau_{suffix}.{P=}.{L=}.{epochs=}'
    elif suffix in ['TV-C', 'EN-C']:
        save_dir = f'{Params.RESULT_BASE}/experiments/basic/nestdau_{suffix}.{P=}.{L=}.{K=}.{epochs=}'
    os.makedirs(save_dir, exist_ok=True)
    training = Train(tools, save_dir=save_dir)
    training.run(epochs=epochs)


if __name__ == '__main__':
    execute_denoising_community_GraphDAU(
        L=2, epochs=5, suffix='TV-E', data_name='dataA')
    execute_denoising_community_NestDAU(
        P=2, L=2, epochs=3, suffix='TV-E', data_name='dataA')

    # for data_name in t.get_args(DataName):
    #     for L in range(1, 15):
    #         execute_denoising_community_GraphDAU(L=L, epochs=5, suffix='TV-E', data_name=data_name)
    #         execute_denoising_community_GraphDAU(L=L, epochs=5, suffix='EN-E', data_name=data_name)

    # for data_name in t.get_args(DataName):
    #     for K in range(2, 30):
    #         execute_denoising_community_GraphDAU(L=10, K=K, epochs=5, suffix='TV-C', data_name=data_name)
    #         execute_denoising_community_GraphDAU(L=10, K=K, epochs=5, suffix='EN-C', data_name=data_name)

    # for data_name in t.get_args(DataName):
    #     for P in range(1, 15):
    #         execute_denoising_community_NestDAU(P=P, L=2, epochs=3, suffix='TV-E', data_name=data_name)
    #         execute_denoising_community_NestDAU(P=P, L=2, epochs=3, suffix='EN-E', data_name=data_name)
    #         execute_denoising_community_NestDAU(P=P, L=6, epochs=3, suffix='TV-E', data_name=data_name)
    #         execute_denoising_community_NestDAU(P=P, L=6, epochs=3, suffix='EN-E', data_name=data_name)
    #         execute_denoising_community_NestDAU(P=P, L=10, epochs=3, suffix='TV-E', data_name=data_name)
    #         execute_denoising_community_NestDAU(P=P, L=10, epochs=3, suffix='EN-E', data_name=data_name)

    # for data_name in t.get_args(DataName):
    #     for K in range(2, 15):
    #         execute_denoising_community_NestDAU(P=8, L=10, K=K, epochs=3, suffix='TV-C', data_name=data_name)
    #         execute_denoising_community_NestDAU(P=8, L=10, K=K, epochs=3, suffix='EN-C', data_name=data_name)
