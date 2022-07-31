from params import Params

import os
import numpy as np
from dataclasses import dataclass

from dataset.modules import community
from utils.logger import logger


@dataclass(frozen=True)
class CommunityProps:
    """
    Paramters:
    - title: dataA
        data name
    --n_vertex 250
        the number of vertices of the community graph
    --n_class 3
        the number of classes of the community graph
    --n_train 500
        the number of training samples
    --n_valid 50
        the number of validation samples
    --n_test 50
        the number of test samples
    --sigma 0.5
        the standard deviation of the Gaussian noise
    --savedir $SAVE_DIR
        save directory
    """
    title: str
    sigma: float = 1.0
    n_vertex: int = 250
    n_cluster: int = 3
    n_train: int = 500
    n_valid: int = 50
    n_test: int = 50
    _min: float = 1  # 1から5までの原信号を生成
    _max: float = 5  # 1から5までの原信号を生成


def community_graph_dataset(props: CommunityProps, seed: int=42):
    """
    """
    logger.debug(f'{props!r}')
    SAVE_DIR = f"{Params.DATASET_BASE}/denoising_community_{props.title}"
    IMG_DIR = f"{SAVE_DIR}/img"
    logger.info(f'image save path: {IMG_DIR}')
    os.makedirs(IMG_DIR, exist_ok=True)
    G = community.create_graph(
        n_vertices=props.n_vertex,
        n_clusters=props.n_cluster,
        seed=seed)

    for i, (part, N) in enumerate([
        ('train', props.n_train),
        ('valid', props.n_valid),
        ('test', props.n_test)
    ]):
        logger.debug(f'# [{part}: ({i+1}/3)]:  generating {part} signal.')
        noise_seed = i
        np.random.seed(noise_seed)

        # original signal
        original_data = community.generate_original_signals(
            G, N, _min=props._min, _max=props._max, Nclass=props.n_cluster)
        save_name = f"{SAVE_DIR}/data_{part}.npy"
        np.save(save_name, original_data)
        logger.debug(f"# [{part}: ({i+1}/3)]: original data shape: {original_data.shape}")
        logger.debug(f"# [{part}: ({i+1}/3)]: save to -> {save_name}")

        # noisy data
        noise = np.random.normal(loc=0, scale=props.sigma, size=original_data.shape)
        noisy_data = original_data + noise
        name_noisy = f"{SAVE_DIR}/data_noise_{part}.npy"
        np.save(name_noisy, noisy_data)
        logger.debug(f"# [{part}: ({i+1}/3)]:  noisy data shape: {noisy_data.shape}")
        logger.debug(f"# [{part}: ({i+1}/3)]:  save to -> {name_noisy}")

        for n in range(3):
            # save image of original signal
            save_name = f'{IMG_DIR}/id_{n}_origin'
            community.save_signal_plot(G, original_data[n], save_name)

            # save image of noisy signal
            save_name = f'{IMG_DIR}/id_{n}_noisy'
            community.save_signal_plot(G, noisy_data[n], save_name)


if __name__ == '__main__':
    propsA = CommunityProps(title='dataA', sigma=0.5)
    community_graph_dataset(propsA)
    propsB = CommunityProps(title='dataB', sigma=1.0)
    community_graph_dataset(propsB)
