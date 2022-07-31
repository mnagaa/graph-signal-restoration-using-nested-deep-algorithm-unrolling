import os
import numpy as np
from dataclasses import dataclass

from dataset.modules import community
from utils.logger import logger
from params import Params


@dataclass(frozen=True)
class CommunityProps:
    title: str
    percent: int
    sigma: float = 1.0
    n_vertex: int = 250
    n_cluster: int = 3
    n_train: int = 500
    n_valid: int = 50
    n_test: int = 50
    _min: float = 1  # 1から5までの原信号を生成
    _max: float = 5  # 1から5までの原信号を生成


def community_graph_dataset(props: CommunityProps, seed: int=42):
    logger.debug(f'{props!r}')
    SAVE_DIR = f"{Params.DATASET_BASE}/interpolation_community_{props.title}_{props.percent}"
    os.makedirs(SAVE_DIR, exist_ok=True)
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

        original_data = community.generate_original_signals(
            G, N, _min=props._min, _max=props._max, Nclass=props.n_cluster)
        noise = np.random.normal(loc=0, scale=props.sigma, size=original_data.shape)
        noisy_data = original_data + noise
        missing_vectors = community.generate_degradation_vectors(G, N, percent=props.percent)
        missing_signals = missing_vectors * noisy_data
        np.save(f"{SAVE_DIR}/data_{part}.npy", original_data)
        np.save(f"{SAVE_DIR}/data_noise_{part}.npy", noisy_data)
        np.save(f"{SAVE_DIR}/missing_signals_{part}.npy", missing_signals)
        np.save(f"{SAVE_DIR}/missing_vectors_{part}.npy", missing_vectors)

        IMG_DIR = f"{SAVE_DIR}/img"
        logger.info(f'image save path: {IMG_DIR}')
        os.makedirs(IMG_DIR, exist_ok=True)
        for n in range(3):
            # save image of original signal
            sname = f'{IMG_DIR}/id_{n}_origin'
            community.save_signal_plot(G, original_data[n], sname)

            # save image of noisy signal
            sname = f'{IMG_DIR}/id_{n}_noisy'
            community.save_signal_plot(G, noisy_data[n], sname)

            # save image of original signal
            sname = f'{IMG_DIR}/id_{n}_missing'
            community.save_signal_plot(G, missing_signals[n], sname)


if __name__ == '__main__':
    for p in [30, 50, 70]:
        propsA = CommunityProps(title='dataA', sigma=0.5, percent=p)
        community_graph_dataset(propsA)
        propsB = CommunityProps(title='dataB', sigma=1.0, percent=p)
        community_graph_dataset(propsB)
