import random
import numpy as np
from pygsp import graphs
from utils.plot import plt


def create_graph(n_vertices, n_clusters, seed=42):
    """create community graph dataset

    Args:
        n_vertices (int): size of vertices
        n_clusters (int): size of cluster
        seed (int): seed
    """
    G = graphs.StochasticBlockModel(
        N=n_vertices,
        k=n_clusters,
        p=[0.1, 0.2, 0.15],
        q =0.004,  # 下げるとcommunity 間のスパース
        seed=seed
    )
    G.set_coordinates(seed=seed)
    G.compute_differential_operator()
    return G

def save_signal_plot(G, signal, sname):
    G.plot_signal(
        signal, limits=[0, 6], # sigが1-5の値をとるので
        vertex_size=120, plot_name='')
    plt.axis('off')
    plt.savefig(f"{sname}.pdf", bbox_inches='tight', pad_inches=0.1)
    print(f"save pdf to ... {sname}.pdf")
    plt.clf()
    plt.close()

def generate_original_signal(node_com, sig_min, sig_max, Nclass, seed):
    """
    >>> G = create_graph(10, 3)
    >>> generate_original_signal(G.info['node_com'], sig_min=1, sig_max=5, Nclass=3, seed=0)
    array([4, 4, 4, 5, 1, 1, 1, 1, 1, 1], dtype=int8)
    >>> generate_original_signal(G.info['node_com'], sig_min=1, sig_max=8, Nclass=3, seed=10)
    array([1, 1, 1, 4, 7, 7, 7, 7, 7, 7], dtype=int8)
    """
    random.seed(seed)
    zeros_signal = np.zeros(len(node_com), dtype=np.int8)
    base_signal = list(range(sig_min, sig_max+1))
    select = random.sample(base_signal, Nclass)
    for i, s in enumerate(select):
        idx = np.where(node_com==i)[0]
        zeros_signal[idx] = s
    return zeros_signal

def generate_original_signals(G, N: int, _min: float, _max: float, Nclass: int):
    """
    >>> G = create_graph(10, 3)
    >>> print(generate_original_signals(G, N=5, signal_range=(1,5), Nclass=3))
    [[4 4 4 5 1 1 1 1 1 1]
     [2 2 2 1 5 5 5 5 5 5]
     [1 1 1 5 4 4 4 4 4 4]
     [2 2 2 5 4 4 4 4 4 4]
     [2 2 2 3 1 1 1 1 1 1]]
    """
    data = [generate_original_signal(
            G.info['node_com'], sig_min=_min, sig_max=_max, Nclass=Nclass, seed=n) for n in range(N)]
    data = np.stack(data)
    return data

def generate_degradation_vector(node_com, percent, seed):
    """
    >>> G = create_graph(10, 3)
    >>> generate_degradation_vector(G.info['node_com'], percent=50, seed=0)
    array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0])
    >>> generate_degradation_vector(G.info['node_com'], percent=70, seed=0)
    array([0, 1, 0, 0, 0, 0, 0, 1, 1, 0])
    """
    N = len(node_com)  # node number
    ratio = percent / 100
    assert 0 < ratio < 1
    np.random.seed(seed)
    miss_vector = np.random.choice([0, 1], size=N, p=[ratio, 1 - ratio])
    return miss_vector

def generate_degradation_vectors(G, N, percent):
    """
    >>> G = create_graph(10, 3)
    >>> print(generate_degradation_vectors(G, N=5, percent=10))
    [[1 1 1 1 1 1 1 1 1 1]
     [1 1 0 1 1 0 1 1 1 1]
     [1 0 1 1 1 1 1 1 1 1]
     [1 1 1 1 1 1 1 1 0 1]
     [1 1 1 1 1 1 1 0 1 1]]
    """
    missing_vectors = [generate_degradation_vector(
            G.info['node_com'], percent=percent, seed=n) for n in range(N)]
    missing_vectors = np.stack(missing_vectors)
    return missing_vectors
