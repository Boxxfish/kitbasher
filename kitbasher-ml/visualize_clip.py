from pathlib import Path
import pickle as pkl
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from safetensors.torch import load_model
from kitbasher.env import ConstructionEnv
from kitbasher.pretraining import Pretrained
from kitbasher.pretraining import ExpMeta as PretrainingExpMeta
from kitbasher.scorers import single_start, volume_fill_scorer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from argparse import ArgumentParser

def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sum(x**2, 1, keepdim=True).sqrt()

def main():
    parser = ArgumentParser()
    parser.add_argument("--fe-path", type=str)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--normalize", default=False, action="store_true")
    args = parser.parse_args()

    # Fit PCA
    loader_train: DataLoader = pkl.load(open("dataset/train.pkl", "rb"))
    y = torch.cat([batch.y for batch in loader_train], 0)
    if args.normalize:
        y = normalize(y)
    pca = PCA(3)
    pca.fit(y.numpy())

    # Load feature extractor
    fe_path = args.fe_path
    meta_path = Path(fe_path).parent.parent / "meta.json"
    with open(meta_path, "r") as f:
        meta = PretrainingExpMeta.model_validate_json(f.read())
    clip_dim = 512
    env = ConstructionEnv(volume_fill_scorer, single_start, False, 1, True, 16)
    pretrained = Pretrained(
        env.num_parts,
        meta.cfg.part_emb_size,
        meta.cfg.num_steps,
        env.observation_space.node_space.shape[0],
        64,
        clip_dim,
    )
    load_model(pretrained, fe_path)

    # Load valid set
    loader_train = pkl.load(open("dataset/valid.pkl", "rb"))
    y = torch.cat([batch.y for batch in loader_train], 0)[:100]
    if args.normalize:
        y = normalize(y)
    xformed_y = pca.transform(y).T  # Shape: (3, num_samples)

    # Get model output
    index = args.index
    graph = Batch.from_data_list([loader_train.dataset[index]])
    emb = pretrained(graph)  # Shape: (1, clip_dim)
    if args.normalize:
        emb = normalize(emb)
    emb = pca.transform(emb.detach().cpu()).T # Shape: (3, 1)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xformed_y[0], xformed_y[1], xformed_y[2])
    ax.scatter(
        [xformed_y[0][index]], [xformed_y[1][index]], [xformed_y[2][index]], c="red"
    )
    ax.scatter(emb[0], emb[1], emb[2], marker="^")
    plt.show()


if __name__ == "__main__":
    main()
