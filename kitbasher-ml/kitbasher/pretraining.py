from pathlib import Path
from typing import *
from pydantic import BaseModel
from torch import nn
import torch_geometric
from torch_geometric.utils import subgraph
from torch_geometric.nn import aggr
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.sequential import Sequential
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import wandb
from safetensors.torch import save_model
import pickle as pkl

from kitbasher.env import ConstructionEnv
from kitbasher.scorers import single_start, volume_fill_scorer
from kitbasher.utils import create_directory, parse_args


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        num_parts: int,
        part_emb_size: int,
        num_steps: int,
        node_feature_dim: int,
        hidden_dim: int,
    ):
        nn.Module.__init__(self)

        # Part embeddings
        self.embeddings = nn.Parameter(torch.rand([num_parts, part_emb_size]))

        # Encode-process-encode architecture
        self.encode = nn.Linear(part_emb_size + node_feature_dim, hidden_dim)
        process_layers: List[Union[Tuple[nn.Module, str], nn.Module]] = []
        for _ in range(num_steps):
            process_layers.append(
                (GCNConv(hidden_dim, hidden_dim), "x, edge_index -> x")
            )
            process_layers.append(nn.ReLU())
        self.process = Sequential("x, edge_index, batch", process_layers)

    def forward(self, data: Data):
        data = data.sort()
        x, edge_index, part_ids = (
            data.x,
            data.edge_index,
            data.part_ids,
        )
        edge_index = torch_geometric.utils.add_self_loops(edge_index)[0]
        part_embs = self.embeddings.index_select(
            0, part_ids
        )  # Shape: (num_nodes, part_emb_dim)
        node_embs = torch.cat(
            [part_embs, x], 1
        )  # Shape: (num_nodes, node_dim + part_emb_dim)
        x = self.encode(node_embs)  # Shape: (num_nodes, hidden_dim)
        return x


class Pretrained(nn.Module):
    def __init__(
        self,
        num_parts: int,
        part_emb_size: int,
        num_steps: int,
        node_feature_dim: int,
        hidden_dim: int,
        clip_dim: int,
    ):
        nn.Module.__init__(self)

        self.feature_extractor = FeatureExtractor(
            num_parts, part_emb_size, num_steps, node_feature_dim, hidden_dim
        )
        self.mean_aggr = aggr.MeanAggregation()
        self.out = nn.Linear(hidden_dim, clip_dim)

    def forward(self, data: Data):
        batch = data.batch
        x = self.feature_extractor(data)
        x = self.mean_aggr(x, batch)  # Shape: (num_graphs, hidden_dim)
        x = self.out(x)  # Shape: (num_graphs, clip_dim)
        return x


class Config(BaseModel):
    out_dir: str = "runs/"
    ds_dir: str = "dataset"
    batch_size: int = 32
    num_epochs: int = 1000
    ds_size: int = 1000
    part_emb_size: int = 32
    lr: float = 1e-5
    num_steps: int = 16
    contrastive_coeff: float = 0.0
    device: str = "cuda"


class ExpMeta(BaseModel):
    cfg: Config


def compute_loss(model: Pretrained, batch: Batch, contrastive_coeff: float) -> torch.Tensor:
    pred: torch.Tensor = model(batch)  # Shape: (batch_size, clip_dim)
    norm_pred = pred / torch.sum(pred**2, 1, keepdim=True).sqrt()
    actual: torch.Tensor = batch.y  # Shape: (batch_size, clip_dim)
    norm_actual = actual / torch.sum(actual**2, 1, keepdim=True).sqrt()
    
    # Get in-batch negatives
    batch_size = actual.shape[0]
    permuted_actual = torch.randperm(batch_size)
    while torch.any(torch.arange(batch_size) == permuted_actual):
        permuted_actual = torch.randperm(batch_size)
    actual_permuted = actual[permuted_actual]  # Shape: (batch_size, clip_dim)
    norm_actual_perm = actual_permuted / torch.sum(actual_permuted**2, 1, keepdim=True).sqrt()

    # Perform contrastive loss
    actual_logit = (-torch.sum(norm_pred * norm_actual, 1) + 1) / 2 # Shape: (batch_size)
    actual_perm_logit = (-torch.sum(norm_pred * norm_actual_perm, 1) + 1) / 2 # Shape: (batch_size)
    c_loss = (torch.max(actual_logit - actual_perm_logit + 0.001, torch.zeros(actual_logit.shape, device=actual_logit.device))).mean()
    
    # Perform cosine loss
    loss = -torch.sum(norm_pred * norm_actual, 1).mean()
    return loss + c_loss * contrastive_coeff


def main():
    # Args
    cfg = parse_args(Config)
    clip_dim = 512
    num_train = int(cfg.ds_size * 0.6)
    num_val = int(cfg.ds_size * 0.2)
    num_test = int(cfg.ds_size * 0.2)

    # Create env
    env = ConstructionEnv(
        volume_fill_scorer, single_start, False, 1, True, cfg.num_steps
    )

    # Set up CLIP
    model_url = "openai/clip-vit-base-patch32"
    clip = CLIPVisionModelWithProjection.from_pretrained(model_url)
    processor = CLIPImageProcessor.from_pretrained(model_url)

    # Generate dataset
    ds_dir = Path(cfg.ds_dir)
    if not ds_dir.exists():
        ds_dir.mkdir()
        ds_x = []
        print("Generating data...")
        for _ in tqdm(range(cfg.ds_size)):
            # Generate model
            env.reset()
            while True:
                graph, _, done, trunc, _ = env.step(len(env.model))
                if done or trunc:
                    break
            img = env.screenshot()[0]

            # Remove all action nodes
            edge_index = subgraph(
                graph.action_mask.bool(), graph.edge_index, relabel_nodes=True
            )[0]
            nodes = graph.x[graph.action_mask.bool()]
            part_ids = graph.part_ids[graph.action_mask.bool()]

            # Generate CLIP embedding
            inputs = processor(
                images=[img],
                return_tensors="pt",
                do_rescale=False,
            )
            outputs = clip(**inputs)
            img_emb = outputs.image_embeds[0].detach()

            new_graph = Data(
                nodes, edge_index, part_ids=part_ids, y=img_emb.unsqueeze(0)
            ).cpu()

            ds_x.append(new_graph)
        ds_x_train = ds_x[:num_train]
        ds_x_valid = ds_x[num_train : num_train + num_val]
        loader_train = DataLoader(ds_x_train, batch_size=cfg.batch_size, shuffle=True)
        loader_valid = DataLoader(ds_x_valid, batch_size=cfg.batch_size, shuffle=True)

        with open(ds_dir / "train.pkl", "wb") as f:
            pkl.dump(loader_train, f)
        with open(ds_dir / "valid.pkl", "wb") as f:
            pkl.dump(loader_valid, f)
    else:
        print("Using cached dataset.")
        with open(ds_dir / "train.pkl", "rb") as f:
            loader_train = pkl.load(f)
        with open(ds_dir / "valid.pkl", "rb") as f:
            loader_valid = pkl.load(f)

    # Initialize experiment
    cfg = parse_args(Config)
    wandb.init(project="kitbasher", config=cfg.__dict__, tags=["pretraining"])
    chkpt_path = create_directory(cfg.out_dir, ExpMeta(cfg=cfg))

    # Train network
    print("Training network...")
    model = Pretrained(
        env.num_parts,
        cfg.part_emb_size,
        cfg.num_steps,
        env.observation_space.node_space.shape[0],
        64,
        clip_dim,
    )
    model.to(device=cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for epoch in tqdm(range(cfg.num_epochs), desc="epoch"):
        total_loss = 0.0
        for batch in tqdm(loader_train, desc="batch", leave=False):
            opt.zero_grad()
            loss = compute_loss(model, batch.to(device=cfg.device), cfg.contrastive_coeff)
            total_loss += loss.detach().item()
            loss.backward()
            opt.step()
        total_loss /= num_train // cfg.batch_size

        # Run validation
        total_valid_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(loader_valid, desc="batch", leave=False):
                loss = compute_loss(model, batch.to(device=cfg.device), 0.0)
                total_valid_loss += loss.item()
        total_valid_loss /= num_val // cfg.batch_size

        # Report stats
        wandb.log(
            {
                "total_loss": total_loss,
                "total_valid_loss": total_valid_loss,
            }
        )

        # Save model
        if epoch % 10 == 0:
            save_model(
                model,
                str(chkpt_path / f"net-{epoch}.safetensors"),
            )


if __name__ == "__main__":
    main()
