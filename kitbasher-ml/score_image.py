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
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from argparse import ArgumentParser

from kitbasher.train import LABELS


def main():
    parser = ArgumentParser()
    parser.add_argument("--image", type=str)
    args = parser.parse_args()


    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    prompts = ["a lego " + cat for cat in LABELS]
    img = Image.open(open(args.image, "rb"))
    inputs = processor(
        text=prompts,
        images=[img],
        return_tensors="pt",
        padding=True,
    )

    outputs = clip(**inputs)
    logits = torch.softmax(outputs.logits_per_text.squeeze(1), 0).tolist()
    for prompt, l in zip(prompts, logits):
        print(f"{prompt}: {l}")


if __name__ == "__main__":
    main()
