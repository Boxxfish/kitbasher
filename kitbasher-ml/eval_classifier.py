"""
Evaluates how well zero-shot classifiers perform with our classes.

Datasets should be structured in the following way:
- dataset_folder
    - meta.json     # Dataset metadata, see `DatasetMeta` for more info
    - images
        - class1
            - image1.jpg
            - image2.jpg
        - class2
            - image3.jpg
"""

import json
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from transformers import CLIPProcessor, CLIPModel
from matplotlib import pyplot as plt

from kitbasher.utils import parse_args

class DatasetClass(BaseModel):
    id: str     # Name in the `images` directory
    prompt: str # Name of the class when inserted into a prompt

class DatasetMeta(BaseModel):
    classes: list[DatasetClass]


class Args(BaseModel):
    ds_path: str
    model: str # Can be either URL or local path
    prompt: str # E.g. "a lego "

def main():
    args: Args = parse_args(Args)
    
    # Load dataset
    ds_path = Path(args.ds_path)
    meta_path = ds_path / "meta.json"
    imgs_path = ds_path / "images"
    with open(meta_path, "r") as f:
        meta = DatasetMeta.model_validate(json.load(f))
    cls_imgs = {}
    for c in meta.classes:
        cls_imgs[c.id] = []
        cls_path = imgs_path / c.id
        for img_path in cls_path.iterdir():
            img = Image.open(img_path)
            cls_imgs[c.id].append(img)

    # Load model
    clip = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)

    # Run images through model
    prompts = [args.prompt + c.prompt for c in meta.classes]
    cls_probs = {}
    idx_to_cls = {i: c.id for i, c in enumerate(meta.classes)}
    for cls_id, images in cls_imgs.items():
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        outputs = clip(**inputs)
        cls_probs[cls_id] = outputs.logits_per_image.tolist() # Shape: (num_imgs, num_prompts)

    # Create confusion matrix
    y_true = sum([[c.id] * len(cls_probs[c.id]) for c in meta.classes], start=[])
    y_pred = sum([[idx_to_cls[torch.tensor(prob).argmax().item()] for prob in cls_probs[c.id]] for c in meta.classes], start=[])
    c_matrix = confusion_matrix(y_true, y_pred, labels=[c.id for c in meta.classes], normalize="true")
    c_disp = ConfusionMatrixDisplay(c_matrix, display_labels=[c.id for c in meta.classes])
    c_disp.plot()
    plt.show()

if __name__ == "__main__":
    main()
