from io import BytesIO
import numpy as np
import zmq
from transformers import CLIPProcessor, CLIPModel
from base64 import b64decode
from PIL import Image
from .messages import ScoredMessage, ScorerMessage

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--score-port-in", type=int)
    parser.add_argument("--train-port-in", type=int)
    parser.add_argument("--render-one-side", action="store_true")
    args = parser.parse_args()

    # Set up sockets
    context = zmq.Context()

    receiver = context.socket(zmq.PULL)
    receiver.bind(f"tcp://*:{args.score_port_in}")

    sender = context.socket(zmq.PUSH)
    sender.connect(f"tcp://localhost:{args.train_port_in}")

    # Load model
    model_url = "openai/clip-vit-base-patch32"
    clip = CLIPModel.from_pretrained(model_url)
    processor = CLIPProcessor.from_pretrained(model_url)

    while True:
        # Get renders and convert to images
        s = receiver.recv()
        scorer_msg = ScorerMessage.model_validate_json(s)
        images = [
            np.array(list(b64decode(img_b64))).reshape([512, 512, 4])[::-1, :, :3] for img_b64 in scorer_msg.images
        ]

        # Score renders
        if scorer_msg.scorer_fn == "clip":
            inputs = processor(
                text=[scorer_msg.prompts[scorer_msg.label_idx]],
                images=images,
                return_tensors="pt",
                padding=True,
            )

            outputs = clip(**inputs)
            logits_per_image = outputs.logits_per_image
            if args.render_one_side:
                logits_per_image = logits_per_image[0].unsqueeze(0)
            score = logits_per_image.mean().item() / 30.0
        else:
            inputs = processor(
                text=scorer_msg.prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            outputs = clip(**inputs)
            logits_per_image = outputs.logits_per_image.softmax(1)
            if args.render_one_side:
                logits_per_image = logits_per_image[0].unsqueeze(0)
            score = logits_per_image.mean(0)[scorer_msg.label_idx].item()

        # Send message to training loop
        sender.send_json(
            ScoredMessage(
                buffer_idx=scorer_msg.buffer_idx, score=score, traj_id=scorer_msg.traj_id,
            ).model_dump()
        )


if __name__ == "__main__":
    main()
