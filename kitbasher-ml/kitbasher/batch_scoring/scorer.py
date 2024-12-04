from io import BytesIO
import zmq
from transformers import CLIPProcessor, CLIPModel
from base64 import b64decode
from PIL import Image
from .messages import ScoredMessage, ScorerMessage, TO_SCORER_ADDR, TO_TRAINER_ADDR


def main():
    # Set up sockets
    context = zmq.Context()

    receiver = context.socket(zmq.PULL)
    receiver.bind(TO_SCORER_ADDR)

    sender = context.socket(zmq.PUSH)
    sender.bind(TO_TRAINER_ADDR)

    # Load model
    model_url = "openai/clip-vit-base-patch32"
    clip = CLIPModel.from_pretrained(model_url)
    processor = CLIPProcessor.from_pretrained(model_url)

    while True:
        # Get renders and convert to images
        s = receiver.recv()
        scorer_msg = ScorerMessage.model_validate_json(s)
        images = [
            Image.open(BytesIO(b64decode(img_b64))) for img_b64 in scorer_msg.images
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
            score = logits_per_image.mean(0)[scorer_msg.label_idx].item()

        # Send message to training loop
        sender.send_json(
            ScoredMessage(
                buffer_idx=scorer_msg.buffer_idx, score=score
            ).model_dump_json()
        )


if __name__ == "__main__":
    main()
