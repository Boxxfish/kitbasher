# Kitbasher

**Note: This repository is in semi-archived mode. Feel free to fork and make any changes, but aside from major issues no changes will be made to this repository. Thanks!**

[![Video](https://img.youtube.com/vi/54bOSzWGBEw/Tfj4VqN3uzg.jpg)](https://youtu.be/Tfj4VqN3uzg)

This repo contains code to train a neural network to generate Lego models from a description (albeit poorly 😅).

## Running RL Experiments

Everything related to ML can be found in the `kitbasher-ml` directory.

You'll first need to set up a virtual Poetry environment. First, make sure you've downloaded
[Poetry](https://python-poetry.org/). Once you have, create the environment with the following:

```bash
cd kitbasher-ml
poetry install
poetry shell
```

You'll need to run `poetry shell` whenever you want to activate the environment.

Next, you'll have to build the Rust extension, which allows our Python code to talk to the simulation. Run the following:

```bash
cd kitbasher_rust
maturin develop --release
```

Make sure you have [a WandB account](https://wandb.ai/).
Now, run an experiment, remembering to replace `YOUR_WANDB_ID` and `YOUR_CLASS` with your WandB ID and description of your Lego model, respectively:

```bash
mkdir runs
# These are the best parameters I've found
WANDB_ENTITY=YOUR_WANDB_ID python kitbasher/train.py --process-type gcn --score-fn clip --train-batch-size 256 --buffer-size 100_000 --max-actions-per-step 100 --device cuda --eval-every 500 --process-layers 7 --use-mirror --no-advantage --iterations 1_000_000 --tanh-logit --norm-min 0.7 --norm-max 1.2 --distr-scorer  --single-class "YOUR_CLASS" --hidden-dim 256 --add-steps --last-step-sample-bonus 16.0 --max-steps 16 --use-polyak --polyak-tau 0.0002 --q-epsilon 0.5 --use-gcn-skips --render-one-side
```

If you open up your WandB dashboard, you should see models being generated every 500 iterations!

## License and Copyright

This repo is licensed under the MIT license.

LEGO® is a trademark of the LEGO Group of companies which does not sponsor, authorize or endorse this work.
