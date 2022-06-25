import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder, train_model, test_model
from utils.tools import to_device, log, synth_one_sample, calc_diffusion_hyperparams
from model import FastSpeech2Loss
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, diffusion_hyperparams, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    for batchs in loader:
        for batch in batchs:
            batch, diffusion_hyperparams = to_device(batch, device, diffusion_hyperparams)
            with torch.no_grad():
                # Forward
                output, model = train_model(batch, model, configs, diffusion_hyperparams)

                # Cal Loss
                losses = Loss(batch, output, diffusion_hyperparams, configs)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        output, batch = test_model(batch, model, configs, diffusion_hyperparams)
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "-g", "--tag", type=str, default='_', required=False, help="output tag"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"] + '_' + model_config["mode"] + '_' + args.tag
    train_config["path"]["log_path"] = train_config["path"]["log_path"] + '_' + model_config["mode"] + '_' + args.tag
    train_config["path"]["result_path"] = train_config["path"]["result_path"] + '_' + model_config["mode"] + '_' + args.tag
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    diffusion_hyperparams = calc_diffusion_hyperparams(
        model_config["diffusion_config"]["noise_schedule_naive"],
        model_config["diffusion_config"]["T"],
        model_config["diffusion_config"]["beta_0"],
        model_config["diffusion_config"]["beta_T"],
        model_config["diffusion_config"]["s"],
    )

    message = evaluate(model, args.restore_step, configs, diffusion_hyperparams)
    print(message)