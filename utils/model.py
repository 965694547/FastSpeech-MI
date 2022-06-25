import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, FastSpeech2_MI1, FastSpeech2_MI2, FastSpeech2_MI3, FastSpeech2_MI4, FastSpeech2_MI5, ScheduledOptim
from utils.tools import transforme_mel, std_normal

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    if model_config["mode"] == "FastSpeech2":
        model = FastSpeech2(preprocess_config, model_config).to(device)
    elif model_config["mode"] == "FastSpeech2_MI1":
        model = FastSpeech2_MI1(preprocess_config, model_config).to(device)
    elif model_config["mode"] == "FastSpeech2_MI2":
        model = FastSpeech2_MI2(preprocess_config, model_config).to(device)
    elif model_config["mode"] == "FastSpeech2_MI3":
        model = FastSpeech2_MI3(preprocess_config, model_config).to(device)
    elif model_config["mode"] == "FastSpeech2_MI4":
        model = FastSpeech2_MI4(preprocess_config, model_config).to(device)
    elif model_config["mode"] == "FastSpeech2_MI5":
        model = FastSpeech2_MI5(preprocess_config, model_config).to(device)
    else:
        raise ValueError("mode is None")

    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location='cpu')
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location='cpu')
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs


def train_model(batch, model, configs, diffusion_hyperparams):
    (preprocess_config, model_config, train_config) = configs
    if model_config["mode"] == "FastSpeech2":
        # Forward
        output = model(*(batch[2:]))

    elif model_config["mode"] == "FastSpeech2_MI1":
        _dh = diffusion_hyperparams
        B = batch[6].shape[0]
        T = _dh["T"]
        diffusion_steps = torch.randint(T, size=(B, 1, 1))  # randomly sample diffusion steps from 1~T
        transformed_X = transforme_mel(diffusion_steps, diffusion_hyperparams, batch[6])
        # Forward
        output = model(*(batch[2:]), diffusion_steps=diffusion_steps.view(B, 1), transformed_X=transformed_X)

    elif model_config["mode"] == "FastSpeech2_MI2":
        _dh = diffusion_hyperparams
        B = batch[6].shape[0]
        T = _dh["T"]
        diffusion_steps = torch.randint(T, size=(B, 1, 1))  # randomly sample diffusion steps from 1~T
        transformed_X = transforme_mel(diffusion_steps, diffusion_hyperparams, batch[6])
        # Forward
        output = model(*(batch[2:]), diffusion_steps=diffusion_steps.view(B, 1), transformed_X=transformed_X)

    elif model_config["mode"] == "FastSpeech2_MI3":
        _dh = diffusion_hyperparams
        B = batch[6].shape[0]
        T = _dh["T"]
        diffusion_steps = torch.randint(T, size=(B, 1, 1))  # randomly sample diffusion steps from 1~T
        transformed_X = transforme_mel(diffusion_steps, diffusion_hyperparams, batch[6])
        # Forward
        output = model(*(batch[2:]), diffusion_steps=diffusion_steps.view(B, 1), transformed_X=transformed_X)

    elif model_config["mode"] == "FastSpeech2_MI4":
        _dh = diffusion_hyperparams
        B = batch[6].shape[0]
        T = _dh["T"]
        diffusion_steps = torch.randint(T, size=(B, 1, 1))  # randomly sample diffusion steps from 1~T
        transformed_X = transforme_mel(diffusion_steps, diffusion_hyperparams, batch[6])
        # Forward
        output = model(*(batch[2:]), diffusion_steps=diffusion_steps.view(B, 1), transformed_X=transformed_X)

    elif model_config["mode"] == "FastSpeech2_MI5":
        _dh = diffusion_hyperparams
        B = batch[6].shape[0]
        T = _dh["T"]
        diffusion_steps = torch.randint(T, size=(B, 1, 1))  # randomly sample diffusion steps from 1~T
        Gauss = std_normal(batch[6].shape)
        transformed_X = transforme_mel(diffusion_steps, diffusion_hyperparams, batch[6], z=Gauss)
        # Forward
        output = model(*(batch[2:]), diffusion_steps=diffusion_steps.view(B, 1), transformed_X=transformed_X, z=Gauss)

    else:
        raise ValueError("mode is None")

    return output, model

def test_model(batch, model, configs, diffusion_hyperparams, pitch_control=1.0,energy_control=1.0, duration_control=1.0):
    (preprocess_config, model_config, train_config) = configs
    if model_config["mode"] == "FastSpeech2":
        with torch.no_grad():
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )

    elif model_config["mode"] == "FastSpeech2_MI1":
        with torch.no_grad():
            _dh = diffusion_hyperparams
            T = _dh["T"]
            x = std_normal(batch[6].shape)
            for t in range(T - 1, -1, -1):
                diffusion_steps = (t * torch.ones((batch[3].shape[0], 1, 1)).type(torch.long))
                y = model(
                    *(batch[2:]),
                    p_control=pitch_control,
                    e_control=energy_control,
                    d_control=duration_control,
                    diffusion_steps=diffusion_steps.view(batch[3].shape[0], 1),
                    transformed_X=x
                )
                x = y[0]
                x= transforme_mel(diffusion_steps-1, diffusion_hyperparams, x)
            output = y

    elif model_config["mode"] == "FastSpeech2_MI2":
        with torch.no_grad():
           _dh = diffusion_hyperparams
           T = _dh["T"]
           x = std_normal(batch[6].shape)
           diffusion_steps = ((T - 1) * torch.ones((batch[3].shape[0], 1)).type(torch.long))
           y = model(
               *(batch[2:]),
               p_control=pitch_control,
               e_control=energy_control,
               d_control=duration_control,
               diffusion_steps=diffusion_steps.view(batch[3].shape[0], 1),
               transformed_X=x
           )
           output = y

    elif model_config["mode"] == "FastSpeech2_MI3":
        with torch.no_grad():
           _dh = diffusion_hyperparams
           T = _dh["T"]
           x = std_normal(batch[6].shape)
           diffusion_steps = ((T - 1) * torch.ones((batch[3].shape[0], 1)).type(torch.long))
           y = model(
               *(batch[2:]),
               p_control=pitch_control,
               e_control=energy_control,
               d_control=duration_control,
               diffusion_steps=diffusion_steps.view(batch[3].shape[0], 1),
               transformed_X=x
           )
           output = y

    elif model_config["mode"] == "FastSpeech2_MI4":
        with torch.no_grad():
           _dh = diffusion_hyperparams
           T = _dh["T"]
           x = std_normal(batch[6].shape)
           for t in range(T - 1, -1, -1):
               diffusion_steps = (t * torch.ones((batch[3].shape[0], 1, 1)).type(torch.long))
               y = model(
                   *(batch[2:]),
                   p_control=pitch_control,
                   e_control=energy_control,
                   d_control=duration_control,
                   diffusion_steps=diffusion_steps.view(batch[3].shape[0], 1),
                   transformed_X=x
               )
               x = y[0]
               x = transforme_mel(diffusion_steps - 1, diffusion_hyperparams, x)
           output = y

    elif model_config["mode"] == "FastSpeech2_MI5":
        with torch.no_grad():
           _dh = diffusion_hyperparams
           T = _dh["T"]
           x = None
           for t in range(T - 1, -1, -1):
               diffusion_steps = (t * torch.ones((batch[3].shape[0], 1, 1)).type(torch.long))
               y = model(
                   *(batch[2:]),
                   p_control=pitch_control,
                   e_control=energy_control,
                   d_control=duration_control,
                   diffusion_steps=diffusion_steps.view(batch[3].shape[0], 1),
                   transformed_X=x
               )
               x = y[0]
               Gauss = y[11]
               x = transforme_mel(diffusion_steps - 1, diffusion_hyperparams, x, z=Gauss)
           output = y

    else:
        raise ValueError("mode is None")

    return output, batch