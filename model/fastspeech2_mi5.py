import os
import json

import torch
import torch.nn as nn

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths, calc_diffusion_step_embedding, std_normal


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x.transpose(2, 1)).transpose(2, 1)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x.transpose(2, 1)).transpose(2, 1)
        return out


class FastSpeech2_MI5(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2_MI5, self).__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config

        self.conv_1 = nn.Sequential(
            Conv(
                self.preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                self.model_config["transformer"]["decoder_hidden"],
                kernel_size=1),
            nn.ReLU()
        )
        self.fc_t1 = nn.Linear(
            self.model_config["step_embed"]["diffusion_step_embed_dim_in"],
            self.model_config["step_embed"]["diffusion_step_embed_dim_mid"]
        )
        self.fc_t2 = nn.Linear(
            self.model_config["step_embed"]["diffusion_step_embed_dim_mid"],
            self.model_config["step_embed"]["diffusion_step_embed_dim_out"]
        )
        self.fc_t = nn.Linear(
            self.model_config["step_embed"]["diffusion_step_embed_dim_out"],
            self.model_config["transformer"]["encoder_hidden"]
        )

        self.encoder = Encoder(self.model_config)
        self.variance_adaptor = VarianceAdaptor(self.preprocess_config, self.model_config)

        self.model_config_MI = self.model_config
        self.model_config_MI["transformer"]["encoder_layer"] = 1
        self.encoder_MI = Encoder(self.model_config_MI)

        self.decoder = Decoder(self.model_config)
        self.mel_linear = nn.Linear(
            self.model_config["transformer"]["decoder_hidden"],
            self.preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        self.preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                self.model_config["transformer"]["encoder_hidden"],
            )

    def forward(
            self,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels=None,
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
            diffusion_steps=None,
            transformed_X=None,
            z=None
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        B, C = mel_masks.shape
        if transformed_X == None:
            z = std_normal((B, C, self.preprocess_config["preprocessing"]["mel"]["n_mel_channels"]))
            transformed_X = z
        transformed_X = self.conv_1(transformed_X)
        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps,
            self.model_config["step_embed"]["diffusion_step_embed_dim_in"]
        )
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, 1, self.model_config["transformer"]["decoder_hidden"]])
        transformed_X = part_t + transformed_X

        transformed_X = self.encoder_MI(transformed_X, mel_masks, mel_encoder=True)
        output = output + transformed_X
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            diffusion_steps.view(B, 1, 1),
            z
        )