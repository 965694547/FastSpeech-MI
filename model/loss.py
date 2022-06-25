import torch
import torch.nn as nn
from utils.tools import transforme_mel


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions, diffusion_hyperparams, configs):
        (preprocess_config, model_config, train_config) = configs

        if model_config["mode"] == "FastSpeech2":
            (
                mel_targets,
                _,
                _,
                pitch_targets,
                energy_targets,
                duration_targets,
            ) = inputs[6:]
            (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                _,
                src_masks,
                mel_masks,
                _,
                _,
            ) = predictions
        elif model_config["mode"] == "FastSpeech2_MI1":
            (
                mel_targets,
                _,
                _,
                pitch_targets,
                energy_targets,
                duration_targets,
            ) = inputs[6:]
            (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                _,
                src_masks,
                mel_masks,
                _,
                _,
                diffusion_steps,
            ) = predictions
        elif model_config["mode"] == "FastSpeech2_MI2":
            (
                mel_targets,
                _,
                _,
                pitch_targets,
                energy_targets,
                duration_targets,
            ) = inputs[6:]
            (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                _,
                src_masks,
                mel_masks,
                _,
                _,
                diffusion_steps,
            ) = predictions
            diffusion_steps -= 1
        elif model_config["mode"] == "FastSpeech2_MI3":
            (
                mel_targets,
                _,
                _,
                pitch_targets,
                energy_targets,
                duration_targets,
            ) = inputs[6:]
            (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                _,
                src_masks,
                mel_masks,
                _,
                _,
                diffusion_steps,
            ) = predictions
        elif model_config["mode"] == "FastSpeech2_MI4":
            (
                mel_targets,
                _,
                _,
                pitch_targets,
                energy_targets,
                duration_targets,
            ) = inputs[6:]
            (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                _,
                src_masks,
                mel_masks,
                _,
                _,
                diffusion_steps,
            ) = predictions
        elif model_config["mode"] == "FastSpeech2_MI5":
            (
                mel_targets,
                _,
                _,
                pitch_targets,
                energy_targets,
                duration_targets,
            ) = inputs[6:]
            (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                _,
                src_masks,
                mel_masks,
                _,
                _,
                diffusion_steps,
                Gauss
            ) = predictions
        else:
            raise ValueError("mode is None")

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False


        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        if model_config["mode"] == "FastSpeech2":
            mel_predictions = mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        elif model_config["mode"] == "FastSpeech2_MI1":
            diffusion_steps -= 1
            postnet_mel_predictions_T = transforme_mel(diffusion_steps, diffusion_hyperparams, postnet_mel_predictions)
            mel_predictions_T = transforme_mel(diffusion_steps, diffusion_hyperparams, mel_predictions)
            mel_targets_T = transforme_mel(diffusion_steps, diffusion_hyperparams, mel_targets)
            mel_predictions_T = mel_predictions_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            postnet_mel_predictions_T = postnet_mel_predictions_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets_T = mel_targets_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_loss = self.mae_loss(mel_predictions_T, mel_targets_T)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions_T, mel_targets_T)

        elif model_config["mode"] == "FastSpeech2_MI2":
            diffusion_steps -= 1
            mel_predictions_T = transforme_mel(diffusion_steps, diffusion_hyperparams, mel_predictions)
            mel_targets_T = transforme_mel(diffusion_steps, diffusion_hyperparams, mel_targets)
            mel_predictions_T = mel_predictions_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets_T = mel_targets_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_loss = self.mae_loss(mel_predictions_T, mel_targets_T)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        elif model_config["mode"] == "FastSpeech2_MI3":
            mel_predictions = mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        elif model_config["mode"] == "FastSpeech2_MI4":
            diffusion_steps -= 1
            mel_predictions_T = transforme_mel(diffusion_steps, diffusion_hyperparams, mel_predictions)
            mel_targets_T = transforme_mel(diffusion_steps, diffusion_hyperparams, mel_targets)
            mel_predictions_T = mel_predictions_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets_T = mel_targets_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_loss_T = self.mae_loss(mel_predictions_T, mel_targets_T)
            mel_predictions = mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_loss = self.mae_loss(mel_predictions, mel_targets) + mel_loss_T
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        elif model_config["mode"] == "FastSpeech2_MI5":
            diffusion_steps -= 1
            postnet_mel_predictions_T = transforme_mel(diffusion_steps, diffusion_hyperparams, postnet_mel_predictions, z=Gauss)
            mel_predictions_T = transforme_mel(diffusion_steps, diffusion_hyperparams, mel_predictions, z=Gauss)
            mel_targets_T = transforme_mel(diffusion_steps, diffusion_hyperparams, mel_targets, z=Gauss)
            mel_predictions_T = mel_predictions_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            postnet_mel_predictions_T = postnet_mel_predictions_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets_T = mel_targets_T.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_loss = self.mae_loss(mel_predictions_T, mel_targets_T)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions_T, mel_targets_T)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
