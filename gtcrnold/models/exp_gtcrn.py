from typing import Literal
import torch
from torch import nn
from models.exp_enhancement import EnhancementExp 


class GTCRNExp(EnhancementExp):

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float,
                 weight_decay: float,
                 loss_alpha: float,
                 stft_length: int,
                 stft_shift: int,
                 cirm_comp_K: float,
                 cirm_comp_C: float,
                 reference_channel: int = 0,
                 **kwargs):
        """
        Initializes the GTCRN Experiment wrapper.
        This class is nearly identical to JNFExp, leveraging the common logic
        from the EnhancementExp base class.
        """
        super(GTCRNExp, self).__init__(model=model, cirm_comp_K=cirm_comp_K, cirm_comp_C=cirm_comp_C, **kwargs)

        # Save hyperparameters for easy access and logging
        self.save_hyperparameters('learning_rate', 'weight_decay', 'loss_alpha',
                                  'stft_length', 'stft_shift', 'cirm_comp_K', 'cirm_comp_C')
                                  
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_alpha = loss_alpha
        self.stft_length = stft_length
        self.stft_shift = stft_shift
        self.reference_channel = reference_channel

    def forward(self, input_stft):
        """
        Defines the forward pass of the experiment.

        :param input_stft: The stacked complex spectrogram [B, 2, F, T].
        :return: The estimated mask from the model.
        """
        speech_mask = self.model(input_stft)
        return speech_mask

    def shared_step(self, batch, batch_idx, stage: Literal['train', 'val']):
        """
        The main logic for a single training or validation step.
        """
        # 1. Unpack data and convert to STFT domain
        noisy_td, clean_td, noise_td = batch['noisy_td'], batch['clean_td'], batch['noise_td']
        noisy_stft, clean_stft, noise_stft = self.get_stft_rep(noisy_td, clean_td, noise_td)

        # 2. Prepare model input by stacking real and imaginary parts
        stacked_noisy_stft = torch.cat((torch.real(noisy_stft), torch.imag(noisy_stft)), dim=1)

        # 3. Get the estimated complex mask from the GTCRN model
        # Our adapted GTCRN directly outputs a stacked mask of shape [B, 2, F, T]
        stacked_speech_mask = self.forward(stacked_noisy_stft)
        speech_mask, noise_mask = self.get_complex_masks_from_stacked(stacked_speech_mask)

        # 4. Apply masks to get estimated clean and noise signals (reference channel only)
        ref_noisy_stft = noisy_stft[:, self.reference_channel, ...]
        ref_clean_stft = clean_stft[:, self.reference_channel, ...]
        ref_noise_stft = noise_stft[:, self.reference_channel, ...]

        est_clean_stft = ref_noisy_stft * speech_mask
        est_noise_stft = ref_noisy_stft * noise_mask

        # 5. Convert estimates back to time domain
        ref_clean_td = clean_td[:, self.reference_channel, ...]
        ref_noise_td = noise_td[:, self.reference_channel, ...]
        est_clean_td, est_noise_td = self.get_td_rep(est_clean_stft, est_noise_stft)

        # --- FIX: Truncate the ground truth signals to match the estimated length ---
        output_len = est_clean_td.shape[-1]
        ref_clean_td = ref_clean_td[..., :output_len]
        ref_noise_td = ref_noise_td[..., :output_len]
        # --------------------------------------------------------------------------

        # 6. Compute the loss with tensors of matching sizes
        clean_td_loss, noise_td_loss, clean_mag_loss, noise_mag_loss = self.loss(
            ref_clean_td, est_clean_td,
            ref_noise_td, est_noise_td,
            ref_clean_stft, est_clean_stft,
            ref_noise_stft, est_noise_stft
        )
        loss = torch.mean(self.loss_alpha * (clean_td_loss + noise_td_loss) + (clean_mag_loss + noise_mag_loss))

        # 7. Logging
        on_step = False
        self.log(f'{stage}/loss', loss, on_step=on_step, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/noise_td_loss', noise_td_loss.mean(), on_step=on_step, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/clean_td_loss', clean_td_loss.mean(), on_step=on_step, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/clean_mag_loss', clean_mag_loss.mean(), on_step=on_step, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/noise_mag_loss', noise_mag_loss.mean(), on_step=on_step, on_epoch=True, logger=True, sync_dist=True)

        # Log detailed audio and spectrograms for the first batch only
        if batch_idx == 0:
            self.log_batch_detailed_audio(noisy_td[:, self.reference_channel, ...], est_clean_td, batch_idx, stage)
            self.log_batch_detailed_spectrograms(
                [ref_noisy_stft, ref_clean_stft, est_clean_stft],
                batch_idx,
                stage,
                n_samples=4
            )

        if stage == 'val':
            # Use a separate metric for monitoring and checkpointing
            self.log(f'monitor_loss', loss, on_step=False, on_epoch=True, logger=True)
            # Calculate and log SI-SDR for validation
            si_sdr = self.compute_global_si_sdr(est_clean_td, ref_clean_td)
            self.log('val/si_sdr', si_sdr.mean(), on_epoch=True, logger=True, sync_dist=True)

        return loss