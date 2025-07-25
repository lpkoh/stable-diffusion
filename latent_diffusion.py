"""
Implementation of latent diffusion models

Latent diffusion models use an auto-encoder to map between image space and latent space.
The diffusion model works on the latent space, which makes it a lot easier to train.

They use a pre-trained auto-encoder and train the diffusion U-Net on the latent space of the pre-trained auto-encoder.
"""

from typing import List
import torch
import torch.nn as nn
from labml_nn.diffusion.stable_diffusion.model.autoencoder import Autoencoder
from labml_nn.diffusion.stable_diffusion.model.clip_embedder import CLIPTextEmbedder
from labml_nn.diffusion.stable_diffusion.model.unet import UNetModel


class DiffusionWrapper(nn.Module):
    """
    This is an empty wrapper class around U-Net.
    """

    def __init__(self, diffusion_model: UNetModel):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
        return self.diffusion_model(x, time_steps, context)


class LatentDiffusion(nn.Module):
    """
    Latent diffusion model

    This contains following components:
    - AutoEncoder
    - U-Net
    - CLIP embeddings generator
    """
    model: DiffusionWrapper
    first_stage_model: Autoencoder
    cond_stage_model: CLIPTextEmbedder

    def __init__(self,
                 unet_model: UNetModel,
                 autoencoder: Autoencoder,
                 clip_embedder: CLIPTextEmbedder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
                 ):
        """
        unet_model: is the U-Net that predicts noise epsilon_cond(x_t, c), in latent space
        autoencoder: is the AutoEncoder
        clip_embedder: is the CLIP embeddings generator
        latent_scaling_factor: is the scaling factor for the latent space. The encodings of the autoencoder are scaled by this before feeding into the U-Net.
        n_steps: is the number of diffusion steps T.
        linear_start: is the start of the beta schedule.
        linear_end: is the end of the beta schedule.
        """
        super().__init__()
        # Wrap the U-Net
        self.model = DiffusionWrapper(unet_model)
        # Auto-encoder and scaling factor
        self.first_stage_model = autoencoder
        self.latent_scaling_factor = latent_scaling_factor
        # CLIP embeddings generator
        self.cond_stage_model = clip_embedder

        # Number of steps T
        self.n_steps = n_steps

        # beta schedule
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        # alpha_t = 1 - beta_t
        alpha = 1. - beta
        # ᾱ_t = ∏(s=1 to t) α_s
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    @property
    def device(self):
        """
        Get model device
        """
        return next(iter(self.model.parameters())).device

    def get_text_conditioning(self, prompts: List[str]):
        """
        Get CLIP embeddings for a list of text prompts
        """
        return self.cond_stage_model(prompts)

    def autoencoder_encode(self, image: torch.Tensor):
        """
        Get scaled latent space representation of the image

        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        return self.latent_scaling_factor * self.first_stage_model.encode(image).sample()

    def autoencoder_decode(self, z: torch.Tensor):
        """
        Get image from the latent representation

        We scale down by the scaling factor and then decode.
        """
        return self.first_stage_model.decode(z / self.latent_scaling_factor)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        Predict noise

        Predict noise given the latent representation x_t, time step t, and the conditioning context c.

        epsilon_cond(x_t, c)
        """
        return self.model(x, t, context)