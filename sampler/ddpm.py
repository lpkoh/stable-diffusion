"""
Denoising Diffusion Probabilistic Models (DDPM) Sampling for stable diffusion model.
"""

from typing import Optional, List
import numpy as np
import torch
from labml import monit
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.sampler import DiffusionSampler


class DDPMSampler(DiffusionSampler):
    """
    DDPM Sampler

    This extends the DiffusionSampler base class

    DDPM samples images by repeatedly removing noise by sampling step by step from p_θ(x_(t-1) | x_t),

    p_θ(x_(t-1) | x_t) = N(x_(t-1); μ_θ(x_t, t), β̃_t * I)
    
    μ_t(x_t, t) = (sqrt(ᾱ_(t-1)) * β_t)/(1 - ᾱ_t) * x_0 + (sqrt(α_t) * (1 - ᾱ_(t-1)))/(1 - ᾱ_t) * x_t

    β̃_t = ((1 - ᾱ_(t-1))/(1 - ᾱ_t)) * β_t

    x_0 = (1/sqrt(ᾱ_t)) * x_t - sqrt(1/ᾱ_t - 1) * ε_θ
    """

    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion):
        """
        model: is the model to predict noise epsilon_cond(x_t, c)
        """
        super().__init__(model)

        # Sampling steps 1, 2, ..., T
        self.time_steps = np.asarray(list(range(self.n_steps)))

        with torch.no_grad():
            # ᾱ_t
            alpha_bar = self.model.alpha_bar
            # β_t schedule
            beta = self.model.beta
            # ᾱ_(t-1)
            alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.]), alpha_bar[:-1]])

            # sqrt(ᾱ)
            self.sqrt_alpha_bar = alpha_bar ** .5
            # sqrt(1 - ᾱ)
            self.sqrt_1m_alpha_bar = (1. - alpha_bar) ** .5
            # 1/sqrt(ᾱ_t)
            self.sqrt_recip_alpha_bar = alpha_bar ** -.5
            # sqrt(1/ᾱ_t - 1)
            self.sqrt_recip_m1_alpha_bar = (1 / alpha_bar - 1) ** .5

            # (1 - ᾱ_(t-1))/(1 - ᾱ_t) * β_t
            variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
            # Clamped log of β̃_t
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))
            # (sqrt(ᾱ_(t-1)) * β_t)/(1 - ᾱ_t)
            self.mean_x0_coef = beta * (alpha_bar_prev ** .5) / (1. - alpha_bar)
            # (sqrt(α_t) * (1 - ᾱ_(t-1)))/(1 - ᾱ_t)
            self.mean_xt_coef = (1. - alpha_bar_prev) * ((1 - beta) ** 0.5) / (1. - alpha_bar)

    @torch.no_grad()
    def sample(self,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: Optional[torch.Tensor] = None,
               uncond_scale: float = 1.,
               uncond_cond: Optional[torch.Tensor] = None,
               skip_steps: int = 0,
               ):
        """
        Sampling Loop

        shape: is the shape of the generated images in the form (batch_size, channels, height, width)
        cond: is the conditional embeddings c
        temperature: is the noise temperature (random noise gets multiplied by this)
        x_last: is x_T. If not provided random noise will be used.
        uncond_scale: is the unconditional guidance scale s. This is used for epsilon_theta(x_t, c) = s * epsilon_cond(x_t, c) + (s - 1) * epsilon_cond(x_t, c_u)
        uncond_cond: is the conditional embedding for empty prompt c_u
        skip_steps: is the number of time steps to skip t'. We start sampling from T - t'. And x_last is then x_(T-t').
        """

        # Get device and batch size
        device = self.model.device
        bs = shape[0]

        # Get x_T
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at T - t', T - t' - 1, ..., 1
        time_steps = np.flip(self.time_steps)[skip_steps:]

        # Sampling loop
        for step in monit.iterate('Sample', time_steps):
            # Time step t
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample x_(t-1)
            x, pred_x0, e_t = self.p_sample(x, cond, ts, step,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale,
                                            uncond_cond=uncond_cond)

        # Return x_0
        return x

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None):
        """
        Sample x_(t-1) from p_θ(x_(t-1) | x_t)

        x: is x_t of shape (batch_size, channels, height, width)
        c: is the conditional embeddings c of shape (batch_size, emb_size)
        t: is t of shape (batch_size)
        step: is the step t as an integer
        repeat_noise: specified whether the noise should be same for all samples in the batch
        temperature: is the noise temperature (random noise gets multiplied by this)
        uncond_scale: is the unconditional guidance scale s. This is used for epsilon_theta(x_t, c) = s * epsilon_cond(x_t, c) + (s - 1) * epsilon_cond(x_t, c_u)
        uncond_cond: is the conditional embedding for empty prompt c_u
        """

        # Get ε_θ
        e_t = self.get_eps(x, t, c,
                           uncond_scale=uncond_scale,
                           uncond_cond=uncond_cond)

        # Get batch size
        bs = x.shape[0]

        # 1/sqrt(ᾱ_t)
        sqrt_recip_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_alpha_bar[step])
        # sqrt(1/ᾱ_t - 1)
        sqrt_recip_m1_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[step])

        # Calculate x_0 with current ε_θ
        # x_0 = (1/sqrt(ᾱ_t)) * x_t - sqrt(1/ᾱ_t - 1) * ε_θ
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t

        # (sqrt(ᾱ_(t-1)) * β_t)/(1 - ᾱ_t)
        mean_x0_coef = x.new_full((bs, 1, 1, 1), self.mean_x0_coef[step])
        # (sqrt(α_t) * (1 - ᾱ_(t-1)))/(1 - ᾱ_t)
        mean_xt_coef = x.new_full((bs, 1, 1, 1), self.mean_xt_coef[step])

        # Calculate μ_t(x_t, t)
        # μ_t(x_t, t) = (sqrt(ᾱ_(t-1)) * β_t)/(1 - ᾱ_t) * x_0 + (sqrt(α_t) * (1 - ᾱ_(t-1)))/(1 - ᾱ_t) * x_t
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        # log(β̃_t)
        log_var = x.new_full((bs, 1, 1, 1), self.log_var[step])

        # Do not add noise when t = 1 (final step sampling process).
        # Note that step is 0 when t = 1)
        if step == 0:
            noise = 0
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]))
        # Different noise for each sample
        else:
            noise = torch.randn(x.shape)

        # Multiply noise by the temperature
        noise = noise * temperature

        # Sample from,
        # p_θ(x_(t-1) | x_t) = N(x_(t-1); μ_θ(x_t, t), β̃_t * I)
        x_prev = mean + (0.5 * log_var).exp() * noise

        # Return x_prev, x0, e_t
        return x_prev, x0, e_t

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        Sample from q(x_t|x_0)

        q(x_t|x_0) = N(x_t; sqrt(ᾱ_t) * x_0, (1 - ᾱ_t) * I)

        x0: is x_0 of shape (batch_size, channels, height, width)
        index: is the time step t index
        noise: is the noise, ε
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from N(x_t; sqrt(ᾱ_t) * x_0, (1 - ᾱ_t) * I)
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise