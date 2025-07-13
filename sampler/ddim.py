"""
Denoising Diffusion Implicit Models (DDIM) Sampling for stable diffusion model.
"""

from typing import Optional, List
import numpy as np
import torch
from labml import monit
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.sampler import DiffusionSampler


class DDIMSampler(DiffusionSampler):
    """
    DDIM Sampler

    This extends the DiffusionSampler base class

    DDIM samples images by repeatedly removing noise by sampling step by step using,

    x_τ(i-1) = sqrt(α_τ(i-1)) * (x_τi - sqrt(1 - α_τi) * ε_θ(x_τi)) / sqrt(α_τi)
          + sqrt(1 - α_τ(i-1) - σ_τi²) * ε_θ(x_τi)
          + σ_τi * ε_τi
    where ε_τi is random noise, τ is a subsequence of [1, 2, ..., T] of length S, and σ_τi = η * sqrt((1 - α_τ(i-1))/(1 - α_τi)) * sqrt(1 - α_τi/α_τ(i-1))

    Note that α_t in DDIM paper refers to ᾱ_t from DDPM.
    """

    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion, n_steps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0.):
        """
        model: is the model to predict noise epsilon_cond(x_t, c)
        n_steps: is the number of DDIM sampling steps, S
        ddim_discretize: specifies how to extract τ from [1, 2, ..., T]. It can be either uniform or quad.
        ddim_eta: is η used to calculate σ_τi. η = 0 makes the sampling process deterministic.
        """
        super().__init__(model)
        # Number of steps, T
        self.n_steps = model.n_steps

        # Calculate τ to be uniformly distributed across [1,2,...,T]
        if ddim_discretize == 'uniform':
            c = self.n_steps // n_steps
            self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1
        # Calculate τ to be quadratically distributed across [1,2,...,T]
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps * .8), n_steps)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            # Get ᾱ_t
            alpha_bar = self.model.alpha_bar

            # α_τi
            self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
            # sqrt(α_τi)
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            # α_τ(i-1)
            self.ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])

            # σ_τi = η * sqrt((1 - α_τ(i-1))/(1 - α_τi)) * sqrt(1 - α_τi/α_τ(i-1))
            self.ddim_sigma = (ddim_eta *
                               ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)

            # sqrt(1 - α_τi)
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

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
        x_last: is x_τ. If not provided random noise will be used.
        uncond_scale: is the unconditional guidance scale s. This is used for epsilon_theta(x_t, c) = s * epsilon_cond(x_t, c) + (s - 1) * epsilon_cond(x_t, c_u)
        uncond_cond: is the conditional embedding for empty prompt c_u
        skip_steps: is the number of time steps to skip i'. We start sampling from S - i'. And x_last is then x_τ(S-i').
        """

        # Get device and batch size
        device = self.model.device
        bs = shape[0]

        # Get x_τS
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at τ(S-i'), τ(S-i'-1), ..., τ₁
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in monit.enum('Sample', time_steps):
            # Index i in the list [τ₁, τ₂, ..., τS]
            index = len(time_steps) - i - 1
            # Time step τᵢ
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample x_τ(i-1)
            x, pred_x0, e_t = self.p_sample(x, cond, ts, step, index=index,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale,
                                            uncond_cond=uncond_cond)

        # Return x_0
        return x

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int, *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1.,
                 uncond_cond: Optional[torch.Tensor] = None):
        """
        Sample x_τ(i-1)

        x: is x_τi of shape (batch_size, channels, height, width)
        c: is the conditional embeddings c of shape (batch_size, emb_size)
        t: is τᵢ of shape (batch_size)
        step: is the step τᵢ as an integer
        index: is index i in the list [τ₁, τ₂, ..., τS]
        repeat_noise: specified whether the noise should be same for all samples in the batch
        temperature: is the noise temperature (random noise gets multiplied by this)
        uncond_scale: is the unconditional guidance scale s. This is used for epsilon_theta(x_t, c) = s * epsilon_cond(x_t, c) + (s - 1) * epsilon_cond(x_t, c_u)
        uncond_cond: is the conditional embedding for empty prompt c_u
        """

        # Get epsilon_theta(x_τi)
        e_t = self.get_eps(x, t, c,
                           uncond_scale=uncond_scale,
                           uncond_cond=uncond_cond)

        # Calculate x_τ(i-1) and predicted x_0
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x,
                                                      temperature=temperature,
                                                      repeat_noise=repeat_noise)

        # Return x_τ(i-1), pred_x0, e_t
        return x_prev, pred_x0, e_t

    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index: int, x: torch.Tensor, *,
                               temperature: float,
                               repeat_noise: bool):
        """
        Sample x_τ(i-1) given epsilon_theta(x_τi)
        """

        # α_τi
        alpha = self.ddim_alpha[index]
        # α_τ(i-1)
        alpha_prev = self.ddim_alpha_prev[index]
        # σ_τi
        sigma = self.ddim_sigma[index]
        # sqrt(1 - α_τi)
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        # Current prediction for x_0: (x_τi - sqrt(1 - α_τi) * epsilon_theta(x_τi)) / sqrt(α_τi)
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        # Direction pointing to x_t: sqrt(1 - α_τ(i-1) - σ_τi²) * epsilon_theta(x_τi)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        # No noise is added when η = 0
        if sigma == 0.:
            noise = 0.
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
            # Different noise for each sample
        else:
            noise = torch.randn(x.shape, device=x.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        # x_τ(i-1) = sqrt(α_τ(i-1)) * (x_τi - sqrt(1 - α_τi) * epsilon_theta(x_τi)) / sqrt(α_τi) + sqrt(1 - α_τ(i-1) - σ_τi²) * epsilon_theta(x_τi) + σ_τi * ε_τi
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        # Return x_τ(i-1), pred_x0
        return x_prev, pred_x0

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        Sample from q_σ,τ(x_τi|x_0), q_σ,τ(x_t|x_0) = N(x_t; sqrt(α_τi) * x_0, (1 - α_τi) * I)
        x0: is x_0 of shape (batch_size, channels, height, width)
        index: is the time step τᵢ index i
        noise: is the noise, ε
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from q_σ,τ(x_t|x_0) = N(x_t; sqrt(α_τi) * x_0, (1 - α_τi) * I)
        return self.ddim_alpha_sqrt[index] * x0 + self.ddim_sqrt_one_minus_alpha[index] * noise

    @torch.no_grad()
    def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        #Painting Loop

        x: is x_S' of shape (batch_size, channels, height, width)
        cond: is the conditional embeddings c
        t_start: is the sampling step to start from, S'
        orig: is the original image in latent page which we are inpainting. If this is not provided, it'll be an image to image transformation.
        mask: is the mask to keep the original image.
        orig_noise: is fixed noise to be added to the original image.
        uncond_scale: is the unconditional guidance scale s. This is used for epsilon_theta(x_t, c) = s * epsilon_cond(x_t, c) + (s - 1) * epsilon_cond(x_t, c_u)
        uncond_cond: is the conditional embedding for empty prompt c_u
        """
        # Get  batch size
        bs = x.shape[0]

        # Time steps to sample at τS', τ(S'-1), ..., τ₁
        time_steps = np.flip(self.time_steps[:t_start])

        for i, step in monit.enum('Paint', time_steps):
            # Index i in the list [τ₁, τ₂, ..., τS]
            index = len(time_steps) - i - 1
            # Time step τᵢ
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample x_τ(i-1)
            x, _, _ = self.p_sample(x, cond, ts, step, index=index,
                                    uncond_scale=uncond_scale,
                                    uncond_cond=uncond_cond)

            # Replace the masked area with original image
            if orig is not None:
                # Get the q_σ,τ(x_τi|x_0) for original image in latent space
                orig_t = self.q_sample(orig, index, noise=orig_noise)
                # Replace the masked area
                x = orig_t * mask + x * (1 - mask)

        # Return x
        return x