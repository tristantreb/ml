import torch
import torch.nn as nn


class SimpleLatentDiffusion(nn.Module):
    """
    Simple latent diffusion class to infer images from text
    """

    def __init__(self, vae, text_encoder, unet, train_steps=1000):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.unet = unet

        # Define the noise schedule
        beta = torch.linspace(0.0001, 0.002, train_steps).to("cuda")
        alpha = 1.0 - beta
        # Cumulative signal (a1 x a2 .. x at).
        # How much of the original image (x_0) is still left after many steps of adding noise
        self.alpha_bar = torch.cumprod(alpha, dim=0)

    def forward(self, prompt, timesteps=50):
        # About timestep
        # 1-10 steps blurry image with missing objects
        # 20-50 steps: clear structure with sharp details
        # 100+: 2x electricity spend for a 1% improvement (diminishing returns)

        # Encode text into semantic space
        # Output shape: [batch, sequence_len, embed_dim]
        context = self.text_encoder.embed(prompt)

        # Start with pure noise
        latents = self.vae.init_latents()
        latents = torch.randn((1, 4, 64, 64)).to("cuda")

        # Denoising loop
        for (
            i,
            t,
        ) in enumerate(timesteps):
            # Score function estimation
            noise_pred = self.unet(latents, t, context)

            # Update latents
            latents = self.step_scheduler(latents, noise_pred, t, i, timesteps)

        image = self.vae.decode(latents / 0.18215)
        return image

    def step_scheduler(self, x_t, noise_pred, t, i, timesteps):
        f"""
        Simplified DDIM step: x_{t-1} calculation
        """
        a_t = self.alpha_bar[t]

        prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else 0
        a_prev = self.alpha_bar[prev_t]

        # Estimate the clean latent
        pred_x0 = (x_t - torch.sqrt(1 - a_t) * noise_pred) / torch.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - a_prev) * noise_pred

        # Combine them to get the next (cleaner) latent
        x_prev = torch.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev
