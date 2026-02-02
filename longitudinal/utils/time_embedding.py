import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """
    TVRL-style sinusoidal time embedding + MLP.

    Input:
      - timepoints: [B, T]  (e.g. years since baseline or [0,1,2] for screenings)

    Output:
      - time_emb:   [B, T, D] where D = dim
    """

    def __init__(self, dim, max_period=100, learnable=True):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("TimeEmbedding dim must be even (for sin/cos pairing).")

        self.dim = dim
        self.max_period = max_period

        if learnable:
            # MLP applied after sinusoidal embedding (Stable Diffusion-style)
            self.time_embed = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
        else:
            self.time_embed = nn.Identity()

    @staticmethod
    def time_step_embedding(time_steps: torch.Tensor,
                            dim: int,
                            max_period: int = 100) -> torch.Tensor:
        """
        Returns sinusoidal embeddings for 1D time steps.

        Args:
            time_steps: [N]  (flattened B*T)
            dim:        embedding dimension (must be even)
            max_period: controls frequency scale
        Output:
            [N, dim]
        """
        assert time_steps.ndim == 1, "time_steps must be 1D (flattened B*T)."
        if dim % 2 != 0:
            raise ValueError("TimeEmbedding dim must be even.")

        half = dim // 2

        # frequencies: [half]
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=time_steps.device)
            / half
        )
        # Broadcast multiply (N, 1) * (1, half) -> (N, half)
        args = time_steps.float().unsqueeze(-1) * freqs.unsqueeze(0)

        # [N, dim] = [N, half] + [N, half]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, timepoints: torch.Tensor) -> torch.Tensor:
        """
        timepoints: [B, T]
        Returns:    [B, T, D]
        """
        B, T = timepoints.shape

        # Flatten to [B*T]
        flat = timepoints.reshape(-1)

        # Sinusoidal embedding → [B*T, D]
        emb = self.time_step_embedding(flat, self.dim, self.max_period)

        # Reshape back to [B, T, D]
        emb = emb.view(B, T, self.dim)

        # Apply MLP (or Identity)
        return self.time_embed(emb)
