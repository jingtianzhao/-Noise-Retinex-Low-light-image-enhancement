import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel2d(ks: int, sigma: float, device, dtype):
    # ks must be odd
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


class DoGGradEnhance(nn.Module):
    """
    DoG + gradient magnitude attention for feature enhancement.
    - Works on feature maps (B,C,H,W)
    - No dependency on kornia
    """
    def __init__(self,channel_in:int, channels_out: int, ksize: int = 5, sigma1: float = 1.0, sigma2: float = 2.0, alpha: float = 1.0):
        super().__init__()
        assert ksize % 2 == 1, "ksize must be odd"
        self.channels = channel_in
        self.ksize = ksize
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha

        # gate: mag -> 1-channel attention -> expand to C by broadcast
        self.gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # register fixed sobel kernels as buffers
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        # gaussian kernels as buffers (created lazily to match device/dtype)
        self.register_buffer("gk1", torch.empty(0), persistent=False)
        self.register_buffer("gk2", torch.empty(0), persistent=False)

    def _ensure_gaussian(self, x: torch.Tensor):
        # create kernels on correct device/dtype
        if self.gk1.numel() == 0 or self.gk1.device != x.device or self.gk1.dtype != x.dtype:
            k1 = _gaussian_kernel2d(self.ksize, self.sigma1, x.device, x.dtype).view(1, 1, self.ksize, self.ksize)
            k2 = _gaussian_kernel2d(self.ksize, self.sigma2, x.device, x.dtype).view(1, 1, self.ksize, self.ksize)
            self.gk1 = k1
            self.gk2 = k2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        """
        self._ensure_gaussian(x)

        # work on single-channel proxy for gradient (reduce channel noise)
        # you can also try mean or max; mean is stable
        x_gray = x.mean(dim=1, keepdim=True)

        # Gaussian blur
        pad = self.ksize // 2
        g1 = F.conv2d(x_gray, self.gk1, padding=pad)
        g2 = F.conv2d(x_gray, self.gk2, padding=pad)

        # DoG
        dog = g1 - g2

        # Sobel gradients on DoG
        gx = F.conv2d(dog, self.sobel_x.to(device=x.device, dtype=x.dtype), padding=1)
        gy = F.conv2d(dog, self.sobel_y.to(device=x.device, dtype=x.dtype), padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)  # (B,1,H,W)

        # attention gate
        att = self.gate(mag)  # (B,1,H,W)

        # enhance (broadcast att to channels)
        return x * (1.0 + self.alpha * att)
