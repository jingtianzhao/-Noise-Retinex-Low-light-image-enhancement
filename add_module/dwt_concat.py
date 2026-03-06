from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class dwt_concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.align = None  # lazy 1x1 conv (x2 -> x1 channels)

    def _get_align(self, cin, cout, device, dtype):
        if self.align is None or self.align.in_channels != cin or self.align.out_channels != cout:
            self.align = nn.Conv2d(cin, cout, 1, 1, 0, bias=False).to(device=device, dtype=dtype)
        return self.align

    def dwt(self, x):
        x1 = x[:, :, 0::2, :] / 2
        x2 = x[:, :, 1::2, :] / 2
        x1_1 = x1[:, :, :, 0::2]
        x2_1 = x2[:, :, :, 0::2]
        x1_2 = x1[:, :, :, 1::2]
        x2_2 = x2[:, :, :, 1::2]
        A = x1_1 + x2_1 + x1_2 + x2_2
        B = -x1_1 - x2_1 + x1_2 + x2_2
        C = -x1_1 + x2_1 - x1_2 + x2_2
        D = x1_1 - x2_1 - x1_2 + x2_2
        return A, B, C, D

    def idwt(self, A, B, C, D):
        x1_1 = (A - B - C + D) / 4
        x2_1 = (A - B + C - D) / 4
        x1_2 = (A + B - C - D) / 4
        x2_2 = (A + B + C + D) / 4

        Bn, Cn, H, W = x1_1.size()
        x1 = torch.zeros(Bn, Cn, H, W * 2, device=A.device, dtype=A.dtype)
        x2 = torch.zeros(Bn, Cn, H, W * 2, device=A.device, dtype=A.dtype)

        x1[:, :, :, 0::2] = x1_1
        x1[:, :, :, 1::2] = x1_2
        x2[:, :, :, 0::2] = x2_1
        x2[:, :, :, 1::2] = x2_2

        x = torch.zeros(Bn, Cn, H * 2, W * 2, device=A.device, dtype=A.dtype)
        x[:, :, 0::2, :] = x1 * 2
        x[:, :, 1::2, :] = x2 * 2
        return x

    def ffm(self, A1, A2):
        mask = (torch.abs(A1) >= torch.abs(A2)).to(A1.dtype)
        return mask * A1 + (1 - mask) * A2

    def forward(self, x: List[torch.Tensor]):
        x1, x2 = x
        x2_raw = x2  # ✅ keep raw for concat

        # spatial align if needed
        if x1.shape[-2:] != x2.shape[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode="nearest")

        # ✅ channel align x2 -> x1 channels for DWT add
        if x2.shape[1] != x1.shape[1]:
            conv = self._get_align(x2.shape[1], x1.shape[1], x2.device, x2.dtype)
            x2 = conv(x2)

        # DWT fusion (channels equal now)
        A1, B1, C1, D1 = self.dwt(x1)
        A2, B2, C2, D2 = self.dwt(x2)

        fin_A = (A1 + A2) / 2
        fin_B = self.ffm(B1, B2)
        fin_C = self.ffm(C1, C2)
        fin_D = self.ffm(D1, D2)

        x_rec = self.idwt(fin_A, fin_B, fin_C, fin_D)

        # ✅ output channels consistent with Concat rule: C(x1) + C(x2_raw)
        return torch.cat([x_rec, x2_raw], dim=self.d)
