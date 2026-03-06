from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class dwt_concat3(nn.Module):
    def __init__(
        self,
        c1: Union[int, List[int], Tuple[int, ...]],
        c2: int,
        dim: int = 1,
        resize_mode: str = "nearest",
        auto_pad: bool = True,
        align_to: str = "max",  # "max" or "x1"
    ):
        super().__init__()
        self.dim = dim
        self.resize_mode = resize_mode
        self.auto_pad = auto_pad
        self.c2 = c2
        self.align_to = align_to

        self.align_convs = nn.ModuleDict()
        self.out_conv = None  # lazy build

    def _align_channels(self, x: torch.Tensor, cout: int) -> torch.Tensor:
        cin = x.shape[1]
        if cin == cout:
            return x
        key = f"{cin}->{cout}"
        if key not in self.align_convs:
            self.align_convs[key] = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        return self.align_convs[key](x)

    @staticmethod
    def _pad_to_even_hw(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    @staticmethod
    def dwt(x: torch.Tensor):
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

    @staticmethod
    def idwt(A, B, C, D):
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

    @staticmethod
    def ffm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mask = (torch.abs(a) >= torch.abs(b)).to(a.dtype)
        return mask * a + (1.0 - mask) * b

    def _get_out_conv(self, in_ch: int, device, dtype):
        if self.out_conv is None or self.out_conv.in_channels != in_ch or self.out_conv.out_channels != self.c2:
            self.out_conv = nn.Conv2d(in_ch, self.c2, 1, 1, 0, bias=False).to(device=device, dtype=dtype)
        return self.out_conv

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        assert isinstance(x, (list, tuple)) and len(x) == 3, "Input must be [x1, x2, x3]"
        x1, x2, x3 = x

        # ✅ 1) 目标空间尺寸：以 x1 为准（x1/x2 不放大！）
        target_hw = x1.shape[-2:]

        # x2 理论上应与 x1 同尺度；若偶尔不一致，只把它 resize 到 x1（仍然不放大到 x3）
        if x2.shape[-2:] != target_hw:
            x2 = F.interpolate(x2, size=target_hw, mode=self.resize_mode)

        # ✅ 只降采样 x3 到 x1 的空间尺寸（省显存关键）
        if x3.shape[-2:] != target_hw:
            x3 = F.interpolate(x3, size=target_hw, mode=self.resize_mode)

        # ✅ 2) 通道对齐：建议对齐到 maxC（更信息保守），也可以对齐到 x1 的C（更省算力）
        if self.align_to == "x1":
            cout = x1.shape[1]
        else:
            cout = max(x1.shape[1], x2.shape[1], x3.shape[1])

        x1a = self._align_channels(x1, cout)
        x2a = self._align_channels(x2, cout)
        x3a = self._align_channels(x3, cout)

        # ✅ 3) pad for DWT
        if self.auto_pad:
            x1p, (ph, pw) = self._pad_to_even_hw(x1a)
            x2p, _ = self._pad_to_even_hw(x2a)
            x3p, _ = self._pad_to_even_hw(x3a)
        else:
            x1p, x2p, x3p = x1a, x2a, x3a
            ph = pw = 0
            assert x1p.shape[-2] % 2 == 0 and x1p.shape[-1] % 2 == 0, "H/W must be even for DWT"

        # ✅ 4) DWT + fusion
        A1, B1, C1, D1 = self.dwt(x1p)
        A2, B2, C2, D2 = self.dwt(x2p)
        A3, B3, C3, D3 = self.dwt(x3p)

        A = (A1 + A2 + A3) / 3.0
        B = self.ffm(self.ffm(B1, B2), B3)
        C = self.ffm(self.ffm(C1, C2), C3)
        D = self.ffm(self.ffm(D1, D2), D3)

        x_rec = self.idwt(A, B, C, D)

        # ✅ 5) crop back
        if self.auto_pad and (ph != 0 or pw != 0):
            x_rec = x_rec[:, :, :target_hw[0], :target_hw[1]]

        # ✅ 6) concat 在 x1 尺度上进行 -> 固定输出通道 c2
        y = torch.cat([x_rec, x2a, x3a], dim=self.dim)  # channels = 3*cout
        y = self._get_out_conv(y.shape[1], y.device, y.dtype)(y)
        return y
