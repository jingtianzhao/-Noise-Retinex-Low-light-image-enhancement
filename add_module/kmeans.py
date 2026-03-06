import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str):
    name = name.lower()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported act: {name}")


class CBS(nn.Module):
    """Conv + BN + Act (YOLO-style basic block)"""
    def __init__(self, c1, c2, k=1, s=1, p=None, act="silu"):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = _act(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DKCA(nn.Module):
    """
    Prototype-based Soft Clustering Attention (PSCA-style, QKV with prototype aggregation):
      - Channel prototype attention:
          q: global descriptor (GAP+GMP) -> B×d
          k,v: learnable prototypes -> Kc×d
          A = softmax(q k^T) -> B×Kc
          z = A v -> B×d
          gate_c = sigmoid(MLP([q,z])) -> B×C×1×1
      - Spatial prototype attention:
          q: pixel embeddings -> B×(HW)×d
          k,v: learnable prototypes -> Ks×d
          A = softmax(q k^T) -> B×(HW)×Ks
          z = A v -> B×(HW)×d -> reshape B×d×H×W
          gate_s = sigmoid(conv1x1(z)) -> B×1×H×W
    """

    def __init__(
        self,
        c: int,
        kc: int = 8,
        ks: int = 8,
        reduction: int = 16,
        temp_c: float = 1.0,
        temp_s: float = 1.0,
        act: str = "silu",
        use_gap_gmp: bool = True,
        alpha: float = 0.5,  # channel branch weight
        beta: float = 0.5,   # spatial branch weight
    ):
        super().__init__()
        assert c > 0
        self.c = c
        self.kc = kc
        self.ks = ks
        self.temp_c = temp_c
        self.temp_s = temp_s
        self.use_gap_gmp = use_gap_gmp
        self.alpha = alpha
        self.beta = beta

        d = max(c // reduction, 8)  # embedding dim (use a bit larger than 4 for stability)

        # -------- Channel branch (global q) --------
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # q from pooled descriptor
        in_qc = 2 * c if use_gap_gmp else c
        self.ch_q = CBS(in_qc, d, k=1, act=act)

        # learnable prototypes for channel attention (keys & values)
        self.ch_proto = nn.Parameter(torch.randn(kc, d) * 0.02)  # Kc×d

        # map [q, z] -> channel gate (B×C)
        self.ch_mlp = nn.Sequential(
            nn.Linear(2 * d, c, bias=True),
            nn.Sigmoid()
        )

        # -------- Spatial branch (pixel q) --------
        # pixel embedding q: B×d×H×W
        self.sp_q = CBS(c, d, k=1, act=act)

        # learnable prototypes for spatial attention (keys & values)
        self.sp_k = nn.Parameter(torch.randn(ks, d) * 0.02)  # Ks×d
        self.sp_v = nn.Parameter(torch.randn(ks, d) * 0.02)  # Ks×d

        # map aggregated pixel descriptor z_s (B×d×H×W) -> spatial gate (B×1×H×W)
        self.sp_gate = nn.Sequential(
            nn.Conv2d(d, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # -------- output projection --------
        self.out_proj = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            _act(act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        assert c == self.c, f"Channel mismatch: got {c}, expect {self.c}"

        # =========================
        # 1) Channel prototype attention (global)
        # =========================
        if self.use_gap_gmp:
            desc = torch.cat([self.gap(x), self.gmp(x)], dim=1)  # B×2C×1×1
        else:
            desc = self.gap(x)                                  # B×C×1×1

        q_c = self.ch_q(desc).flatten(1)                         # B×d

        # normalize for cosine-like similarity (optional but usually stabilizes)
        qn = F.normalize(q_c, dim=1)                             # B×d
        kn = F.normalize(self.ch_proto, dim=1)                   # Kc×d

        # logits: B×Kc
        logits_c = torch.matmul(qn, kn.t())
        A_c = F.softmax(logits_c / max(self.temp_c, 1e-6), dim=1)  # B×Kc

        # prototype aggregation (the key change vs your old DKCA)
        # z_c: B×d
        z_c = torch.matmul(A_c, self.ch_proto)

        # channel gate: B×C×1×1
        u_c = torch.cat([q_c, z_c], dim=1)                       # B×2d
        gate_c = self.ch_mlp(u_c).view(b, c, 1, 1)               # B×C×1×1

        x_c = x * gate_c

        # =========================
        # 2) Spatial prototype attention (pixel-wise)
        # =========================
        q_s = self.sp_q(x_c)                                     # B×d×H×W (use x_c to couple channel→spatial)
        q_s_flat = q_s.flatten(2).transpose(1, 2)                # B×(HW)×d

        qsn = F.normalize(q_s_flat, dim=2)                       # B×(HW)×d
        ksn = F.normalize(self.sp_k, dim=1)                      # Ks×d

        # logits: B×(HW)×Ks
        logits_s = torch.matmul(qsn, ksn.t())
        A_s = F.softmax(logits_s / max(self.temp_s, 1e-6), dim=2)  # soft assignment over Ks

        # aggregation: B×(HW)×d
        z_s = torch.matmul(A_s, self.sp_v)

        # reshape: B×d×H×W
        z_s_map = z_s.transpose(1, 2).reshape(b, -1, h, w)

        # spatial gate: B×1×H×W
        gate_s = self.sp_gate(z_s_map)
        x_s = x_c * gate_s

        # =========================
        # 3) Combine (residual-friendly)
        # =========================
        out = x + x_c + x_s
        out = self.out_proj(out)
        return out