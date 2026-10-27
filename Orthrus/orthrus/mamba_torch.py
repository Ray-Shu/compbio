"""Pure-PyTorch stand-ins for the mamba-ssm pieces used by orthrus.mamba.

mamba-ssm and causal-conv1d only ship CUDA/Triton kernels, so they cannot be
installed on macOS. These classes mirror mamba_ssm.modules.mamba_simple.Mamba
and mamba_ssm.modules.block.Block (non-fused path) with identical parameter
names, so checkpoints trained with the CUDA kernels load unchanged. They run on
CPU or MPS, and are much slower than the fused kernels: use them for inference
and debugging, not for training.

Upstream: https://github.com/state-spaces/mamba
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RMSNorm = nn.RMSNorm


class Mamba(nn.Module):
    """Selective SSM block, matching mamba_ssm's Mamba (v1) with no inference cache."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # accepted for signature compatibility, unused
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialise dt so softplus(dt_bias) falls in [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (B, L, D). Returns the same shape."""
        if inference_params is not None:
            raise NotImplementedError("step-wise inference cache is not supported by the PyTorch fallback")
        seqlen = hidden_states.shape[1]

        xz = self.in_proj(hidden_states).transpose(1, 2)  # (B, 2*d_inner, L)
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(x.transpose(1, 2))  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (dt @ self.dt_proj.weight.t()).transpose(1, 2)  # bias is added inside the scan
        A = -torch.exp(self.A_log.float())

        y = selective_scan(
            x, dt, A, B.transpose(1, 2), C.transpose(1, 2), self.D.float(), z, self.dt_proj.bias.float()
        )
        return self.out_proj(y.transpose(1, 2))


def selective_scan(u, delta, A, B, C, D, z, delta_bias, chunk_size=256):
    """Reference selective scan (mamba_ssm.ops.selective_scan_interface.selective_scan_ref).

    u, delta, z: (B, D, L). A: (D, N). B, C: (B, N, L). D, delta_bias: (D,).

    The discretised A and B*u terms are (B, D, L, N), which for a 10 kb transcript
    is several GB, so they are built one chunk of positions at a time.
    """
    dtype_in = u.dtype
    u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
    delta = F.softplus(delta + delta_bias[:, None])

    batch, dim, seqlen = u.shape
    h = u.new_zeros(batch, dim, A.shape[1])
    ys = []
    for start in range(0, seqlen, chunk_size):
        end = min(start + chunk_size, seqlen)
        dt = delta[..., start:end]
        dA = torch.exp(torch.einsum("bdl,dn->bdln", dt, A))
        dBu = torch.einsum("bdl,bnl,bdl->bdln", dt, B[..., start:end], u[..., start:end])
        hs = []
        for i in range(end - start):
            h = dA[:, :, i] * h + dBu[:, :, i]
            hs.append(h)
        ys.append(torch.einsum("bdln,bnl->bdl", torch.stack(hs, dim=2), C[..., start:end]))

    y = torch.cat(ys, dim=-1) + u * D[:, None]
    y = y * F.silu(z.float())
    return y.to(dtype_in)


class Block(nn.Module):
    """Pre-norm residual block, matching mamba_ssm's Block without fused add+norm.

    fused_add_norm is accepted but computed unfused; the result is the same.
    """

    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None

    def forward(self, hidden_states, residual=None, inference_params=None, **mixer_kwargs):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            residual = hidden_states + residual
            hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
