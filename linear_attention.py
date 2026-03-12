import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_attention(q, k, v, eps=1e-6):
    """
    Computes linear attention in O(N) time/space.
    Shapes: (B, H, N, D) -> Batch, Heads, Seq_Len, Dim
    In proteins, D << N
    """
    # 1. Apply non-negative feature map (phi)
    # Using ELU + 1 is a common stable choice for linear kernels
    q = F.elu(q) + 1
    k = F.elu(k) + 1

    # 2. Compute the 'KV' context matrix: O(N * D^2)
    # Instead of (N x N), we compute a (D x D) context
    # k: (B, H, N, D) -> k.transpose: (B, H, D, N)
    # context shape: (B, H, D, D)
    context = torch.einsum("bhnd, bhne -> bhde", k, v)

    # 3. Compute the denominator (normalizer): O(N * D)
    # k_sum: (B, H, D)
    k_sum = k.sum(dim=-2)
    # denom: (B, H, N)
    denom = torch.einsum("bhnd, bhd -> bhn", q, k_sum)

    # 4. Compute the final weighted sum: O(N * D^2)
    # out: (B, H, N, D)
    num = torch.einsum("bhnd, bhde -> bhne", q, context)

    # 5. Normalize (add eps to avoid div by zero)
    out = num / (denom.unsqueeze(-1) + eps)

    return out
