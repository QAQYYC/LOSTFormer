import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import random


class SpatioCausalNumerator(Function):
    @staticmethod
    def forward(ctx, qs, ks, vs, channel):
        L, B, H, M = qs.shape
        D = vs.shape[-1]

        qs = qs.reshape(L // channel, channel, B, H, M)
        ks = ks.reshape(L // channel, channel, B, H, M)
        vs = vs.reshape(L // channel, channel, B, H, D)

        sums = torch.sum(ks.view(L // channel, channel, B, H, M, 1) @ vs.view(L // channel, channel, B, H, 1, D), dim=1)
        sums = torch.cumsum(sums, dim=0)
        sums = sums.unsqueeze(1).expand(-1, channel, -1, -1, -1, -1)

        result = (qs.unsqueeze(4) @ sums).squeeze(4)

        ctx.save_for_backward(qs, ks, vs)
        ctx.sums = sums
        ctx.channel = channel
        result = result.reshape(L, B, H, D)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        qs, ks, vs = ctx.saved_tensors
        gr_sums = ctx.sums
        channel = ctx.channel
        L, B, H, D = grad_output.shape
        M = qs.shape[-1]

        grad_output = grad_output.reshape(L // channel, channel, B, H, D)
        grads = torch.sum(qs.unsqueeze(-1) @ grad_output.unsqueeze(4), dim=1)
        grads = torch.flip(torch.cumsum(torch.flip(grads, dims=(0,)), dim=0), dims=(0,))
        grads = grads.unsqueeze(1).expand(-1, channel, -1, -1, -1, -1)

        q_grads = (gr_sums @ grad_output.unsqueeze(-1)).squeeze(-1)
        v_grads = (ks.view(L // channel, channel, B, H, 1, M) @ grads).squeeze(4)
        k_grads = (grads  @ vs.unsqueeze(-1)).squeeze(-1)

        q_grads = q_grads.reshape(L, B, H, M)
        k_grads = k_grads.reshape(L, B, H, M)
        v_grads = v_grads.reshape(L, B, H, D)

        return q_grads, k_grads, v_grads, None


class SpatioCausalDenominator(Function):
    @staticmethod
    def forward(ctx, qs, ks, channel):
        L, B, H, M = qs.shape
        qs = qs.reshape(L // channel, channel, B, H, M)
        ks = ks.reshape(L // channel, channel, B, H, M)
        sums = torch.cumsum(torch.sum(ks,dim=1), dim=0)

        result = torch.sum(qs*sums.unsqueeze(1), dim=-1).reshape(L, B, H)

        ctx.save_for_backward(qs, ks)
        ctx.sums = sums
        ctx.channel = channel
        return result

    @staticmethod
    def backward(ctx, grad_output):
        qs, ks = ctx.saved_tensors
        gr_sums = ctx.sums
        channel = ctx.channel
        L, B, H = grad_output.shape
        M = qs.shape[-1]

        grad_output = grad_output.reshape(L // channel, channel, B, H)
        grads = torch.sum(qs * grad_output.unsqueeze(-1), dim=1)
        grads = torch.flip(torch.cumsum(torch.flip(grads, dims=(0,)), dim=0), dims=(0,))

        q_grads = gr_sums.unsqueeze(1) * grad_output.unsqueeze(-1)
        k_grads = grads.unsqueeze(1).repeat(1, channel, 1, 1, 1)
        q_grads = q_grads.reshape(L, B, H, M)
        k_grads = k_grads.reshape(L, B, H, M)

        return q_grads, k_grads, None

def causal_numerator(qs, ks, vs, channel=None):
    L, B, H, M = qs.shape
    D = vs.shape[-1]

    qs = qs.reshape(L // channel, channel, B, H, M)
    ks = ks.reshape(L // channel, channel, B, H, M, 1)
    vs = vs.reshape(L // channel, channel, B, H, 1, D)

    sums = torch.sum(torch.matmul(ks, vs), dim=1)
    del ks
    del vs
    sums = torch.cumsum(sums, dim=0)
    sums = sums.unsqueeze(1).expand(-1, channel, -1, -1, -1, -1)

    result = (qs.unsqueeze(4) @ sums).squeeze(4)
    del qs
    del sums

    result = result.reshape(L, B, H, D)
    return result

def causal_denominator(qs, ks, channel=None):
    L, B, H, M = qs.shape
    qs = qs.reshape(L // channel, channel, B, H, M)
    ks = ks.reshape(L // channel, channel, B, H, M)
    sums = torch.cumsum(torch.sum(ks,dim=1), dim=0)
    result = torch.sum(qs*sums.unsqueeze(1), dim=-1).reshape(L, B, H)
    del qs
    del ks
    del sums

    return result

def causal_numerator_without_optimization(qs, ks, vs, channel=None):
    if channel is None:
        raise ValueError("Channel must be specified for causal numerator computation.")
    L, B, H, M = qs.shape
    assert L % channel == 0, "Length L must be divisible by channel."

    D = vs.shape[-1]
    result = []
    sums = torch.zeros(B, H, M, D, device=qs.device, dtype=qs.dtype)

    for t in range(L // channel):
        for c in range(channel):
            sums = sums + torch.einsum('bhi,bhj->bhij', ks[t * channel + c], vs[t * channel + c])
        for c in range(channel):
            idx = t * channel + c
            out = torch.einsum('bhij,bhi->bhj', sums, qs[idx])
            result.append(out.unsqueeze(0))

    return torch.cat(result, dim=0)


def causal_denominator_without_optimization(qs, ks, channel=None):
    if channel is None:
        raise ValueError("Channel must be specified for causal denominator computation.")
    L, B, H, M = qs.shape
    assert L % channel == 0, "Length L must be divisible by channel."
    result = []
    sums = torch.zeros(B, H, M, device=qs.device, dtype=qs.dtype)

    for t in range(L // channel):
        for c in range(channel):
            idx = t * channel + c
            sums = sums + ks[idx]
        for c in range(channel):
            idx = t * channel + c
            result.append(torch.sum(qs[idx] * sums, dim=-1).unsqueeze(0))

    return torch.cat(result, dim=0)


def create_projection_matrix(m, d, seed=0, scaling=0):
    # torch.manual_seed(seed)
    nb_full_blocks = m // d
    block_list = []

    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn(d, d)
        q, _ = torch.linalg.qr(unstructured_block)
        block_list.append(q.t())

    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        unstructured_block = torch.randn(d, d)
        q, _ = torch.linalg.qr(unstructured_block)
        block_list.append(q.t()[:remaining_rows])

    final_matrix = torch.cat(block_list, dim=0)

    if scaling == 0:
        multiplier = torch.norm(torch.randn(m, d), dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(d) * torch.ones(m)
    else:
        raise ValueError("Scaling must be 0 or 1.")

    projection = torch.diag(multiplier) @ final_matrix
    return projection


def relu_kernel_transformation(data, projection_matrix, numerical_stabilizer=1e-4):
    ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
    data_dash = torch.einsum("bcpd,md->bcpm", data, projection_matrix)
    return F.relu(data_dash) * ratio + numerical_stabilizer


def softmax_kernel_transformation(data, is_query, projection_matrix, numerical_stabilizer=1e-6):
    B, P, H, D = data.shape
    M = projection_matrix.shape[0]

    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(D, dtype=torch.float32, device=data.device)))
    data = data * data_normalizer
    ratio = 1.0 / math.sqrt(M)
    data_dash = torch.einsum("bcpd,md->bcpm", data, projection_matrix)
    diag_data = torch.sum(data ** 2, dim=-1, keepdim=True) / 2.0

    if is_query:
        data_dash = ratio * torch.exp(data_dash - diag_data - data_dash.max(dim=-1, keepdim=True).values)
    else:
        data_dash = ratio * torch.exp(
            data_dash - diag_data - data_dash.amax(dim=(-1, -3), keepdim=True).expand_as(data_dash)
        )

    return data_dash + numerical_stabilizer

def qk_sampling(qs, ks, q_prime, k_prime,channel=None):
    qs = qs.permute(1, 0, 2, 3)
    ks = ks.permute(1, 0, 2, 3)
    L, B, H, d_h = qs.shape

    idx = list(range(L - channel, L))
    random.shuffle(idx)
    idx = idx[:(channel + 9) // 10]
    qs = qs[idx]
    q_prime = q_prime[idx]

    idx = list(range(L - channel, L))
    random.shuffle(idx)
    idx = idx[:(channel + 9) // 10]
    idx.append(random.randint(0, channel - 1))
    idx.append(random.randint(0, channel - 1) + L - 2 * channel)
    random.shuffle(idx)

    ks = ks[idx]
    k_prime = k_prime[idx]
    return qs, ks,q_prime, k_prime

def causual_loss_without_optimization(qs, ks, q_prime, k_prime, channel=None):
    qs, ks, q_prime, k_prime = qk_sampling(qs,ks,q_prime,k_prime,channel=channel)
    L, B, H, d_h = qs.shape

    loss = []
    sums = torch.zeros((qs.shape[0], B, H), device=qs.device)
    sums_prime = torch.zeros((qs.shape[0], B, H), device=qs.device)
    for i in range(qs.shape[0]):
        for j in range(i + ks.shape[0] - qs.shape[0]):
            sums[i] = sums[i] + torch.exp(torch.sum(qs[i] * ks[j], dim=-1))
            sums_prime[i] = sums_prime[i] + torch.sum(q_prime[i] * k_prime[j], dim=-1)

        sum = torch.zeros((B, H), device=qs.device)
        for j in range(i + ks.shape[0] - qs.shape[0]):
            sum = sum + torch.exp(torch.sum(qs[i] * ks[j], dim=-1)) * \
                   torch.log(torch.sum(q_prime[i] * k_prime[j], dim=-1) / (sums_prime[i] + 1e-5))
        sum = sum / (sums[i] + 1e-5)
        loss.append(sum.unsqueeze(0))

    loss = torch.concat(loss, dim=0)
    loss = torch.sum(loss,dim=(0,1,2))

    return loss

def test_spatio_causal_numerator():
    B, H, M, D = 7, 3, 4, 5  # 小规模测试维度
    L = 8  # 序列长度
    channel = 2  # 通道数

    # 创建随机输入数据
    qs = torch.randn(L, B, H, M, requires_grad=True, dtype=torch.double)
    ks = torch.randn(L, B, H, M, requires_grad=True, dtype=torch.double)
    vs = torch.randn(L, B, H, D, requires_grad=True, dtype=torch.double)

    print("Testing SpatioCausalNumerator...")

    # 使用自定义Function计算
    custom_num = SpatioCausalNumerator.apply(qs, ks, vs, channel)
    custom_num.sum().backward()
    custom_q_grad = qs.grad.clone()
    custom_k_grad = ks.grad.clone()
    custom_v_grad = vs.grad.clone()

    # 重置梯度
    qs.grad.zero_()
    ks.grad.zero_()
    vs.grad.zero_()

    # 使用原始循环实现计算
    ref_num = causal_numerator_without_optimization(qs, ks, vs, channel)
    ref_num.sum().backward()
    ref_q_grad = qs.grad.clone()
    ref_k_grad = ks.grad.clone()
    ref_v_grad = vs.grad.clone()

    # 比较结果
    num_diff = torch.norm(custom_num - ref_num).item()
    q_grad_diff = torch.norm(custom_q_grad - ref_q_grad).item()
    k_grad_diff = torch.norm(custom_k_grad - ref_k_grad).item()
    v_grad_diff = torch.norm(custom_v_grad - ref_v_grad).item()

    print(f"  Forward diff: {num_diff:.6e}")
    print(f"  q_grad diff: {q_grad_diff:.6e}")
    print(f"  k_grad diff: {k_grad_diff:.6e}")
    print(f"  v_grad diff: {v_grad_diff:.6e}")



    return num_diff < 1e-6 and q_grad_diff < 1e-6 and k_grad_diff < 1e-6 and v_grad_diff < 1e-6


def test_spatio_causal_denominator():
    B, H, M = 2, 3, 4  # 小规模测试维度
    L = 8  # 序列长度
    channel = 2  # 通道数

    # 创建随机输入数据
    qs = torch.randn(L, B, H, M, requires_grad=True, dtype=torch.double)
    ks = torch.randn(L, B, H, M, requires_grad=True, dtype=torch.double)

    print("Testing SpatioCausalDenominator...")

    # 使用自定义Function计算
    custom_den = SpatioCausalDenominator.apply(qs, ks, channel)
    custom_den.sum().backward()
    custom_q_grad = qs.grad.clone()
    custom_k_grad = ks.grad.clone()

    # 重置梯度
    qs.grad.zero_()
    ks.grad.zero_()

    # 使用原始循环实现计算
    ref_den = causal_denominator_without_optimization(qs, ks, channel)
    ref_den.sum().backward()
    ref_q_grad = qs.grad.clone()
    ref_k_grad = ks.grad.clone()

    # 比较结果
    den_diff = torch.norm(custom_den - ref_den).item()
    q_grad_diff = torch.norm(custom_q_grad - ref_q_grad).item()
    k_grad_diff = torch.norm(custom_k_grad - ref_k_grad).item()

    print(f"  Forward diff: {den_diff:.6e}")
    print(f"  q_grad diff: {q_grad_diff:.6e}")
    print(f"  k_grad diff: {k_grad_diff:.6e}")


    return den_diff < 1e-6 and q_grad_diff < 1e-6 and k_grad_diff < 1e-6

if __name__ == "__main__":
    torch.manual_seed(42)

    print("="*60)
    print("Starting Spatio-Temporal Attention Backpropagation Tests")
    print("="*60)

    # 运行测试
    num_passed = test_spatio_causal_numerator()
    den_passed = test_spatio_causal_denominator()

    print("\n" + "="*60)
    print("Test Summary:")
    print(f"SpatioCausalNumerator: {'PASSED' if num_passed else 'FAILED'}")
    print(f"SpatioCausalDenominator: {'PASSED' if den_passed else 'FAILED'}")
    print("="*60)
