import math
import torch
import triton
import triton.language as tl

def flash_attention1_backward(Q, K, V, O, dO, L):
    scale = 1.0 / math.sqrt(Q.shape[-1])

    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()
    Of = O.float()
    dOf = dO.float()
    Lf = L.float()

    S = torch.einsum("bqd,bkd->bqk", Qf, Kf) * scale
    P = torch.exp(S - Lf[:, :, None])

    D_vec = torch.sum(dOf * Of, dim=-1, keepdim=True)
    dV = torch.einsum("bqk,bqd->bkd", P, dOf)
    dP = torch.einsum("bqd,bkd->bqk", dOf, Vf)
    dS = P * (dP - D_vec)
    dQ = torch.einsum("bqk,bkd->bqd", dS, Kf) * scale
    dK = torch.einsum("bqk,bqd->bkd", dS, Qf) * scale

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)



class FlashAttention_pytorch(torch.autograd.Function):
    '''
    Q: (B, N_q, D)
    K: (B, N_k, D)
    V: (B, N_k, D)
    is_causal: ignored for this task

    Returns:
        O: (B, N_q, D_v)
    '''
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # at least 16 * 16
        Bq = 32
        Bk = 32

        B, Nq, d = Q.shape
        _, Nv, d_v = V.shape
        _, Nk, d_k = K.shape

        assert d == d_k == d_v
        assert Nk == Nv

        scale = 1.0 / math.sqrt(d)

        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)

        O = torch.empty((B, Nq, d), device=Q.device)
        L = torch.empty((B, Nq), device=Q.device)

        Qf = Q.float()
        Kf = K.float()
        Vf = V.float()


        for i in range(Tq):
            q_start = i * Bq
            q_end = min((i+1) * Bq, Nq)
            Q_i = Qf[:, q_start:q_end, :]
            q_tile = q_end - q_start

            O_i = torch.zeros((B, q_tile, d), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((B, q_tile), device=Q.device, dtype=torch.float32)
            m_i = torch.full((B, q_tile), float("-inf"), device=Q.device, dtype=torch.float32)

            for j in range(Tk):
                k_start = j * Bk
                k_end = min((j+1) * Bk, Nk)
                K_j = Kf[:, k_start:k_end, :]
                V_j = Vf[:, k_start:k_end, :]

                S_ij = torch.einsum("bqd,bkd->bqk", Q_i, K_j) * scale
                m_new = torch.maximum(m_i, torch.max(S_ij, dim=-1).values)
                P_ij = torch.exp(S_ij - m_new[:, :, None])
                exp_scale = torch.exp(m_i - m_new)
                l_i = exp_scale * l_i + torch.sum(P_ij, dim=-1)
                O_i = (
                    exp_scale[:, :, None] * O_i
                    + torch.einsum("bqk,bkd->bqd", P_ij, V_j)
                )
                m_i = m_new
            
            O_i = O_i / l_i[:, :, None]
            L_i = m_i + torch.log(l_i)

            O[:, q_start:q_end, :] = O_i
            L[:, q_start:q_end] = L_i
        O = O.to(Q.dtype)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O 

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_attention1_backward(Q, K, V, O, grad_output, L)
        return dQ, dK, dV, None
    

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    query_start = query_tile_index * Q_TILE_SIZE
    q_idx = query_start + tl.arange(0, Q_TILE_SIZE)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)


    for key_start in tl.range(0, N_KEYS, K_TILE_SIZE):
        
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  

        if is_causal:
            k_idx = key_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S_ij = tl.where(causal_mask, S_ij, S_ij + (-1e6))

        m_new = tl.maximum(m_i, tl.max(S_ij, axis=1))

        P_tilde = tl.exp(S_ij - m_new[:, None])

        alpha = tl.exp(m_i - m_new)
        l_i = alpha * l_i + tl.sum(P_tilde, axis=1)

        O_i = O_i * alpha[:, None]
        O_i = tl.dot(P_tilde.to(V_j.dtype), V_j, acc=O_i)

        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    tl.store(
        O_block_ptr,
        O_i.to(O_block_ptr.type.element_ty),
        boundary_check=(0, 1),
    )

    l_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    l_ptrs = L_ptr + batch_index * stride_lb + l_offsets * stride_lq
    tl.store(l_ptrs, L_i, mask=l_offsets < N_QUERIES)

    
class FlashAttentionForwardTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q: (B, N_q, D)
        K: (B, N_k, D)
        V: (B, N_k, D)

        Returns:
            O: (B, N_q, D)
        """
        # assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.ndim == 3 and K.ndim == 3 and V.ndim == 3

        B, N_q, D = Q.shape
        Bk, N_k, Dk = K.shape
        Bv, N_v, Dv = V.shape

        assert B == Bk == Bv, "Batch size mismatch"
        assert D == Dk == Dv, "Expected Q, K, V to all have last dim D"
        assert N_k == N_v, "K and V sequence lengths must match"

        # tile sizes (at least 16 x 16)
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        O = torch.empty_like(Q)
        L = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(N_q, Q_TILE_SIZE), B)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            1.0 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal 

        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        Qf = Q.float()
        Kf = K.float()
        Vf = V.float()
        dOf = dO.float()

        D = Q.shape[-1]
        scale = 1.0 / math.sqrt(D)
        S = torch.matmul(Qf, Kf.transpose(-2, -1)) * scale

        if is_causal:
            Nq, Nk = S.shape[-2], S.shape[-1]
            q_idx = torch.arange(Nq, device=S.device)[:, None]
            k_idx = torch.arange(Nk, device=S.device)[None, :]
            causal_mask = q_idx >= k_idx 
            S = S.masked_fill(~causal_mask, -1e6)

        P = torch.softmax(S, dim=-1) 


        dV = torch.matmul(P.transpose(-2, -1), dOf)


        dP = torch.matmul(dOf, Vf.transpose(-2, -1)) 

        D_i = torch.sum(dP * P, dim=-1, keepdim=True) 
        dS = P * (dP - D_i) 

        if is_causal:
            dS = dS.masked_fill(~causal_mask, 0.0)

        dQ = torch.matmul(dS, Kf) * scale 

        dK = torch.matmul(dS.transpose(-2, -1), Qf) * scale

        dQ = dQ.to(Q.dtype)
        dK = dK.to(K.dtype)
        dV = dV.to(V.dtype)

        return dQ, dK, dV, None



