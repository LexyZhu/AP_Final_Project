import math
import itertools
import pandas as pd
import torch
import triton
import triton.testing

from flash_forward import FlashAttentionForwardTriton, FlashAttention_pytorch


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    return torch.device("cuda")


def pytorch_attention(q, k, v, is_causal=True):
    """
    Regular PyTorch attention (not FlashAttention).
    q, k, v: (B, N, D)
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    if is_causal:
        nq, nk = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones((nq, nk), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out

def flash_attention_pytorch(q, k, v, is_causal=True):
    """
    PyTorch implementation of FlashAttention, for benchmarking against our Triton version.
    You can use this to verify correctness of your Triton implementation, and also to compare performance.
    """
    return FlashAttention_pytorch.apply(q, k, v, is_causal)


def triton_attention(q, k, v, is_causal=True):
    """
    Your Triton FlashAttention implementation.
    """
    return FlashAttentionForwardTriton.apply(q, k, v, is_causal)


def make_inputs(seq_len, d_model, dtype, device):
    """
    Always use batch size 1, as required.
    """
    q = torch.randn(
        1, seq_len, d_model,
        device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        1, seq_len, d_model,
        device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        1, seq_len, d_model,
        device=device, dtype=dtype, requires_grad=True
    )
    return q, k, v


def clone_inputs(q, k, v):
    return (
        q.detach().clone().requires_grad_(True),
        k.detach().clone().requires_grad_(True),
        v.detach().clone().requires_grad_(True),
    )


def clear_grads(*tensors):
    for t in tensors:
        t.grad = None


def bench_forward(fn, q, k, v, is_causal=True):
    def _run():
        fn(q, k, v, is_causal)

    return triton.testing.do_bench(_run, warmup=25, rep=100)


def bench_backward(fn, q, k, v, is_causal=True):
    """
    Backward-only latency:
    precompute forward once, benchmark only .backward(...)
    """
    out = fn(q, k, v, is_causal)
    grad_out = torch.randn_like(out)

    def _run():
        clear_grads(q, k, v)
        out.backward(grad_out, retain_graph=True)

    return triton.testing.do_bench(_run, warmup=25, rep=100)


def bench_end_to_end(fn, q, k, v, is_causal=True):
    """
    End-to-end = forward + backward
    """
    def _run():
        clear_grads(q, k, v)
        out = fn(q, k, v, is_causal)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

    return triton.testing.do_bench(_run, warmup=25, rep=100)


def benchmark_one(seq_len, d_model, dtype, device):
    base_q, base_k, base_v = make_inputs(seq_len, d_model, dtype, device)

    # PyTorch forward
    q, k, v = clone_inputs(base_q, base_k, base_v)
    flpt_fwd = bench_forward(flash_attention_pytorch, q, k, v, is_causal=True)
    pt_fwd = bench_forward(pytorch_attention, q, k, v, is_causal=True)

    # PyTorch backward
    q, k, v = clone_inputs(base_q, base_k, base_v)
    flpt_bwd = bench_backward(flash_attention_pytorch, q, k, v, is_causal=True)
    pt_bwd = bench_backward(pytorch_attention, q, k, v, is_causal=True)

    # PyTorch end-to-end
    q, k, v = clone_inputs(base_q, base_k, base_v)
    flpt_e2e = bench_end_to_end(flash_attention_pytorch, q, k, v, is_causal=True)
    pt_e2e = bench_end_to_end(pytorch_attention, q, k, v, is_causal=True)

    # Triton forward
    q, k, v = clone_inputs(base_q, base_k, base_v)
    tr_fwd = bench_forward(triton_attention, q, k, v, is_causal=True)

    # Triton backward
    q, k, v = clone_inputs(base_q, base_k, base_v)
    tr_bwd = bench_backward(triton_attention, q, k, v, is_causal=True)

    # Triton end-to-end
    q, k, v = clone_inputs(base_q, base_k, base_v)
    tr_e2e = bench_end_to_end(triton_attention, q, k, v, is_causal=True)

    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype": str(dtype).replace("torch.", ""),
        "PyTorch Forward (ms)": pt_fwd,
        "PyTorch Backward (ms)": pt_bwd,
        "PyTorch End-to-End (ms)": pt_e2e,
        "FlashAttention PyTorch Forward (ms)": flpt_fwd,
        "FlashAttention PyTorch Backward (ms)": flpt_bwd,
        "FlashAttention PyTorch End-to-End (ms)": flpt_e2e,
        "Triton Forward (ms)": tr_fwd,
        "Triton Backward (ms)": tr_bwd,
        "Triton End-to-End (ms)": tr_e2e,
    }


def main():
    device = get_device()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Running on GPU: {gpu_name}")

    seq_lens = [2 ** i for i in range(7, 15)]   # 128 ... 16384
    d_models = [2 ** i for i in range(4, 8)]    # 16, 32, 64, 128
    dtypes = [torch.bfloat16, torch.float32]

    rows = []

    for seq_len, d_model, dtype in itertools.product(seq_lens, d_models, dtypes):
        print(f"Benchmarking seq_len={seq_len}, d_model={d_model}, dtype={dtype}")

        try:
            row = benchmark_one(seq_len, d_model, dtype, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {
                "seq_len": seq_len,
                "d_model": d_model,
                "dtype": str(dtype).replace("torch.", ""),
                "PyTorch Forward (ms)": "OOM",
                "PyTorch Backward (ms)": "OOM",
                "PyTorch End-to-End (ms)": "OOM",
                "Triton Forward (ms)": "OOM",
                "Triton Backward (ms)": "OOM",
                "Triton End-to-End (ms)": "OOM",
            }
        except Exception as e:
            row = {
                "seq_len": seq_len,
                "d_model": d_model,
                "dtype": str(dtype).replace("torch.", ""),
                "PyTorch Forward (ms)": f"ERR:{type(e).__name__}",
                "PyTorch Backward (ms)": f"ERR:{type(e).__name__}",
                "PyTorch End-to-End (ms)": f"ERR:{type(e).__name__}",
                "Triton Forward (ms)": f"ERR:{type(e).__name__}",
                "Triton Backward (ms)": f"ERR:{type(e).__name__}",
                "Triton End-to-End (ms)": f"ERR:{type(e).__name__}",
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    print(df)
    print(df.to_latex(index=False, float_format="%.3f"))
    # df.to_csv("flash_benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()
