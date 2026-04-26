# Optimizing Standard Attention Computation with Triton-Based FlashAttention

## Main Goal
Our works is mainly about optimizing the standard attention through the Triton-based FlashAttention.

## How to replicate the results?
If you already have the triton packages downloaded, you can directly run 

`python benchmarking_triton.py`

If not, thanks to Professor Greg Durrett for bulding the uv environment for code running, you can refer to the github: https://github.com/gregdurrett/nyu-llm-reasoners-a2 for 
the code source. In this case, you can replicate our results by simply run:

`uv run benchmarking_triton.py` 

