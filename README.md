# Optimizing Standard Attention Computation with Triton-Based FlashAttention

## Main Goal

Our work mainly focuses on optimizing standard attention computation using Triton-based FlashAttention.

## How to Replicate the Results

If you already have the Triton packages installed, you can directly run:

```bash
python benchmarking_triton.py
```

If not, thanks to Professor Greg Durrett for building the uv environment for running the code. You can refer to this GitHub repository for the code source and environment setup: https://github.com/gregdurrett/nyu-llm-reasoners-a2

In this case, you can replicate our results by simply running:

```bash
uv run python benchmarking_triton.py
```

## Further Instructions
The execution command above would run the whole evaluation pipeline to produce the completed result as shown in the report. However, this execution may take up to 7 hours running on A100 GPU. 

If you want to have a quicker execution just to verify if our code can successfully produce the result, you can simply change 1 line of code by following the instruction below: 
    In the `benchmarking_triton.py` file, line 170 currently tests sequence lengths from 128 to 16384: `[2 ** i for i in range(7, 15)]` The results will only be shown after all sequence-length tests are completed. Therefore, to reduce the time needed to run the experiments, you can adjust it to a smaller range, for example: `[2 ** i for i in range(7, 12)]`. This will test sequence lengths from 128 to 2048 and make the experiment finish faster.
