# Matrix Multiplication on Apple Silicon (M1)
This repository contains code and my tests on different frameworks and matmul implementations for M1 chips.

## Checked solutions

* NumPy
* TensorFlow ANE https://github.com/tlkh/tf-metal-experiments/blob/main/coreml_matmul.py
* TensorFlow Metal https://github.com/tlkh/tf-metal-experiments/blob/main/unified_mem_benchmark.py
* Tuned Metal, Torch and Tinygrad https://github.com/tinygrad/tinygrad/blob/master/extra/gemm/metal_matmul.py
* Apple's Framework https://github.com/ml-explore/mlx
* Python Bindings for GGML https://github.com/abetlen/ggml-python

## Useful links

### Benchmarks
* https://github.com/philipturner/metal-benchmarks
* https://github.com/LaurentMazare/gemm-metal

### New algos description
* AlphaTensor from DeepMind
https://deepmind.google/discover/blog/discovering-novel-algorithms-with-alphatensor/

* MIT research
https://arxiv.org/abs/2210.10173
https://epubs.siam.org/doi/10.1137/1.9781611977912.134
https://www.quantamagazine.org/new-breakthrough-brings-matrix-multiplication-closer-to-ideal-20240307/

### Optimizations
* Llamafile kernels
https://justine.lol/matmul/

* AMD GPU optimization
https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html

* cuBLAS optimization
https://siboehm.com/articles/22/CUDA-MMM

* Burn Framework
https://burn.dev/blog/sota-multiplatform-matmul/

### Other
* Some useful links
https://github.com/yuninxia/awesome-gemm
