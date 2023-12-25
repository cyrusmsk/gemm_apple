import ggml
from ggml.utils import from_numpy
import ctypes
import time
import numpy as np

# Allocate a new context with 16 MB of memory
params = ggml.ggml_init_params(mem_size=16 * 2048 * 2048, mem_buffer=None)
ctx = ggml.ggml_init(params=params)

N = 2048
Ann = np.random.randn(N, N).astype(np.float32)
# N^2
Bnn = np.random.randn(N, N).astype(np.float32)

# 2N compute in N^2 output cells
flop = 2*N*N*N

# Instantiate tensors
#a = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 1)
#b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 1)

a = from_numpy(Ann, ctx)
b = from_numpy(Bnn, ctx)

# Use ggml operations to build a computational graph
f = ggml.ggml_mul_mat(ctx, a, b)
f = ggml.ggml_set_name(f, b"f")

gf = ggml.ggml_new_graph(ctx)
ggml.ggml_build_forward_expand(gf, f)

# Compute the graph
for i in range(10):
    st = time.perf_counter()
    ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)
    #output = ggml.ggml_get_tensor(ctx, b"f")
    et = time.perf_counter()
    s = et-st
    print(f"{flop/s * 1e-9:.2f} GFLOP/S, {s*1e3:.2f} ms")

# Free the context
ggml.ggml_free(ctx)
