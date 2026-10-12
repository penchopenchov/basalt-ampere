// src/kernels/cuda/fused_layer_t3.cu - the window-width-3 instance of the fused layer kernel; see fused_layer.cu.
#define FZ_ONLY_T 3
#define FZ_P fz3_
#define FZ_NSC mkcf3
#define FZ_NSB mkbf3
#include "fused_layer.cu"
