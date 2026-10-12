// src/kernels/cuda/fused_layer_t4.cu - the window-width-4 instance of the fused layer kernel; see fused_layer.cu.
#define FZ_ONLY_T 4
#define FZ_P fz4_
#define FZ_NSC mkcf4
#define FZ_NSB mkbf4
#include "fused_layer.cu"
