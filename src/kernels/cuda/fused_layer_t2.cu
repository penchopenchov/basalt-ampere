// src/kernels/cuda/fused_layer_t2.cu - the window-width-2 instance of the fused layer kernel; see fused_layer.cu.
#define FZ_ONLY_T 2
#define FZ_P fz2_
#define FZ_NSC mkcf2
#define FZ_NSB mkbf2
#include "fused_layer.cu"
