// src/vision/encoder_cuda.cu - see basalt/vision/encoder_cuda.hpp.
#include "basalt/vision/encoder_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "vision_kernels.cuh"

namespace basalt::vision {

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) { fail(std::string(#x) + ": " + cudaGetErrorString(e_)); return false; } \
    } while (0)
#define CB(x)                                                                                    \
    do {                                                                                         \
        cublasStatus_t s_ = (x);                                                                 \
        if (s_ != CUBLAS_STATUS_SUCCESS) { fail(std::string(#x) + " failed: " + std::to_string((int) s_)); return false; } \
    } while (0)

namespace {
constexpr int E = 1152, F = 4304, H = 16, D = 72, DP = 80;
}

struct CudaEncoder::Impl {
    struct Layer {
        float *ln1w, *ln1b, *ln2w, *ln2b, *bqkv, *bo, *bup, *bdown;
        __nv_bfloat16 *wqkv, *wo, *wup, *wdown;
    };
    const Mmproj* mp = nullptr;
    int layers = 0, proj = 0, nmax = 0;
    float eps = 1e-6f;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    std::vector<void*> allocs;
    size_t bytes = 0;
    std::string err;
    std::vector<Layer> L;
    float *wpatch = nullptr, *bpatch = nullptr, *pos = nullptr, *postw = nullptr, *postb = nullptr, *bm0 = nullptr, *bm2 = nullptr;
    __nv_bfloat16 *wm0 = nullptr, *wm2 = nullptr;
    // work buffers (nmax patches)
    float *d_patches = nullptr, *x = nullptr, *g = nullptr, *rope = nullptr, *d_out = nullptr;
    __nv_bfloat16* act = nullptr;
    __half *qh = nullptr, *kh = nullptr, *vh = nullptr;

    void fail(const std::string& s) { err = s; }
    template <class T>
    bool alloc(T*& p, size_t n) {
        void* q = nullptr;
        if (cudaMalloc(&q, n * sizeof(T)) != cudaSuccess) { fail("out of device memory"); return false; }
        allocs.push_back(q);
        bytes += n * sizeof(T);
        p = (T*) q;
        return true;
    }
    template <class T>
    bool upload(T*& p, const void* src, size_t n) {
        if (!alloc(p, n)) return false;
        CK(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice));
        return true;
    }
    ~Impl() {
        for (void* p : allocs) cudaFree(p);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }

    // Y[M][N] = X[M][K] . W[N][K]^T
    bool gemm_bf16(const __nv_bfloat16* X, const __nv_bfloat16* W, float* Y, int M, int N, int K) {
        const float one = 1.f, zero = 0.f;
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, Y,
                        CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        return true;
    }
    bool gemm_f32(const float* X, const float* W, float* Y, int M, int N, int K) {
        const float one = 1.f, zero = 0.f;
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_32F, K, X, CUDA_R_32F, K, &zero, Y,
                        CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        return true;
    }

    bool init(const Mmproj& m, int max_tokens, std::string& e) {
        mp = &m;
        const Hparams& h = m.hp();
        if (h.embd != E || h.heads != H || h.ffn != F || h.patch != 16 || h.merge != 2 || h.layers < 1) {
            e = "unexpected projector hyper-parameters";
            return false;
        }
        layers = h.layers;
        proj = h.proj;
        eps = h.eps;
        nmax = max_tokens * 4;
        CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CB(cublasCreate(&blas));
        CB(cublasSetStream(blas, stream));
        std::string er;
        auto f32 = [&](const std::string& n, int64_t c) { const auto* t = m.need(n, 0, c, er); return t ? (const float*) t->data : nullptr; };
        auto b16 = [&](const std::string& n, int64_t c) { const auto* t = m.need(n, 30, c, er); return t ? t->data : nullptr; };
        auto upf = [&](float*& d, const std::string& n, int64_t c) {
            const float* s = f32(n, c);
            return s && upload(d, s, (size_t) c);
        };
        auto upb = [&](__nv_bfloat16*& d, const std::string& n, int64_t c) {
            const void* s = b16(n, c);
            return s && upload(d, s, (size_t) c);
        };
        bool ok = true;
        {
            const float *w0 = f32("v.patch_embd.weight", (int64_t) E * 768), *w1 = f32("v.patch_embd.weight.1", (int64_t) E * 768);
            if (w0 && w1) {
                std::vector<float> w((size_t) E * 768);
                for (size_t i = 0; i < w.size(); ++i) w[i] = w0[i] + w1[i];
                ok = upload(wpatch, w.data(), w.size());
            } else ok = false;
        }
        ok = ok && upf(bpatch, "v.patch_embd.bias", E) && upf(pos, "v.position_embd.weight", (int64_t) E * 2304) &&
             upf(postw, "v.post_ln.weight", E) && upf(postb, "v.post_ln.bias", E) && upf(bm0, "mm.0.bias", 4 * E) &&
             upf(bm2, "mm.2.bias", proj) && upb(wm0, "mm.0.weight", (int64_t) 4 * E * 4 * E) &&
             upb(wm2, "mm.2.weight", (int64_t) 4 * E * proj);
        L.resize(layers);
        for (int l = 0; ok && l < layers; ++l) {
            const std::string p = "v.blk." + std::to_string(l) + ".";
            Layer& y = L[l];
            ok = upf(y.ln1w, p + "ln1.weight", E) && upf(y.ln1b, p + "ln1.bias", E) && upf(y.ln2w, p + "ln2.weight", E) &&
                 upf(y.ln2b, p + "ln2.bias", E) && upf(y.bqkv, p + "attn_qkv.bias", 3 * E) && upf(y.bo, p + "attn_out.bias", E) &&
                 upf(y.bup, p + "ffn_up.bias", F) && upf(y.bdown, p + "ffn_down.bias", E) &&
                 upb(y.wqkv, p + "attn_qkv.weight", (int64_t) 3 * E * E) && upb(y.wo, p + "attn_out.weight", (int64_t) E * E) &&
                 upb(y.wup, p + "ffn_up.weight", (int64_t) F * E) && upb(y.wdown, p + "ffn_down.weight", (int64_t) E * F);
        }
        if (!ok) { e = er.empty() ? err : er; return false; }
        const size_t N = nmax;
        ok = alloc(d_patches, N * 768) && alloc(x, N * E) && alloc(g, N * F) && alloc(act, N * F) && alloc(rope, N * 36 * 2) &&
             alloc(qh, (size_t) H * N * DP) && alloc(kh, (size_t) H * N * DP) && alloc(vh, (size_t) H * N * DP) &&
             alloc(d_out, N / 4 * proj);
        if (!ok) { e = err; return false; }
        // the padding columns of q, k, v stay zero for good: the kernels write columns 0..71 only
        CK(cudaMemset(qh, 0, (size_t) H * N * DP * sizeof(__half)));
        CK(cudaMemset(kh, 0, (size_t) H * N * DP * sizeof(__half)));
        CK(cudaMemset(vh, 0, (size_t) H * N * DP * sizeof(__half)));
        CK(cudaStreamSynchronize(stream));
        return true;
    }

    bool run(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap) {
        const int gx = canvas.w / 16, gy = canvas.h / 16, n = gx * gy, m = n / 4;
        if (n % 4 != 0 || n > nmax || (int) patches.size() != n * 768) { fail("image outside the sized range"); return false; }
        auto host_tap = [&](const std::string& name, const void* dev, size_t count, bool bf) -> bool {
            if (!tap) return true;
            CK(cudaStreamSynchronize(stream));
            std::vector<float> h(count);
            if (!bf) CK(cudaMemcpy(h.data(), dev, count * 4, cudaMemcpyDeviceToHost));
            else {
                std::vector<__nv_bfloat16> b(count);
                CK(cudaMemcpy(b.data(), dev, count * 2, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < count; ++i) h[i] = __bfloat162float(b[i]);
            }
            tap(name, h.data(), count);
            return true;
        };
        CK(cudaMemcpyAsync(d_patches, patches.data(), patches.size() * 4, cudaMemcpyHostToDevice, stream));
        if (!gemm_f32(d_patches, wpatch, x, n, E, 768)) return false;
        launch_embed_finish(x, bpatch, pos, n, gx, gy, stream);
        launch_rope_table(rope, n, gx, stream);
        if (!host_tap("embed", x, (size_t) n * E, false)) return false;
        const float scale_log2 = (1.0f / sqrtf((float) D)) * 1.4426950408889634f;
        for (int l = 0; l < layers; ++l) {
            const Layer& y = L[l];
            launch_layernorm(x, y.ln1w, y.ln1b, act, n, eps, stream);
            if (!gemm_bf16(act, y.wqkv, g, n, 3 * E, E)) return false;
            launch_qkv_epilogue(g, y.bqkv, rope, qh, kh, vh, n, stream);
            launch_attention(qh, kh, vh, act, n, scale_log2, stream);
            if (!gemm_bf16(act, y.wo, g, n, E, E)) return false;
            launch_bias_residual(x, g, y.bo, n, E, stream);
            if (!host_tap("ffn_inp." + std::to_string(l), x, (size_t) n * E, false)) return false;
            launch_layernorm(x, y.ln2w, y.ln2b, act, n, eps, stream);
            if (!gemm_bf16(act, y.wup, g, n, F, E)) return false;
            launch_bias_gelu(g, y.bup, act, n, F, stream);
            if (!gemm_bf16(act, y.wdown, g, n, E, F)) return false;
            launch_bias_residual(x, g, y.bdown, n, E, stream);
            if (!host_tap("layer_out." + std::to_string(l), x, (size_t) n * E, false)) return false;
        }
        launch_layernorm(x, postw, postb, act, n, eps, stream);
        if (!host_tap("post_ln", act, (size_t) n * E, true)) return false;
        if (!gemm_bf16(act, wm0, g, m, 4 * E, 4 * E)) return false;
        launch_bias_gelu(g, bm0, act, m, 4 * E, stream);
        if (!gemm_bf16(act, wm2, g, m, proj, 4 * E)) return false;
        launch_bias_copy(g, bm2, d_out, m, proj, stream);
        out.resize((size_t) m * proj);
        CK(cudaMemcpyAsync(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        CK(cudaGetLastError());
        if (tap) tap("out", out.data(), out.size());
        return true;
    }
};

CudaEncoder::CudaEncoder() : impl_(new Impl) {}
CudaEncoder::~CudaEncoder() = default;

bool CudaEncoder::init(const Mmproj& mp, int max_tokens, std::string& err) {
    if (impl_->init(mp, max_tokens, err)) return true;
    if (err.empty()) err = impl_->err;
    return false;
}

bool CudaEncoder::encode(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap) {
    return impl_->run(patches, canvas, out, tap);
}

size_t CudaEncoder::device_bytes() const { return impl_->bytes; }
std::string CudaEncoder::last_error() const { return impl_->err; }

}  // namespace basalt::vision
