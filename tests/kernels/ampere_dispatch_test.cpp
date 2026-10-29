#include "basalt/core/device.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/prefill/moe_fp4.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

int main() {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
        const auto dev = basalt::core::device_info();
        if (dev.cc_major != 8 || dev.cc_minor != 6) return 77;
        using namespace basalt::kernels;
        if (dense_tc_supported(8, 2560) || dense_tc_pair_supported(8, 8, 2560) ||
            hc_fused_supported() || dec_gdnf_mode() != 0 || basalt::prefill::fp4::supported()) {
            std::puts("FAIL: an unavailable specialized kernel was selected");
            return 1;
        }
        cudaStream_t stream;
        check(cudaStreamCreate(&stream));
        for (int nin : {512, 2560, 6144}) {
            constexpr int nout = 32;
            std::vector<unsigned char> w(native_mmvq_weight_bytes(8, nin, nout));
            for (int r = 0; r < nout; ++r)
                for (int b = 0; b < nin / 32; ++b) {
                    auto* block = w.data() + (r * (nin / 32) + b) * 34;
                    const unsigned short scale = 0x3c00;
                    std::memcpy(block, &scale, 2);
                    std::memset(block + 2, r % 5 + 1, 32);
                }
            void *dw, *dxq;
            float *dx, *dy;
            check(cudaMalloc(&dw, w.size()));
            check(cudaMalloc(&dxq, native_q8_1_bytes(nin, 8)));
            check(cudaMalloc(&dx, nin * 8 * sizeof(float)));
            check(cudaMalloc(&dy, nout * 8 * sizeof(float)));
            check(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
            for (int cols : {1, 2, 8}) {
                std::vector<float> x(nin * cols), y(nout * cols);
                for (int c = 0; c < cols; ++c)
                    for (int k = 0; k < nin; ++k) x[c * nin + k] = 127.f * (c + 1);
                check(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemset(dy, 0xff, y.size() * sizeof(float)));
                native_quantize_q8_1(dx, dxq, nin, cols, stream);
                native_mmvq(8, dw, dxq, dy, nin, nout, cols, stream);
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (int c = 0; c < cols; ++c)
                    for (int r = 0; r < nout; ++r) {
                        const double ref = (double) nin * 127 * (c + 1) * (r % 5 + 1);
                        if (!std::isfinite(y[c * nout + r]) || y[c * nout + r] != ref) {
                            std::printf("FAIL nin=%d cols=%d c=%d r=%d got=%g ref=%g\n",
                                        nin, cols, c, r, y[c * nout + r], ref);
                            return 1;
                        }
                    }
            }
            cudaFree(dw); cudaFree(dxq); cudaFree(dx); cudaFree(dy);
        }
        struct MmaCase { int type, rows, cols; };
        for (const auto c : {MmaCase{21, 6144, 5}, MmaCase{23, 10240, 7}, MmaCase{23, 12288, 8}}) {
            constexpr int nin = 2560;
            const size_t bytes = (size_t) c.rows * (nin / 256) * (c.type == 21 ? 110 : 136);
            void *dw, *dxq;
            float *dx, *dy;
            check(cudaMalloc(&dw, bytes));
            check(cudaMemset(dw, 0, bytes));
            check(cudaMalloc(&dxq, native_q8_1_bytes(nin, c.cols)));
            check(cudaMalloc(&dx, nin * c.cols * sizeof(float)));
            check(cudaMalloc(&dy, c.rows * c.cols * sizeof(float)));
            const std::vector<float> x(nin * c.cols, 127.f);
            check(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
            native_quantize_q8_1(dx, dxq, nin, c.cols, stream);
            for (int cols : {1, 4, c.cols}) {
                std::vector<float> y(c.rows * cols);
                check(cudaMemset(dy, 0xff, y.size() * sizeof(float)));
                const long before = c.type == 21 ? iq_mmvq_mma_launches() : native_mmvq_mma_launches();
                if (c.type == 21) iq_mmvq(c.type, dw, dxq, dy, nin, c.rows, cols, stream);
                else native_mmvq(c.type, dw, dxq, dy, nin, c.rows, cols, stream);
                check(cudaStreamSynchronize(stream));
                const long after = c.type == 21 ? iq_mmvq_mma_launches() : native_mmvq_mma_launches();
                if (after - before != (cols == c.cols ? 1 : 0)) {
                    std::printf("FAIL: default MMA dispatch type=%d cols=%d\n", c.type, cols);
                    return 1;
                }
                check(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (float v : y) if (v != 0.f) {
                    std::puts("FAIL: zero-weight projection differs from CPU reference");
                    return 1;
                }
            }
            cudaFree(dw); cudaFree(dxq); cudaFree(dx); cudaFree(dy);
        }
        cudaStreamDestroy(stream);
        std::puts("ampere_dispatch_test: fallback outputs match CPU double reference at all widths");
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
