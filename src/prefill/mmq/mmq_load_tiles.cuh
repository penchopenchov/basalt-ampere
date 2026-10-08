// SPDX-License-Identifier: MIT
// Loads of the quantized weight tiles into shared memory, per format.
// llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d (https://github.com/ggml-org/llama.cpp),
// ggml/src/ggml-cuda/mmq-load-tiles.cuh.  Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#pragma once

#include "vecdotq.cuh"

#include "mmq.cuh"

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int blocks_per_iter = MMQ_ITER_K / QK2_0;
    constexpr int threads_per_row = blocks_per_iter * QI2_0;
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int scale_entries_per_block = QK2_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_block;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI2_0;
    const int kqsx = txi % QI2_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i*stride + kbx;
        const int16_t    * qxi = (const int16_t *) bxi->qs + kqsx * 4;

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q  = qxi[j];

            // unpack even and odd crumbs into byte values
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            // unshuffle values
            const int qx = __byte_perm(qe, qo, 0x5140);
            const int qy = __byte_perm(qe, qo, 0x7362);

            x_qs[i*sram_stride           + dst_offset + j*2+0] = qx;
            x_qs[i*sram_stride           + dst_offset + j*2+1] = qy;
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i*stride + scale_block;

        x_df[i*sram_stride                           + ksx] = bxi->d;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q8_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2*MMQ_TILE_NE_K);

    // MMQ_ITER_K / (4 * QR8_0) == 64 required. but NV has only 32 threads per warp
    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI8_0;
    const int kqsx = txi % QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;

        x_qs[i*sram_stride + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
        x_qs[i*sram_stride + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
    }

    constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / QI8_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;

        x_df[i*sram_stride                           + kbxd] = bxi->d;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xxs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_xxs * bxi = (const block_iq2_xxs *) x + kbx0 + i*stride;

        const int q2 = get_int_b2(bxi->qs, 2*kqsx+0);
        const uint8_t * aux8 = (const uint8_t *) &q2;
        const uint32_t aux32 = get_int_b2(bxi->qs, 2*kqsx+1);

#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[l]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid1;
        }

        const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
        const float d = bxi->d;
        x_df[i*sram_stride             + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_xs * bxi = (const block_iq2_xs *) x + kbx0 + i*stride;

        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;

    #pragma unroll
        for (int l = 0; l < QR2_XS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l] >> 9);

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
        x_df[i*sram_stride                             + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*sram_stride                             + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_s * bxi = (const block_iq2_s *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->qs, QK_K/32 + kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR2_S; ++l) {
            const int * grid_pos = (const int *)(iq2s_grid + (qs[l] | ((qh << (8-2*l)) & 0x300)));

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
        x_df[i*sram_stride                             + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*sram_stride                             + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_xxs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_xxs * bxi = (const block_iq3_xxs *) x + kbx0 + i*stride;

        const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * q3 = (const uint8_t *) &q3_packed;
        const uint32_t aux32 = get_int_b2(bxi->qs, QK_K/16 + kqsx);

#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[2*l+0]], iq3xxs_grid[q3[2*l+1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7*l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
        }

        const int ls = aux32 >> 28;
        const float d = bxi->d;
        x_df[i*sram_stride             + kqsx] = (ls*d + d/2)/2;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_s * bxi = (const block_iq3_s *) x + kbx0 + i*stride;

        const int2      qs_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->signs, kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR3_S; ++l) {
            const int2 grid_pos = make_int2(
                iq3s_grid[qs[2*l+0] | ((qh << (8 - 2*l)) & 0x100)],
                iq3s_grid[qs[2*l+1] | ((qh << (7 - 2*l)) & 0x100)]);

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = grid_h;
        }

        const int ls = 1 + 2*((bxi->scales[kqsx/2] >> (((2*kqsx) << 1) & 0x04)) & 0x0F);
        const float d = bxi->d;
        x_df[i*sram_stride             + kqsx] = ls*d;
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_XS);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const int aux_q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = 8 * (kqsx / 4) + kqsx % 4;

        x_qs[i*sram_stride + k0 + 0] = v.x;
        x_qs[i*sram_stride + k0 + 4] = v.y;
    }

    constexpr int rows_per_warp = warp_size / 8;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / (MMQ_TILE_NE_K/4);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const float d = __half2float(bxi->d);

        const int ls = ((bxi->scales_l[(threadIdx.x % 8)/2] >> (4*(threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2*(threadIdx.x % 8))) & 0x03) << 4);

        x_df[i*sram_stride             + threadIdx.x % 8] = d * (ls - 32);
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_nl(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_NL);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_NL;
    const int kqsx = txi % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbx;

        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = kbx * (2 * QI4_NL) + kqsx;

        x_qs[i*sram_stride + k0 + 0]      = v.x;
        x_qs[i*sram_stride + k0 + QI4_NL] = v.y;
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_NL;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbxd;

        x_df[i*sram_stride                       + kbxd] = __half2float(bxi->d);
    }
}
