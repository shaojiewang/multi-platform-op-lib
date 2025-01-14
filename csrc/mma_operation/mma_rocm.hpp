#pragma once

#include "bf16_wrapper.hpp"

#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
#define BUFFER_LOAD_DWORD3 0x00020000   // This is valid for 
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};
__device__ int32x4_t make_buffer_resource(const void * ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res {ptr, size, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(int32x4_t, res);
}

__global__ void 
__launch_bounds__(512, 1)
mma_operations(const void* __restrict__ ptr_in,
               const float random_number,
               const uint32_t mma_count)
{
    int offset_in = threadIdx.x * sizeof(bf16x4_t) + blockIdx.x * blockDim.x * sizeof(bf16x4_t) * mma_count;

    bf16x4_t v_a[32], v_b[32];
    fp32x4_t v_c[1] = {0.f};

#pragma unroll
    for(int i = 0; i < 32; i++)
    {
        v_a[i] = {__builtin_bit_cast(bf16x1_t, type_convert<bhalf_t, float>(random_number - i))};
        v_b[i] = {__builtin_bit_cast(bf16x1_t, type_convert<bhalf_t, float>(random_number - i + 0.1))};
    }

    auto r_ptr_in = make_buffer_resource(ptr_in);

    for(int i = 0; i < mma_count; i++)
    {
        offset_in += blockDim.x * sizeof(bf16x4_t); 
        asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
                     : "+v"(v_a[0])
                     : "v"(offset_in), "s"(r_ptr_in), "n"(0)
                     : "memory");
#pragma unroll
        for(int j = 0; j < 32; j++)
        {
            __asm__ __volatile__("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %3\n"
                                 : "+v"(v_c[0])
                                 : "v"(v_a[j]), "v"(v_b[j]), "v"(v_c[0]));
        }
    }
}

void mma_launcher(const void* __restrict__ ptr_in,
                  const float random_number,
                  const uint32_t mma_count,
                  const uint32_t gdx,
                  const uint32_t bdx)
{
    mma_operations<<<gdx, bdx>>>(ptr_in, random_number, mma_count);
}

