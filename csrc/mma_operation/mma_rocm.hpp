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

__device__ void mma_operations(const void* __restrict__ ptr_in,
                               const float random_number,
                               const uint32_t mma_count)
{
    bf16x4_t v_a[8], v_b[8];
    fp32x4_t v_c[8];

    for(int i = 0; i < mma_count; i++)
    {
#pragma unroll
        for(int j = 0; j < 32; j++)
        {
            __asm__ __volatile__("v_mfma_f32_16x16x16_bf16, %0, %1, %2, %3\n"
                                 : "+v"(v_c)
                                 : "v"(v_a), "v"(v_b), "v"(v_c));
        }
    }
}

void mma_launcher(const void* __restrict__ ptr_in,
                  const float random_number;
                  const uint32_t mma_count,
                  const uint32_t gdx,
                  const uint32_t bdx)
{
    mma_operation<<<gdx, bdx>>>(ptr_in, random_number, mma_count);
}

