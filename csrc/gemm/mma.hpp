#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

template<class ACC_TYPE>
__device__ __forceinline__ void mma_sync(unsigned int* fragA,
                                         unsigned int* fragB,
                                         ACC_TYPE* accum)
{
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16 "
               "{} ");
}

