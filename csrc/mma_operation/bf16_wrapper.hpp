#pragma once

#if defined(__CUDACC__)
#include <cuda_bf16.h>
#else

struct bfloat16_t 
{
    unsigned short x;
}

using __nv_bfloat16 = bfloat16_t;
using bhalf_t = bfloat16_t;

using bf16x1_t = bfloat16_t;
using bf16x2_t = bf16x1_t __attribute__((ext_vector_type(2)));
using bf16x4_t = bf16x1_t __attribute__((ext_vector_type(4)));
using bf16x8_t = bf16x1_t __attribute__((ext_vector_type(8)));

using fp32x1_t = float;
using fp32x2_t = fp32x1_t __attribute__((ext_vector_type(2)));
using fp32x4_t = fp32x1_t __attribute__((ext_vector_type(4)));
using fp32x8_t = fp32x1_t __attribute__((ext_vector_type(8)));

// Declare a template function for bf16 conversion using RTN
template <typename Y, typename X>
__host__ __device__ constexpr Y bf16_convert_rtn(X x);

// Convert fp32 to bf16 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, float>(float x)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    // When the exponent bits are not all 1s, then the value is zero, normal,
    // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
    // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
    // This causes the bfloat16's mantissa to be incremented by 1 if the 16
    // least significant bits of the float mantissa are greater than 0x8000,
    // or if they are equal to 0x8000 and the least significant bit of the
    // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
    // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
    // has the value 0x7f, then incrementing it causes it to become 0x00 and
    // the exponent is incremented by one, which is the next higher FP value
    // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
    // with an exponent of 0x00 and a mantissa of 0x7f, it may be rounded up
    // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
    // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
    // incrementing it causes it to become an exponent of 0xFF and a mantissa
    // of 0x00, which is Inf, the next higher value to the unrounded value.
    bool flag0 = ~u.int32 & 0x7f800000;

    // When all of the exponent bits are 1, the value is Inf or NaN.
    // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
    // mantissa bit. Quiet NaN is indicated by the most significant mantissa
    // bit being 1. Signaling NaN is indicated by the most significant
    // mantissa bit being 0 but some other bit(s) being 1. If any of the
    // lower 16 bits of the mantissa are 1, we set the least significant bit
    // of the bfloat16 mantissa, in order to preserve signaling NaN in case
    // the bfloat16's mantissa bits are all 0.
    bool flag1 = !flag0 && (u.int32 & 0xffff);

    u.int32 += flag0 ? 0x7fff + ((u.int32 >> 16) & 1) : 0; // Round to nearest, round to even
    u.int32 |= flag1 ? 0x10000 : 0x0;                      // Preserve signaling NaN

    return __builtin_bit_cast(bhalf_t, uint16_t(u.int32 >> 16));
}

// Convert X to Y
template <typename Y, typename X>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    return static_cast<Y>(x);
}

// convert bfp16 to fp32
template <>
inline __host__ __device__ constexpr float type_convert<float, bhalf_t>(bhalf_t x)
{
    union
    {
        uint32_t int32;
        float fp32;
    } u = {uint32_t(x.x) << 16};

    return u.fp32;
}

// convert fp32 to bf16
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, float>(float x)
{
    int val_i = __builtin_bit_cast(int, x);
    return __builtin_bit_cast(bhalf_t, uint16_t(val_i >> 16));
}

#endif
