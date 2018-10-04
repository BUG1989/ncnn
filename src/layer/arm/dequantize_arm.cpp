// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "dequantize_arm.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Dequantize_arm)

int Dequantize_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        int* intptr = bottom_top_blob;
        float* ptr = bottom_top_blob;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                ptr[i] = intptr[i] * scale + bias_data[i];
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                ptr[i] = intptr[i] * scale;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const int* intptr = bottom_top_blob.row<const int>(i);
                float* ptr = bottom_top_blob.row(i);

                float bias = bias_data_size > 1 ? bias_data[i] : bias_data[0];

                for (int j=0; j<w; j++)
                {
                    ptr[j] = intptr[j] * scale + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const int* intptr = bottom_top_blob.row<const int>(i);
                float* ptr = bottom_top_blob.row(i);

                for (int j=0; j<w; j++)
                {
                    ptr[j] = intptr[j] * scale;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                int* intptr = bottom_top_blob.channel(q);
                float* ptr = bottom_top_blob.channel(q);

                float bias = bias_data[q];

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                float32x4_t _bias = vdupq_n_f32(bias);
                float32x4_t _scale = vdupq_n_f32(scale);

                for (; nn>0; nn--)
                {
                    // load top_s32
                    int32x4_t _out0_s32 = vld1q_s32(intptr);
                    int32x4_t _out0n_s32 = vld1q_s32(intptr+4);

                    // top_s32 -> top_f32
                    float32x4_t _out0_f32 = vcvtq_f32_s32(_out0_s32);
                    float32x4_t _out0n_f32 = vcvtq_f32_s32(_out0n_s32);

                    // top_f32 = top_f32 * scale_out
                    _out0_f32 = vmulq_f32(_out0_f32, _scale);
                    _out0n_f32 = vmulq_f32(_out0n_f32, _scale);

                    // top_f32 = top_f32 + bias_tm
                    _out0_f32 = vaddq_f32(_out0_f32, _bias);
                    _out0n_f32 = vaddq_f32(_out0n_f32, _bias);

                    // save top_f32
                    vst1q_f32(ptr, _out0_f32);
                    vst1q_f32(ptr+4, _out0n_f32);

                    intptr += 8;
                    ptr += 8;
                }             
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1]!      \n" //q0-q1 data
                    "vdup.f32   q10, %6             \n" //q10 scale
                    "vdup.f32   q12, %7             \n" //q12 bias

                    "0:                             \n"
                    "vcvt.f32.s32 q0, q0           \n"
                    "vcvt.f32.s32 q1, q1           \n"

                    "vmul.f32   q0,q0,q10           \n"
                    "vmul.f32   q1,q1,q10           \n"

                    "vadd.f32   q2,q0,q12           \n"
                    "vadd.f32   q3,q1,q12           \n"

                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1]!      \n"
                    "vst1.f32   {d4-d7}, [%2]!      \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %1, #32             \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(scale),       // %6
                      "r"(bias)         // %7
                    : "cc", "memory", "q0", "q1", "q2", "q4", "q10", "q12"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *ptr = *intptr * scale + bias;

                    intptr++;
                    ptr++;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                int* intptr = bottom_top_blob.channel(q);
                float* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                float32x4_t _scale = vdupq_n_f32(scale);

                for (; nn>0; nn--)
                {
                    // load top_s32
                    int32x4_t _out0_s32 = vld1q_s32(intptr);
                    int32x4_t _out0n_s32 = vld1q_s32(intptr+4);

                    // top_s32 -> top_f32
                    float32x4_t _out0_f32 = vcvtq_f32_s32(_out0_s32);
                    float32x4_t _out0n_f32 = vcvtq_f32_s32(_out0n_s32);

                    // top_f32 = top_f32 * scale_out
                    _out0_f32 = vmulq_f32(_out0_f32, _scale);
                    _out0n_f32 = vmulq_f32(_out0n_f32, _scale);

                    // save top_f32
                    vst1q_f32(ptr, _out0_f32);
                    vst1q_f32(ptr+4, _out0n_f32);

                    intptr += 8;
                    ptr += 8;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1]!      \n" //q0-q1 data
                    "vdup.f32   q10, %6             \n" //q10 scale

                    "0:                             \n"
                    "vcvt.f32.s32 q0, q0           \n"
                    "vcvt.f32.s32 q1, q1           \n"

                    "vmul.f32   q2,q0,q10           \n"
                    "vmul.f32   q3,q1,q10           \n"

                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1]!      \n"
                    "vst1.f32   {d4-d7}, [%2]!      \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %1, #32             \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(scale)        // %6
                    : "cc", "memory", "q0", "q1", "q2", "q4", "q10", "q12"
                );              
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *ptr = *intptr * scale;

                    intptr++;
                    ptr++;
                }
            }
        }   
    }

    return 0;
}

} // namespace ncnn
