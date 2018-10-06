// SenseNets is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
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

#include "requantize_arm.h"
#include "benchmark.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

// round to nearest
static signed char float2int8(float value)
{
    float tmp;
    if (value >= 0.f) tmp = value + 0.5;
    else tmp = value - 0.5;

    if (tmp > 127)
        return 127;
    if (tmp < -128)
        return -128;

    return tmp;
}

namespace ncnn {

DEFINE_LAYER_CREATOR(Requantize_arm)

int Requantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{ 

    // double start = ncnn::get_current_time();

#if !__aarch64__
    int FPSCR_value = 0;

    asm volatile(
        "vmrs   %0, FPSCR               \n"
        "bic    r10, %0, #0x00c00000    \n"
        "vmsr   FPSCR, r10              \n"
        : "=r"(FPSCR_value)
        :
        : "memory", "r10"
    );
#endif

    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        int w = bottom_blob.w;

        const int* intptr = bottom_blob;
        signed char * ptr = top_blob;

        if (bias_term)
        {
            if (bias_data_size > 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i=0; i<w; i++)
                {
					float bias_tm = bias_data[i] * scale_in;
                    ptr[i] = float2int8((intptr[i] + bias_tm) * scale_out);
                    if (fusion_relu && ptr[i] < 0)
                        ptr[i] = 0;
                }
            }
            else
            {
                float bias = bias_data[0];
				float bias_tm = bias * scale_in;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i=0; i<w; i++)
                {
                    ptr[i] = float2int8((intptr[i] + bias_tm) * scale_out);
                    if (fusion_relu && ptr[i] < 0)
                        ptr[i] = 0;                    
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                ptr[i] = float2int8(intptr[i] * scale_out);
                if (fusion_relu && ptr[i] < 0)
                    ptr[i] = 0;                
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                signed char* ptr = top_blob.row<signed char>(i);

                float bias = bias_data_size > 1 ? bias_data[i] : bias_data[0];
				float bias_tm = bias * scale_in;

                for (int j=0; j<w; j++)
                {
                    ptr[j] = float2int8((intptr[j] + bias_tm) * scale_out);
                    if (fusion_relu && ptr[j] < 0)
                        ptr[j] = 0;                    
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                signed char* ptr = top_blob.row<signed char>(i);

                for (int j=0; j<w; j++)
                {
                    ptr[j] = float2int8(intptr[j] * scale_out);
                    if (fusion_relu && ptr[j] < 0)
                        ptr[j] = 0;                    
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;      

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

                float bias = bias_data_size > 1 ? bias_data[q] : bias_data[0];
				float bias_tm = bias * scale_in;

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                float32x4_t _bias_tm = vdupq_n_f32(bias_tm);
                float32x4_t _scale_out = vdupq_n_f32(scale_out);

                if (nn > 0)
                {
                asm volatile(                                          
                    "dup    v2.4s, %w6                   \n" //q10 scale_out
                    "dup    v3.4s, %w7                   \n" //q12 bias_tm
                    "0:                                  \n"
                    "prfm   pldl1keep, [%1, #128]      \n"
                    "ld1    {v0.4s, v1.4s}, [%1], #32    \n" //q0-q1 data                      
                    // top_s32 -> top_f32
                    "scvtf  v5.4s, v0.4s                 \n"
                    "scvtf  v6.4s, v1.4s                 \n"
                    // top_f32 = top_f32 + bias_tm
                    "fadd   v5.4s, v5.4s, v3.4s          \n"
                    "fadd   v6.4s, v6.4s, v3.4s          \n"
                    // top_f32 = top_f32 * scale_out
                    "fmul   v5.4s, v5.4s, v2.4s          \n"
                    "fmul   v6.4s, v6.4s, v2.4s          \n"
                    // top_f32 -> top_s32
                    "fcvtas v7.4s, v5.4s                 \n"
                    "fcvtas v8.4s, v6.4s                 \n"
                    // top_s32 -> top_s16
                    "sqxtn  v9.4h, v7.4s                 \n"
                    "sqxtn2 v9.8h, v8.4s                 \n"
                    // top_s16 -> top_s8
                    "sqxtn  v10.8b, v9.8h                \n"
                    // save top_s8
                    "st1    {v10.8b}, [%2], #8           \n"
                    "subs   %w0, %w0, #1                 \n"
                    "bne    0b                           \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(scale_out),   // %6
                      "r"(bias_tm)      // %7
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1:128]!  \n" //q0-q1 data
                    "vdup.f32   q10, %6             \n" //q10 scale_out
                    "vdup.f32   q12, %7             \n" //q12 bias_tm
                    "0:                             \n"
                    // top_s32 -> top_f32
                    "vcvt.f32.s32 q0, q0            \n" 
                    "vcvt.f32.s32 q1, q1            \n"
                    // top_f32 = top_f32 + bias_tm
                    "vadd.f32   q0, q0, q12         \n"
                    "vadd.f32   q1, q1, q12         \n"
                    // top_f32 = top_f32 * scale_out
                    "vmul.f32   q0, q0, q10         \n"
                    "vmul.f32   q1, q1, q10         \n"
                    // top_f32 -> top_s32
                    "vcvtr.s32.f32 s0, s0           \n"
                    "vcvtr.s32.f32 s1, s1           \n"
                    "vcvtr.s32.f32 s2, s2           \n"
                    "vcvtr.s32.f32 s3, s3           \n"
                    "vcvtr.s32.f32 s4, s4           \n"
                    "vcvtr.s32.f32 s5, s5           \n"
                    "vcvtr.s32.f32 s6, s6           \n"
                    "vcvtr.s32.f32 s7, s7           \n" 
                    // top_s32 -> top_s16
                    "vqmovn.s32 d4, q0              \n"
                    "vqmovn.s32 d5, q1              \n"
                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1:128]!  \n" //q0-q1 data
                    // top_s16 -> top_s8
                    "vqmovn.s16   d4, q2            \n"
                    // save top_s8
                    "vst1.8     {d4}, [%2:64]!      \n"
                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    "sub        %1, #32             \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(scale_out),   // %6
                      "r"(bias_tm)      // %7
                    : "cc", "memory", "q0", "q1", "q2", "q10", "q12"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    *ptr = float2int8((*intptr + bias_tm) * scale_out);

                    intptr++;
                    ptr ++;        
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                float32x4_t _scale_out = vdupq_n_f32(scale_out);

                if (nn > 0)
                {
                asm volatile(                                          
                    "dup    v2.4s, %w6                   \n" //q10 scale_out
                    "0:                                  \n"
                    "prfm   pldl1keep, [%1, #128]      \n"
                    "ld1    {v0.4s, v1.4s}, [%1], #32    \n" //q0-q1 data                      
                    // top_s32 -> top_f32
                    "scvtf  v5.4s, v0.4s                 \n"
                    "scvtf  v6.4s, v1.4s                 \n"
                    // top_f32 = top_f32 * scale_out
                    "fmul   v5.4s, v5.4s, v2.4s          \n"
                    "fmul   v6.4s, v6.4s, v2.4s          \n"
                    // top_f32 -> top_s32
                    "fcvtas v7.4s, v5.4s                 \n"
                    "fcvtas v8.4s, v6.4s                 \n"
                    // top_s32 -> top_s16
                    "sqxtn  v9.4h, v7.4s                 \n"
                    "sqxtn2 v9.8h, v8.4s                 \n"
                    // top_s16 -> top_s8
                    "sqxtn  v10.8b, v9.8h                \n"
                    // save top_s8
                    "st1    {v10.8b}, [%2], #8           \n"
                    "subs   %w0, %w0, #1                 \n"
                    "bne    0b                           \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(scale_out)    // %6
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1:128]!  \n" //q0-q1 data
                    "vdup.f32   q10, %6             \n" //q10 scale_out

                    "0:                             \n"
                    // top_s32 -> top_f32
                    "vcvt.f32.s32 q0, q0            \n" 
                    "vcvt.f32.s32 q1, q1            \n"

                    // top_f32 = top_f32 * scale_out
                    "vmul.f32   q0, q0, q10         \n"
                    "vmul.f32   q1, q1, q10         \n"

                    // top_f32 -> top_s32
                    "vcvtr.s32.f32 s0, s0           \n"
                    "vcvtr.s32.f32 s1, s1           \n"
                    "vcvtr.s32.f32 s2, s2           \n"
                    "vcvtr.s32.f32 s3, s3           \n"
                    "vcvtr.s32.f32 s4, s4           \n"
                    "vcvtr.s32.f32 s5, s5           \n"
                    "vcvtr.s32.f32 s6, s6           \n"
                    "vcvtr.s32.f32 s7, s7           \n" 

                    // top_s32 -> top_s16
                    "vqmovn.s32 d4, q0              \n"
                    "vqmovn.s32 d5, q1              \n"

                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1:128]!  \n" //q0-q1 data

                    // top_s16 -> top_s8
                    "vqmovn.s16   d4, q2            \n"

                    // save top_s8
                    "vst1.8     {d4}, [%2:64]!      \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %1, #32             \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(scale_out)    // %6
                    : "cc", "memory", "q0", "q1", "q2", "q10"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    *ptr = float2int8(*intptr * scale_out);

                    intptr++;
                    ptr++;                 
                }
            }
        }    
    }

#if !__aarch64__
    asm volatile(
        "vmsr   FPSCR, %0           \n"
        :
        : "r"(FPSCR_value)
        : "memory"
    );
#endif

    //double end = ncnn::get_current_time();
    //fprintf(stderr, "requantize : %8.2lfms\n", end - start);

    return 0;
}

} // namespace ncnn
