// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Requantize_arm)

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
}

static inline signed char short2int8(short v)
{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (signed char)v;
}

static inline signed char int322int8(int v)
{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (signed char)v;
}

int Requantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{ 
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
                    ptr[i] = float2int8(((intptr[i] * scale_in) + bias_data[i]) * scale_out);
                    if (fusion_relu && ptr[i] < 0)
                        ptr[i] = 0;
                }
            }
            else
            {
                float bias = bias_data[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i=0; i<w; i++)
                {
                    ptr[i] = float2int8(((intptr[i] * scale_in) + bias) * scale_out);
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
                ptr[i] = float2int8(intptr[i] * scale_in * scale_out);
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

                for (int j=0; j<w; j++)
                {
                    ptr[j] = float2int8(((intptr[j] * scale_in) + bias) * scale_out);
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
                    ptr[j] = float2int8(intptr[j] * scale_in * scale_out);
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

        float scale_fuse = scale_in * scale_out;

        if (bias_term)
        {
            // #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const short* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

                float bias = bias_data_size > 1 ? bias_data[q] : bias_data[0];
				float bias_tm = bias / scale_in;

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _bias_tm = vdupq_n_f32(bias_tm);
                float32x4_t _scale_fuse = vdupq_n_f32(scale_fuse);
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(                                          
                    "0:                                  \n"
                    "prfm   pldl1keep, [%1, #128]        \n"
                    "ld1    {v0.8h}, [%1], #16           \n"
                    // top_s16 -> top_s32
                    "sshll2  v1.4s, v0.8h, #0            \n"
                    "sshll  v0.4s, v0.4h, #0             \n"
                    // top_s32 -> top_f32
                    "scvtf  v5.4s, v0.4s                 \n"
                    "scvtf  v6.4s, v1.4s                 \n"
                    // top_f32 = top_f32 + bias_tm
                    "fadd   v5.4s, v5.4s, %6.4s          \n"
                    "fadd   v6.4s, v6.4s, %6.4s          \n"
                    // top_f32 = top_f32 * scale_out
                    "fmul   v5.4s, v5.4s, %7.4s          \n"
                    "fmul   v6.4s, v6.4s, %7.4s          \n"
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
                      "w"(_bias_tm),    // %6
                      "w"(_scale_fuse)  // %7
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.s16   {d0-d1}, [%1:128]!  \n" //q0-q1 data
                    "0:                             \n"  
                    // top_s16 -> top_s32
                    "vmovl.s16  q1, d1              \n"
                    "vmovl.s16  q0, d0              \n"
                    // top_s32 -> top_f32
                    "vcvt.f32.s32 q0, q0            \n" 
                    "vcvt.f32.s32 q1, q1            \n"
                    // top_f32 = top_f32 + bias_tm
                    "vadd.f32   q0, q0, %q6         \n"
                    "vadd.f32   q1, q1, %q6         \n"
                    // top_f32 = top_f32 * scale_out
                    "vmul.f32   q0, q0, %q7         \n"
                    "vmul.f32   q1, q1, %q7         \n"
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
                    "vld1.s16   {d0-d1}, [%1:128]!  \n" //q0-q1 data
                    // top_s16 -> top_s8
                    "vqmovn.s16   d4, q2            \n"
                    // save top_s8
                    "vst1.8     {d4}, [%2:64]!      \n"
                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    "sub        %1, #16             \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "w"(_bias_tm),    // %6
                      "w"(_scale_fuse)  // %7
                    : "cc", "memory", "q0", "q1", "q2"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    *ptr = float2int8((*intptr + bias_tm) * scale_fuse);

                    intptr++;
                    ptr ++;        
                }
            }
        }
        else
        {
            // #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const short* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _scale_fuse = vdupq_n_f32(scale_fuse);
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                                  \n"
                    "prfm   pldl1keep, [%1, #128]        \n"
                    "ld1    {v0.8h}, [%1], #16           \n"
                    // top_s16 -> top_s32
                    "sshll2  v1.4s, v0.8h, #0            \n"
                    "sshll  v0.4s, v0.4h, #0             \n"                                           
                    // top_s32 -> top_f32
                    "scvtf  v5.4s, v0.4s                 \n"
                    "scvtf  v6.4s, v1.4s                 \n"
                    // top_f32 = top_f32 * scale_out
                    "fmul   v5.4s, v5.4s, %6.4s          \n"
                    "fmul   v6.4s, v6.4s, %6.4s          \n"
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
                      "w"(_scale_fuse)  // %6
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.s16   {d0-d1}, [%1:128]!  \n" //q0-q1 data
                    // top_s16 -> top_s32
                    "vmovl.s16  q1, d1              \n"
                    "vmovl.s16  q0, d0              \n"
                    "0:                             \n"
                    // top_s32 -> top_f32
                    "vcvt.f32.s32 q0, q0            \n" 
                    "vcvt.f32.s32 q1, q1            \n"
                    // top_f32 = top_f32 * scale_out
                    "vmul.f32   q0, q0, %q6         \n"
                    "vmul.f32   q1, q1, %q6         \n"
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
                    "vld1.s16   {d0-d1}, [%1:128]!  \n" //q0-q1 data
                    // top_s16 -> top_s32
                    "vmovl.s16  q1, d1              \n"
                    "vmovl.s16  q0, d0              \n"
                    // top_s16 -> top_s8
                    "vqmovn.s16   d4, q2            \n"
                    // save top_s8
                    "vst1.8     {d4}, [%2:64]!      \n"
                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    "sub        %1, #16             \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "w"(_scale_fuse)  // %6
                    : "cc", "memory", "q0", "q1", "q2"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    *ptr = float2int8(*intptr * scale_fuse);

                    intptr++;
                    ptr++;                 
                }
            }
        }    
    }   
    
    return 0;
}

} // namespace ncnn