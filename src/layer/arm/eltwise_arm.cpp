// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "eltwise_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Eltwise_arm)

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
}

int Eltwise_arm::forward_quant(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = 1u;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    std::vector<Mat> bottom_blobs_tmp;
    bottom_blobs_tmp.resize(bottom_blobs.size());

#if DEBUG_TIME  
    double start, end;
#endif  

#if DEBUG_TIME  
    start = get_current_time();
#endif    

    // change bottom blob scale to top quant scale
    for (size_t i=0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blob_int8_scales[i] != top_blob_int8_scale && bottom_blob_int8_scales[i] != 0)
        {
            bottom_blobs_tmp[i] = bottom_blobs[i].clone(opt.workspace_allocator);
            float scale_fuse = top_blob_int8_scale / bottom_blob_int8_scales[i];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                signed char* intptr = bottom_blobs_tmp[i].channel(q);
                signed char* ptr = bottom_blobs_tmp[i].channel(q);

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
                    "ld1    {v0.8b}, [%1], #8            \n"
                    // top_s8 -> top_s16
                    "sshll  v0.8h, v0.8b, #0             \n"
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
                    "vld1.s8    {d0} , [%1]!        \n"
                    // top_s8 -> top_s16
                    "vmovl.s8   q0, d0              \n"
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
                    "vld1.s8    {d0}, [%1]!         \n"
                    // top_s8 -> top_s16
                    "vmovl.s8   q0, d0              \n"                    
                    // top_s16 -> top_s32
                    "vmovl.s16  q1, d1              \n"
                    "vmovl.s16  q0, d0              \n"
                    // top_s16 -> top_s8
                    "vqmovn.s16   d4, q2            \n"
                    // save top_s8
                    "vst1.8     {d4}, [%2:64]!      \n"
                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    "sub        %1, #8              \n"
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

#if DEBUG_FEATURE
            fprintf(stderr, "Eltwise int8, the bottom blob is needed requantization\n");
            char comment_in[128] = {'\0'};
            char comment_out[128] = {'\0'};

            sprintf(comment_in, "D_%d_In_S8", i);
            sprintf(comment_out, "D_%d_Out_S8", i);

            extract_feature_blob_s8(comment_in, this->name.c_str(), bottom_blobs[i]);
            extract_feature_blob_s8(comment_out, this->name.c_str(), bottom_blobs_tmp[i]);
#endif                       
        }
        else if(bottom_blob_int8_scales[i] == 0)
        {
            bottom_blobs_tmp[i].create(bottom_blob.w, bottom_blob.h, bottom_blob.c, 1UL, opt.workspace_allocator);
            float scale = top_blob_int8_scale;
            int size = bottom_blob.w * bottom_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blobs[i].channel(q);
                signed char* outptr = bottom_blobs_tmp[i].channel(q);

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "dup    v2.4s, %w6                   \n" //scale
                    "0:                                  \n"
                    "prfm   pldl1keep, [%1, #128]        \n"
                    "ld1    {v0.4s, v1.4s}, [%1], #32    \n" //data
                    // bottom_f32 = bottom_f32 * scale
                    "fmul   v3.4s, v0.4s, v2.4s          \n"
                    "fmul   v4.4s, v1.4s, v2.4s          \n"
                    // top_f32 -> top_s32
                    "fcvtas v5.4s, v3.4s                 \n"
                    "fcvtas v6.4s, v4.4s                 \n"
                    // top_s32 -> top_s16
                    "sqxtn  v7.4h, v5.4s                 \n"
                    "sqxtn2 v7.8h, v6.4s                 \n"
                    // top_s16 -> top_s8
                    "sqxtn  v8.8b, v7.8h                 \n"
                    // save top_s8
                    "st1    {v8.8b}, [%2], #8            \n"
                    "subs   %w0, %w0, #1                 \n"
                    "bne    0b                           \n"
                    : "=r"(nn),       // %0
                      "=r"(ptr),      // %1
                      "=r"(outptr)    // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr),
                      "r"(scale)      // %6
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]!      \n"
                    "vdup.32    q10, %6             \n"

                    "0:                             \n"
                    "vmul.f32   q0,q0,q10           \n"
                    "vmul.f32   q1,q1,q10           \n"

                    "vcvtr.s32.f32 s0,s0            \n"
                    "vcvtr.s32.f32 s1,s1            \n"
                    "vcvtr.s32.f32 s2,s2            \n"
                    "vcvtr.s32.f32 s3,s3            \n"
                    "vcvtr.s32.f32 s4,s4            \n"
                    "vcvtr.s32.f32 s5,s5            \n"
                    "vcvtr.s32.f32 s6,s6            \n"
                    "vcvtr.s32.f32 s7,s7            \n"

                    "vqmovn.s32 d4,q0               \n"
                    "vqmovn.s32 d5,q1               \n"

                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]!      \n"

                    "vqmovn.s16 d4, q2              \n"
                    "vst1.8     {d4}, [%2]!         \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %1, #32             \n"
                    : "=r"(nn),         // %0
                      "=r"(ptr),        // %1
                      "=r"(outptr)      // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr),
                      "r"(scale)        // %6
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q10", "q11"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr = float2int8(*ptr * scale);

                    ptr++;
                    outptr++;
                }
            }   
#if DEBUG_FEATURE
            fprintf(stderr, "Eltwise int8, the bottom blob is needed quantization\n");
            char comment_in[128] = {'\0'};
            char comment_out[128] = {'\0'};

            sprintf(comment_in, "D_%d_In_FP32", i);
            sprintf(comment_out, "D_%d_Out_S8", i);

            extract_feature_blob_f32(comment_in, this->name.c_str(), bottom_blobs[i]);
            extract_feature_blob_s8(comment_out, this->name.c_str(), bottom_blobs_tmp[i]);
#endif                           
        }
        else
        {
            bottom_blobs_tmp[i] = bottom_blobs[i];

#if DEBUG_FEATURE
            fprintf(stderr, "Eltwise int8, the bottom blob is needed nothing\n");
            fprintf(stderr, "in scale = %f\n", bottom_blob_int8_scales[i]);
            fprintf(stderr, "out scale = %f\n", top_blob_int8_scale);
            char comment_in[128] = {'\0'};
            char comment_out[128] = {'\0'};

            sprintf(comment_in, "D_%d_In_S8", i);
            sprintf(comment_out, "D_%d_Out_S8", i);

            extract_feature_blob_s8(comment_in, this->name.c_str(), bottom_blobs[i]);
            extract_feature_blob_s8(comment_out, this->name.c_str(), bottom_blobs_tmp[i]);
#endif              
        }             
    }   

#if DEBUG_TIME 
    end = get_current_time();
    printf("quantize   : %8.3f ms\n", end - start);
    start = get_current_time();
#endif     

    if (op_type == Operation_SUM && coeffs.w == 0)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs_tmp[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const signed char* ptr = bottom_blobs_tmp[0].channel(q);
            const signed char* ptr1 = bottom_blob1.channel(q);
            signed char* outptr = top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 4;
            int remain = size - (nn << 4);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "0:                                   \n"
                "prfm       pldl1keep, [%1, #128]     \n"
                "prfm       pldl1keep, [%2, #128]     \n"
                "ld1        {v0.16b}, [%1], #16       \n"
                "ld1        {v1.16b}, [%2], #16       \n"
                "add        v0.16b, v0.16b, v1.16b    \n"
                "subs       %w0, %w0, #1              \n"
                "st1        {v0.16b}, [%3], #16       \n"
                "bne        0b                        \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "pld        [%2, #128]          \n"
                "vld1.s8    {d0-d1}, [%1 :128]! \n"
                "vld1.s8    {d2-d3}, [%2 :128]! \n"
                "vaddq.s8   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.s8    {d0-d1}, [%3 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *outptr = *ptr + *ptr1;

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b=2; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs_tmp[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const signed char* ptr = bottom_blob1.channel(q);
                signed char* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 4;
                int remain = size - (nn << 4);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                                   \n"
                    "prfm       pldl1keep, [%1, #128]     \n"
                    "prfm       pldl1keep, [%2, #128]     \n"
                    "ld1        {v0.16b}, [%1], #16       \n"
                    "ld1        {v1.16b}, [%2]            \n"
                    "add        v0.16b, v0.16b, v1.16b    \n"
                    "subs       %w0, %w0, #1              \n"
                    "st1        {v0.16b}, [%2], #16       \n"
                    "bne        0b                        \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.s8    {d0-d1}, [%1 :128]! \n"
                    "vld1.s8    {d2-d3}, [%2 :128]  \n"
                    "vaddq.s8   q0, q0, q1          \n"
                    "subs       %0, #1              \n"
                    "vst1.s8    {d0-d1}, [%2 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr += *ptr;

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    else
    {
        fprintf(stderr, "Eltwise int8 just support sum op\n");
        return -1;
    }

#if DEBUG_TIME 
    end = get_current_time();
    printf("eltwise    : %8.3f ms\n", end - start);
#endif    

    return 0;
}

int Eltwise_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (use_int8_inference == true)
        return Eltwise_arm::forward_quant(bottom_blobs, top_blobs, opt);
    if (use_dequant == true)
        return Eltwise::forward_dequant(bottom_blobs, top_blobs, opt);

    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "prfm       pldl1keep, [%2, #128] \n"
                "ld1        {v0.4s}, [%1], #16    \n"
                "ld1        {v1.4s}, [%2], #16    \n"
                "fmul       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%3], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "pld        [%2, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vmul.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%3 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *outptr = *ptr * *ptr1;

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b=2; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2]         \n"
                    "fmul       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%2], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]  \n"
                    "vmul.f32   q0, q0, q1          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%2 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr *= *ptr;

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    else if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2], #16    \n"
                    "fadd       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%3], #16    \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]! \n"
                    "vadd.f32   q0, q0, q1          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%3 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr)
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr = *ptr + *ptr1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b=2; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

#if __ARM_NEON
                    int nn = size >> 2;
                    int remain = size - (nn << 2);
#else
                    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                               \n"
                        "prfm       pldl1keep, [%1, #128] \n"
                        "prfm       pldl1keep, [%2, #128] \n"
                        "ld1        {v0.4s}, [%1], #16    \n"
                        "ld1        {v1.4s}, [%2]         \n"
                        "fadd       v0.4s, v0.4s, v1.4s   \n"
                        "subs       %w0, %w0, #1          \n"
                        "st1        {v0.4s}, [%2], #16    \n"
                        "bne        0b                    \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr)
                        : "cc", "memory", "v0", "v1"
                    );
                    }
#else
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]! \n"
                        "vld1.f32   {d2-d3}, [%2 :128]  \n"
                        "vadd.f32   q0, q0, q1          \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%2 :128]! \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr)
                        : "cc", "memory", "q0", "q1"
                    );
                    }
#endif // __aarch64__
#endif // __ARM_NEON
                    for (; remain>0; remain--)
                    {
                        *outptr += *ptr;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
        else
        {
            const float* coeffs_ptr = coeffs;

            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs_ptr[0];
            float coeff1 = coeffs_ptr[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _coeff0 = vdupq_n_f32(coeff0);
                float32x4_t _coeff1 = vdupq_n_f32(coeff1);
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2], #16    \n"
                    "fmul       v0.4s, v0.4s, %8.4s   \n"
                    "fmla       v0.4s, v1.4s, %9.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%3], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr),
                      "w"(_coeff0), // %8
                      "w"(_coeff1)  // %9
                    : "cc", "memory", "v0", "v1"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]! \n"
                    "vmul.f32   q0, q0, %q8         \n"
                    "vmla.f32   q0, q1, %q9         \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%3 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr),
                      "w"(_coeff0), // %8
                      "w"(_coeff1)  // %9
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr = *ptr * coeff0 + *ptr1 * coeff1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b=2; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs_ptr[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

#if __ARM_NEON
                    int nn = size >> 2;
                    int remain = size - (nn << 2);
#else
                    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                    float32x4_t _coeff = vdupq_n_f32(coeff);
#if __aarch64__
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                               \n"
                        "prfm       pldl1keep, [%1, #128] \n"
                        "prfm       pldl1keep, [%2, #128] \n"
                        "ld1        {v0.4s}, [%1], #16    \n"
                        "ld1        {v1.4s}, [%2]         \n"
                        "fmla       v1.4s, v0.4s, %6.4s   \n"
                        "subs       %w0, %w0, #1          \n"
                        "st1        {v1.4s}, [%2], #16    \n"
                        "bne        0b                    \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr),
                          "w"(_coeff)   // %6
                        : "cc", "memory", "v0", "v1"
                    );
                    }
#else
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]! \n"
                        "vld1.f32   {d2-d3}, [%2 :128]  \n"
                        "vmla.f32   q1, q0, %q6         \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d2-d3}, [%2 :128]! \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr),
                          "w"(_coeff)   // %6
                        : "cc", "memory", "q0", "q1"
                    );
                    }
#endif // __aarch64__
#endif // __ARM_NEON
                    for (; remain>0; remain--)
                    {
                        *outptr += *ptr * coeff;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
    }
    else if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "prfm       pldl1keep, [%2, #128] \n"
                "ld1        {v0.4s}, [%1], #16    \n"
                "ld1        {v1.4s}, [%2], #16    \n"
                "fmax       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%3], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "pld        [%2, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vmax.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%3 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *outptr = std::max(*ptr, *ptr1);

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b=2; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2]         \n"
                    "fmax       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%2], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%2 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr = std::max(*ptr, *outptr);

                    ptr++;
                    outptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
