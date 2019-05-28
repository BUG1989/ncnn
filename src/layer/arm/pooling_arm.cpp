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

#include "pooling_arm.h"
#include "benchmark.h"
#include <float.h>
namespace ncnn {

#include "pooling_2x2.h"
#include "pooling_3x3.h"
#include "pooling_2x2_int8.h"
#include "pooling_3x3_int8.h"

DEFINE_LAYER_CREATOR(Pooling_arm)

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
}

int Pooling_arm::forward_quant(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // The int8 pooling is just support pooling max op(kernel = 2 or 3, stride = 2)
    if (kernel_w != kernel_h || stride_w != stride_h)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    const int kernel_size = kernel_w;
    // const int stride = stride_w;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = 1UL;

    Mat bottom_blob_tmp;
#if DEBUG_TIME  
    double start, end;
#endif  
    // change bottom blob scale to top quant scale
    if (bottom_blob_int8_scale != top_blob_int8_scale && bottom_blob_int8_scale != 0)
    {
#if DEBUG_TIME  
        start = get_current_time();
#endif         
        bottom_blob_tmp = bottom_blob.clone(opt.workspace_allocator);
        float scale_fuse = top_blob_int8_scale / bottom_blob_int8_scale;
        // printf("##########23333 pooling requant : %f, %f, %f\n", bottom_blob_int8_scale, top_blob_int8_scale, scale_fuse);
        int size = bottom_blob_tmp.w * bottom_blob_tmp.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            signed char* src = bottom_blob_tmp.channel(q);
            signed char* dst = bottom_blob_tmp.channel(q);

            for (int i=0; i<size; i++)
            {
                float temp = (float)(*src);
                temp = temp * scale_fuse;
                *dst = float2int8(temp);
                src++;
                dst++;
            }
        }    

#if DEBUG_FEATURE
        extract_feature_blob_s8("D_In_S8", this->name.c_str(), bottom_blob);
        extract_feature_blob_s8("D_Out_S8", this->name.c_str(), bottom_blob_tmp);
#endif  
#if DEBUG_TIME 
        end = get_current_time();
        printf("quantize   : %8.3f ms\n", end - start);
#endif                        
    }
    else if(bottom_blob_int8_scale == 0)
    {
#if DEBUG_TIME  
        start = get_current_time();
#endif         
        // fprintf(stderr, "Eltwise int8, the bottom blob is needed quantization\n");
        bottom_blob_tmp.create(bottom_blob.w, bottom_blob.h, bottom_blob.c, elemsize, opt.workspace_allocator);
        float scale = top_blob_int8_scale;
        int size = bottom_blob_tmp.w * bottom_blob_tmp.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* src = bottom_blob.channel(q);
            signed char* dst = bottom_blob_tmp.channel(q);

            for (int i=0; i<size; i++)
            {
                float temp = *src;
                temp = temp * scale;
                *dst = float2int8(temp);
                src++;
                dst++;
            }
        }     

#if DEBUG_FEATURE
        extract_feature_blob_f32("D_In_F32", this->name.c_str(), bottom_blob);
        extract_feature_blob_s8("D_Out_S8", this->name.c_str(), bottom_blob_tmp);
#endif      
#if DEBUG_TIME 
        end = get_current_time();
        printf("quantize   : %8.3f ms\n", end - start);
#endif            
    }
    else
    {
        bottom_blob_tmp = bottom_blob;

#if DEBUG_FEATURE
        extract_feature_blob_s8("D_In_S8", this->name.c_str(), bottom_blob);
        extract_feature_blob_s8("D_Out_S8", this->name.c_str(), bottom_blob_tmp);
#endif           
    }            
#if DEBUG_TIME  
    start = get_current_time();
#endif 
    Mat bottom_blob_bordered = bottom_blob_tmp;         

    float pad_value = 0.f;
    if (pooling_type == PoolMethod_MAX)
    {
        pad_value = -FLT_MAX;
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;

        copy_make_border(bottom_blob_tmp, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_mode == 1) // valid padding
    {
        copy_make_border(bottom_blob_tmp, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_mode == 2) // tensorflow padding=SAME
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob_tmp, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (kernel_size == 2)
        pooling2x2s2_max_int8_neon(bottom_blob_bordered, top_blob, opt);
    if (kernel_size == 3)
        pooling3x3s2_max_int8_neon(bottom_blob_bordered, top_blob, opt);

#if DEBUG_TIME 
    end = get_current_time();
    printf("pooling    : %8.3f ms\n", end - start);
#endif

    return 0;
}

int Pooling_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    if (kernel_w != kernel_h || stride_w != stride_h)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    const int kernel_size = kernel_w;
    const int stride = stride_w;

    if (pooling_type != PoolMethod_MAX || stride != 2 || global_pooling == 1)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    if (kernel_size != 2 && kernel_size != 3)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    if (use_int8_inference == true)
        return Pooling_arm::forward_quant(bottom_blob, top_blob, opt);
    if (bottom_blob.elemsize == 1u && use_int8_inference == false)
        return Pooling::forward_dequant(bottom_blob, top_blob, opt);     

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    Mat bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (pooling_type == PoolMethod_MAX)
    {
        pad_value = -FLT_MAX;
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;

        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_mode == 1) // valid padding
    {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_mode == 2) // tensorflow padding=SAME
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (kernel_size == 2)
        pooling2x2s2_max_neon(bottom_blob_bordered, top_blob, opt);
    if (kernel_size == 3)
        pooling3x3s2_max_neon(bottom_blob_bordered, top_blob, opt);

    return 0;
}

} // namespace ncnn
