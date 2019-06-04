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

#include "pooling.h"
#include <float.h>
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Pooling)

Pooling::Pooling()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;
    use_int8_inference = false;
}

int Pooling::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);

    return 0;
}

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
}

int Pooling::forward_quant(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{  
    // The int8 pooling is just support pooling max op

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = 1UL;

    Mat bottom_blob_tmp;
    // printf("Pooling quant : %f, %f\n", bottom_blob_int8_scale, top_blob_int8_scale);
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
        // printf("########## 23333 pooling requant : %f, %f, %f\n", bottom_blob_int8_scale, top_blob_int8_scale, scale_fuse);
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
        // fprintf(stderr, "Pooling int8, the bottom blob is needed quantization\n");
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

    Mat bottom_blob_bordered = bottom_blob_tmp;         
#if DEBUG_TIME  
    start = get_current_time();
#endif 
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

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

#if DEBUG_FEATURE
    extract_feature_blob_s8("bottom_blob_bordered", this->name.c_str(), bottom_blob_bordered);
#endif    

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            signed char* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const signed char* sptr = m.row<signed char>(i*stride_h) + j*stride_w;

                    signed char max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        signed char val = sptr[ space_ofs[k] ];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else
    {
        fprintf(stderr, "The int8 pooling is just support pooling max op(kernel = 2 or 3, stride = 2)\n");
        return -1;
    }
#if DEBUG_TIME 
    end = get_current_time();
    printf("pooling    : %8.3f ms\n", end - start);
#endif
#if DEBUG_FEATURE
    extract_feature_blob_s8("Top_OUT_S8", this->name.c_str(), top_blob);
#endif     

    return 0;
}

int Pooling::forward_dequant(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    fprintf(stderr, "pooling dequant TODO!\n");
        return -1;
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = 4UL;

    Mat bottom_blob_tmp;
    bottom_blob_tmp.create(w, h, channels, elemsize, opt.workspace_allocator);
    float scale = 1 / bottom_blob_int8_scale;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        const signed char* src = bottom_blob.channel(q);
        float* dst = bottom_blob_tmp.channel(q);

        for (int i=0; i<size; i++)
        {
            *dst = (float)(*src) * scale;

            src++;
            dst++;
        }
    }            
#if DEBUG_FEATURE
    printf("bottom scale = %f\n", bottom_blob_int8_scale);
    extract_feature_blob_s8("D_In_S8", this->name.c_str(), bottom_blob);
    extract_feature_blob_f32("D_OUT_F32", this->name.c_str(), bottom_blob_tmp);
#endif      

//     fprintf(stderr, "Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling)
    {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob_tmp.channel(q);

                float max = ptr[0];
                for (int i=0; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                top_blob[q] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob_tmp.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                top_blob[q] = sum / size;
            }
        }

        return 0;
    }

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

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float sum = 0;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        sum += val;
                    }

                    outptr[j] = sum / maxk;
                }

                outptr += outw;
            }

            // fix pad
            if (pad_top != 0)
            {
                const float scale = (float)kernel_h / (kernel_h - pad_top);

                outptr = top_blob.channel(q).row(0);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
            if (pad_bottom + htailpad != 0)
            {
                const float scale = (float)kernel_h / (kernel_h - pad_bottom - htailpad);

                outptr = top_blob.channel(q).row(outh - 1);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
            if (pad_left != 0)
            {
                const float scale = (float)kernel_w / (kernel_w - pad_left);

                outptr = top_blob.channel(q);
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
            if (pad_right + wtailpad != 0)
            {
                const float scale = (float)kernel_w / (kernel_w - pad_right - wtailpad);

                outptr = top_blob.channel(q);
                outptr += outw - 1;
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

int Pooling::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_int8_inference == true)
        return Pooling::forward_quant(bottom_blob, top_blob, opt);
    if (bottom_blob.elemsize == 1u && use_int8_inference == false)
        return Pooling::forward_dequant(bottom_blob, top_blob, opt);        


    // printf("layer name : %s, bottom_blob.elemsize == %d, use_int8_inference == %d\n", name.c_str(), bottom_blob.elemsize, use_int8_inference);
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

//     fprintf(stderr, "Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling)
    {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float max = ptr[0];
                for (int i=0; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                top_blob[q] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                top_blob[q] = sum / size;
            }
        }

        return 0;
    }

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

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float sum = 0;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        sum += val;
                    }

                    outptr[j] = sum / maxk;
                }

                outptr += outw;
            }

            // fix pad
            if (pad_top != 0)
            {
                const float scale = (float)kernel_h / (kernel_h - pad_top);

                outptr = top_blob.channel(q).row(0);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
            if (pad_bottom + htailpad != 0)
            {
                const float scale = (float)kernel_h / (kernel_h - pad_bottom - htailpad);

                outptr = top_blob.channel(q).row(outh - 1);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
            if (pad_left != 0)
            {
                const float scale = (float)kernel_w / (kernel_w - pad_left);

                outptr = top_blob.channel(q);
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
            if (pad_right + wtailpad != 0)
            {
                const float scale = (float)kernel_w / (kernel_w - pad_right - wtailpad);

                outptr = top_blob.channel(q);
                outptr += outw - 1;
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
