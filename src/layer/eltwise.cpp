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

#include "eltwise.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Eltwise)

Eltwise::Eltwise()
{
    one_blob_only = false;
    support_inplace = false;// TODO inplace reduction
    use_int8_inference = false;
    use_dequant = false;    
}

int Eltwise::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    coeffs = pd.get(1, Mat());

    return 0;
}

Eltwise::~Eltwise()
{
    bottom_blob_int8_scales.clear();
}

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
}

int Eltwise::forward_quant(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
                signed char* src = bottom_blobs_tmp[i].channel(q);
                signed char* dst = bottom_blobs_tmp[i].channel(q);

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
                const float* src = bottom_blobs[i].channel(q);
                signed char* dst = bottom_blobs_tmp[i].channel(q);

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
            char comment_in[128] = {'\0'};
            char comment_out[128] = {'\0'};

            sprintf(comment_in, "D_%d_In_S8", i);
            sprintf(comment_out, "D_%d_Out_S8", i);

            extract_feature_blob_s8(comment_in, this->name.c_str(), bottom_blobs[i]);
            extract_feature_blob_s8(comment_out, this->name.c_str(), bottom_blobs_tmp[i]);
#endif              
        }             
    }   

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

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] + ptr1[i];
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

                for (int i=0; i<size; i++)
                {
                    outptr[i] += ptr[i];
                }
            }
        }
    }
    else
    {
        fprintf(stderr, "Eltwise int8 just support sum op\n");
        return -1;
    }

    return 0;
}

int Eltwise::forward_dequant(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = 4u;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    std::vector<Mat> bottom_blobs_tmp;
    bottom_blobs_tmp.resize(bottom_blobs.size());

    // dequant bottom blob from int8 to fp32
    for (size_t i=0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blob_int8_scales[i] != 0)
        {
            bottom_blobs_tmp[i].create(w, h, channels, elemsize, opt.workspace_allocator);
            float scale_fuse = 1 / bottom_blob_int8_scales[i];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const signed char* src = bottom_blobs[i].channel(q);
                float* dst = bottom_blobs_tmp[i].channel(q);

                for (int i=0; i<size; i++)
                {
                    *dst = (float)(*src) * scale_fuse;
 
                    src++;
                    dst++;
                }
            }            
#if DEBUG_FEATURE
            printf("bottom %d, scale = %f\n", i, bottom_blob_int8_scales[i]);
            char comment_in[128] = {'\0'};
            char comment_out[128] = {'\0'};

            sprintf(comment_in, "D_%d_In_S8", i);
            sprintf(comment_out, "D_%d_Out_FP32", i);

            extract_feature_blob_s8(comment_in, this->name.c_str(), bottom_blobs[i]);
            extract_feature_blob_f32(comment_out, this->name.c_str(), bottom_blobs_tmp[i]);
#endif             
        }
        else
        {
            bottom_blobs_tmp[i] = bottom_blobs[i];

#if DEBUG_FEATURE
            printf("bottom %d, scale = %f\n", i, bottom_blob_int8_scales[i]);
            char comment_in[128] = {'\0'};
            char comment_out[128] = {'\0'};

            sprintf(comment_in, "D_%d_In_FP32", i);
            sprintf(comment_out, "D_%d_Out_FP32", i);

            extract_feature_blob_f32(comment_in, this->name.c_str(), bottom_blobs[i]);
            extract_feature_blob_f32(comment_out, this->name.c_str(), bottom_blobs_tmp[i]);
#endif             
        }                     
    }

    if (op_type == Operation_SUM && coeffs.w == 0)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs_tmp[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blobs_tmp[0].channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] + ptr1[i];
            }
        }

        for (size_t b=2; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs_tmp[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    outptr[i] += ptr[i];
                }
            }
        }
    }
    else
    {
        fprintf(stderr, "Eltwise int8 dequant just support sum op\n");
        return -1;
    }

    return 0;
}

int Eltwise::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (use_int8_inference == true)
        return Eltwise::forward_quant(bottom_blobs, top_blobs, opt);
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

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] * ptr1[i];
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

                for (int i=0; i<size; i++)
                {
                    outptr[i] *= ptr[i];
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

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] + ptr1[i];
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

                    for (int i=0; i<size; i++)
                    {
                        outptr[i] += ptr[i];
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * coeff0 + ptr1[i] * coeff1;
                }
            }

            for (size_t b=2; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        outptr[i] += ptr[i] * coeff;
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

            for (int i=0; i<size; i++)
            {
                outptr[i] = std::max(ptr[i], ptr1[i]);
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

                for (int i=0; i<size; i++)
                {
                    outptr[i] = std::max(outptr[i], ptr[i]);
                }
            }
        }
    }

#if DEBUG_FEATURE
    extract_feature_out_f32(0, this->name.c_str(), top_blob);
#endif    

    return 0;
}

} // namespace ncnn
