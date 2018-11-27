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

#include "innerproduct.h"

#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct)

InnerProduct::InnerProduct()
{
    one_blob_only = true;
    support_inplace = false;

    quantize = 0;
}

InnerProduct::~InnerProduct()
{
    delete quantize;

    for (int i=0; i<(int)dequantize_ops.size(); i++)
        delete dequantize_ops[i];

    dequantize_ops.clear();
}

int InnerProduct::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(8, 0);

    use_int8_inference = pd.use_int8_inference;

    if (int8_scale_term == 0)
        use_int8_inference = false;

    return 0;
}

int InnerProduct::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    if (int8_scale_term)
    {
        weight_data_int8_scales = mb.load(num_output, 1);
        bottom_blob_int8_scale = mb.load(1, 1)[0];
    }

    bool weight_data_is_int8 = (weight_data.elemsize == (size_t)1u);
    bool weight_data_is_float32 = (weight_data.elemsize == (size_t)4u);

    if (weight_data_is_int8 && !use_int8_inference)
    {
        fprintf(stderr, "quantized int8 weight loaded but use_int8_inference disabled\n");
        return -1;
    }

    if (use_int8_inference)
    {
        quantize = ncnn::create_layer(ncnn::LayerType::Quantize);
        {
            ncnn::ParamDict pd;
            pd.set(0, bottom_blob_int8_scale);// scale

            quantize->load_param(pd);
        }

        dequantize_ops.resize(num_output);
        for (int g=0; g<num_output; g++)
        {
            dequantize_ops[g] = ncnn::create_layer(ncnn::LayerType::Dequantize);

            float top_rescale = 1.f;

            if (weight_data_int8_scales[g] == 0)
                top_rescale = 0;
            else
                top_rescale = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[g]);

            ncnn::ParamDict pd;
            pd.set(0, top_rescale);// scale
            pd.set(1, bias_term);// bias_term
            pd.set(2, 1);// bias_data_size

            dequantize_ops[g]->load_param(pd);

            ncnn::Mat weights[1];
            weights[0] = bias_data.range(g, 1);

            dequantize_ops[g]->load_model(ModelBinFromMatArray(weights));
        }        
    }

    // if (weight_data_is_float32 && use_int8_inference)
    // {
    //     // quantize weight to int8
    //     ncnn::ParamDict pd;
    //     pd.set(0, weight_data_int8_scale);// scale

    //     quantize->load_param(pd);

    //     Mat int8_weight_data;
    //     quantize->forward(weight_data, int8_weight_data);

    //     if (int8_weight_data.empty())
    //         return -100;

    //     weight_data = int8_weight_data;
    // }

    return 0;
}

int InnerProduct::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (use_int8_inference)
    {
        Mat bottom_blob_int8;
        bottom_blob_int8.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
        if (bottom_blob_int8.empty())
            return -100;

        // quantize, scale and round to nearest
        {
            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = bottom_blob_int8.allocator;

            quantize->forward(bottom_blob, bottom_blob_int8, opt_g);
        }

        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            int sum = 0;
            int* out = top_blob;

            // channels
            for (int q=0; q<channels; q++)
            {
                const signed char* w = (const signed char*)weight_data + size * channels * p + size * q;
                const signed char* m = bottom_blob_int8.channel(q);

                for (int i = 0; i < size; i++)
                {
                    sum += m[i] * w[i];
                }
            }

            out[p] = sum;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            int* out_s32 = top_blob;
            float* out_f32 = top_blob;

            float top_rescale = 1.f;

            if (weight_data_int8_scales[p] == 0)
                top_rescale = 0;
            else
                top_rescale = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

            if (bias_term)
            {
                out_f32[p] = out_s32[p] * top_rescale + bias_data[p];
            }
            else
            {
                out_f32[p] = out_s32[p] * top_rescale;
            }
        }  

        return 0;
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* w = (const float*)weight_data + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        top_blob[p] = sum;
    }

    return 0;
}

} // namespace ncnn
