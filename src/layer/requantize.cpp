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

#include "requantize.h"

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

DEFINE_LAYER_CREATOR(Requantize)

Requantize::Requantize()
{
    one_blob_only = true;
    support_inplace = true;
    fusion_relu = false;
}

int Requantize::load_param(const ParamDict& pd)
{
    scale_in = pd.get(0, 1.f);	// bottom_blob_scale * weight_scale
	scale_out = pd.get(1, 1.f);	// top_blob_scale / (bottom_blob_scale * weight_scale)
    bias_term = pd.get(2, 0);
    bias_data_size = pd.get(3, 0);
    fusion_relu = pd.get(4, 0);

    return 0;
}

int Requantize::load_model(const ModelBin& mb)
{
    if (bias_term)
    {
        bias_data = mb.load(bias_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Requantize::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{ 
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        const int* intptr = bottom_top_blob;
        signed char * ptr = bottom_top_blob;

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
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const int* intptr = bottom_top_blob.row<const int>(i);
                signed char* ptr = bottom_top_blob.row<signed char>(i);

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
                const int* intptr = bottom_top_blob.row<const int>(i);
                signed char* ptr = bottom_top_blob.row<signed char>(i);

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
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;      

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const int* intptr = bottom_top_blob.channel(q);
                signed char* ptr = bottom_top_blob.channel(q);

                float bias = bias_data_size > 1 ? bias_data[q] : bias_data[0];
				float bias_tm = bias * scale_in;

                for (int i=0; i<size; i++)
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
            for (int q=0; q<channels; q++)
            {
                const int* intptr = bottom_top_blob.channel(q);
                signed char* ptr = bottom_top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    ptr[i] = float2int8(intptr[i] * scale_out);
                    if (fusion_relu && ptr[i] < 0)
                        ptr[i] = 0;                    
                }
            }
        }    
    }

    return 0;
}

} // namespace ncnn
