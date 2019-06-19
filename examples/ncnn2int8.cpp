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

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

#include <fstream>
#include <vector>
#include <set>
#include <limits>
#include <map>
#include <algorithm>


static bool read_int8scale_table(const char* filepath, std::map<std::string, std::vector<float> >& blob_int8scale_table, std::map<std::string, std::vector<float> >& weight_int8scale_table)
{
    blob_int8scale_table.clear();
    weight_int8scale_table.clear();

    FILE* fp = fopen(filepath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", filepath);
        return false;
    }

    bool in_scale_vector = false;

    std::string keystr;
    std::vector<float> scales;

    while (!feof(fp))
    {
        char key[256];
        int nscan = fscanf(fp, "%255s", key);
        if (nscan != 1)
        {
            break;
        }

        if (in_scale_vector)
        {
            float scale = 1.f;
            int nscan = sscanf(key, "%f", &scale);
            if (nscan == 1)
            {
                scales.push_back(scale);
                continue;
            }
            else
            {
                // XYZ_param_N pattern
                if (strstr(keystr.c_str(), "_param_"))
                {
                    weight_int8scale_table[ keystr ] = scales;
                }
                else
                {
                    blob_int8scale_table[ keystr ] = scales;
                }

                keystr.clear();
                scales.clear();

                in_scale_vector = false;
            }
        }

        if (!in_scale_vector)
        {
            keystr = key;

            in_scale_vector = true;
        }
    }

    if (in_scale_vector)
    {
        // XYZ_param_N pattern
        if (strstr(keystr.c_str(), "_param_"))
        {
            weight_int8scale_table[ keystr ] = scales;
        }
        else
        {
            blob_int8scale_table[ keystr ] = scales;
        }
    }

    fclose(fp);

    return true;
}

// round to nearest
static signed char float2int8(float value)
{
    float tmp;
    if (value >= 0.f) tmp = value + 0.5;
    else tmp = value - 0.5;

    if (tmp > 127)
        return 127;
    if (tmp < -127)
        return -127;

    return tmp;
}

static int quantize_weight(float *data, size_t data_length, std::vector<float> scales, std::vector<signed char>& int8_weights)
{
    int8_weights.resize(data_length);

    int length_per_group = data_length / scales.size();

    for (size_t i = 0; i < data_length; i++)
    {
        float f = data[i];

        signed char int8 = float2int8(f * scales[ i / length_per_group ]);

        int8_weights[i] = int8;
    }

    // magic tag for int8
    return 0x000D4B38;
}

static bool quantize_weight(float *data, size_t data_length, int quantize_level, std::vector<float> &quantize_table, std::vector<unsigned char> &quantize_index) {

    assert(quantize_level != 0);
    assert(data != NULL);
    assert(data_length > 0);

    if (data_length < static_cast<size_t>(quantize_level)) {
        fprintf(stderr, "No need quantize,because: data_length < quantize_level");
        return false;
    }

    quantize_table.reserve(quantize_level);
    quantize_index.reserve(data_length);

    // 1. Find min and max value
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();

    for (size_t i = 0; i < data_length; ++i)
    {
        if (max_value < data[i]) max_value = data[i];
        if (min_value > data[i]) min_value = data[i];
    }
    float strides = (max_value - min_value) / quantize_level;

    // 2. Generate quantize table
    for (int i = 0; i < quantize_level; ++i)
    {
        quantize_table.push_back(min_value + i * strides);
    }

    // 3. Align data to the quantized value
    for (size_t i = 0; i < data_length; ++i)
    {
        size_t table_index = int((data[i] - min_value) / strides);
        table_index = std::min<float>(table_index, quantize_level - 1);

        float low_value  = quantize_table[table_index];
        float high_value = low_value + strides;

        // find a nearest value between low and high value.
        float targetValue = data[i] - low_value < high_value - data[i] ? low_value : high_value;

        table_index = int((targetValue - min_value) / strides);
        table_index = std::min<float>(table_index, quantize_level - 1);
        quantize_index.push_back(table_index);
    }

    return true;
}

int main(int argc, char** argv)
{
    const char* int8scale_table_path = "./squeezenet.table";

    // parse the calibration scale table
    std::map<std::string, std::vector<float> > blob_int8scale_table;
    std::map<std::string, std::vector<float> > weight_int8scale_table;
    if (int8scale_table_path)
    {
        bool s2 = read_int8scale_table(int8scale_table_path, blob_int8scale_table, weight_int8scale_table);
        if (!s2)
        {
            fprintf(stderr, "read_int8scale_table failed\n");
            return -1;
        }
    }

    // std::map<std::string, std::vector<float> >::iterator iter = blob_int8scale_table.begin();
    // while(iter != blob_int8scale_table.end())
    // {
    //     std::vector<float> blob_scales = iter->second;
    //     printf("%s ", iter->first.c_str());
    //     for (size_t i=0; i<blob_scales.size(); i++)
    //         printf("%f ", blob_scales[i]);
    //     printf("\n");

    //     iter++;
    // } 

    // parse ncnn original ncnn.param and ncnn.bin

    // loop every layer and quantize the convolution layer which has been masked in the scale table

        // find the layer depend on the layer_name

            // quantize the weight data
            
            // save the int8 weight data 
            
            // save the float32 bias data

            // save the weight scale and data scale
        

    return 0;
}