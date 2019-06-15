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

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "platform.h"
#include "net.h"
#include "../src/layer/convolution.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

namespace ncnn {

class QuantNet : public Net
{
public:
    std::map<std::string,std::string> get_conv_bottom_blob_names()
    {
        std::map<std::string,std::string> conv_bottom_blob_names;

        // fine conv bottom name or index
        for (size_t i=0; i<layers.size(); i++)
        {
            Layer* layer = layers[i];
            
            if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise")
            {
                std::string name = layer->name;
                std::string bottom_blob_name = blobs[layer->bottoms[0]].name;
                // fprintf(stderr, "%-20s : bottom_index = %-3d name = %-20s\n", layer->name.c_str(), layer->bottoms[0], blobs[layer->bottoms[0]].name.c_str());
                conv_bottom_blob_names[name] = bottom_blob_name;
            }
        }

        return conv_bottom_blob_names;
    }

    std::vector<std::string> get_conv_names()
    {
        std::vector<std::string> conv_names;

        for (size_t i=0; i<layers.size(); i++)
        {
            Layer* layer = layers[i];
            
            if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise")
            {
                std::string name = layer->name;
                conv_names.push_back(name);
            }
        }        

        return conv_names;
    }

    std::map<std::string,std::vector<float> > get_conv_weight_blob_scales()
    {
        std::map<std::string,std::vector<float> > weight_scales;

        for (size_t i=0; i<layers.size(); i++)
        {
            Layer* layer = layers[i];
            
            if (layer->type == "Convolution")
            {
                std::string name = layer->name;
                const int weight_data_size_output = ((Convolution*)layer)->weight_data_size / ((Convolution*)layer)->num_output;
                std::vector<float> scales;

                for (int n=0; n<((Convolution*)layer)->num_output; n++)
                {
                    const Mat weight_data_n = ((Convolution*)layer)->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                    const float *data_n = weight_data_n;
                    float max_value = std::numeric_limits<float>::min();

                    for (int i = 0; i < weight_data_size_output; i++)
                        max_value = std::max(max_value, std::fabs(data_n[i]));

                    scales.push_back(127 / max_value); 
                }

                weight_scales[name] = scales;
            }
        }              

        return weight_scales;
    }
};

} // namespace ncnn


class QuantizeData
{
public:
    QuantizeData(std::string layer_name, int num)
    {
        name = layer_name;
        max_value = 0.0;
        num_bins = num;
        histogram_interval = 0.0;
        histogram.resize(num_bins);
    }

    int initial_blob_max(ncnn::Mat data)
    {
        int channel_num = data.c;
        int size = data.w * data.h;

        for (int q=0; q<channel_num; q++)
        {
            const float *data_n = data.channel(q);
            for(int i=0; i<size; i++)
            {
                max_value = std::max(max_value, std::fabs(data_n[i]));
            }
        }

        return 0;
    }

    int initial_histogram_interval()
    {
        histogram_interval = num_bins / max_value;

        return 0;
    }

    int update_histogram(ncnn::Mat data)
    {
        int channel_num = data.c;
        int size = data.w * data.h;

        for (int q=0; q<channel_num; q++)
        {
            const float *data_n = data.channel(q);
            for(int i=0; i<size; i++)
            {
                if (data_n[i] == 0)
                    continue;

                int index = std::min(std::fabs(data_n[i]) * histogram_interval - 1, 2047.f);

                histogram[index]++;
            }
        }        

        return 0;
    }

    // 计算两个分布之间的 KL 散度
    float compute_kl_divergence(const std::vector<float> &dist_a, const std::vector<float> &dist_b) 
    {
        const int length = dist_a.size();
        assert(dist_b.size() == length);
        float result = 0;

        for (int i=0; i<length; i++) 
        {
            if (dist_a[i] != 0) 
            {
                if (dist_b[i] == 0) 
                {
                    result += 1;
                } 
                else 
                {
                    result += dist_a[i] * log(dist_a[i] / dist_b[i]);
                }
            }
        }

        return result;
    }

    int threshold_distribution(const std::vector<double> &distribution, const int target_bin=128) 
    {
        int target_threshold = target_bin;
        float min_kl_divergence = 1000;
        const int length = distribution.size();

        // 量化后的分布
        std::vector<float> quantize_distribution(target_bin);

        // 计算后面所有值的和
        float threshold_sum = 0;
        for (int threshold=target_bin; threshold<length; threshold++) 
        {
            threshold_sum += distribution[threshold];
        }

        // 取后面的每个值作为阈值
        for (int threshold=target_bin; threshold<length; threshold++) 
        {
            // 获取根据阈值截断的分布
            std::vector<float> t_distribution(distribution.begin(), distribution.begin()+threshold);
            // 将后面的值加到最后一个位置
            t_distribution[threshold-1] += threshold_sum;
            // 更新 threshold_sum 的值，减掉当前的值，下一次循环就能直接使用
            threshold_sum -= distribution[threshold];

            // ************************ 对原分布的前 threshold 个元素进行量化 ************************
            // 初始化量化分布为 0
            fill(quantize_distribution.begin(), quantize_distribution.end(), 0);
            // 计算每个 bin 的元素个数，这里的 num_per_bin >= 1
            const float num_per_bin = static_cast<float>(threshold) / target_bin;

            // 对于每一个 bin, 将元素分配到 bin 里面
            for (int i=0; i<target_bin; i++) 
            {
                // 起始下标
                const float start = i * num_per_bin;
                const float end = start + num_per_bin;

                // 左边的部分，向上取整
                const int left_upper = ceil(start);
                if (left_upper > start) 
                {
                    // 减掉取整前的数值，得到左边不足一格的比例
                    const float left_scale = left_upper - start;
                    // 获取左边不足一格的值
                    quantize_distribution[i] += left_scale * distribution[left_upper - 1];
                }

                // 右边的部分，向下取整
                const int right_lower = floor(end);
                // 判断是否到达最后一个元素了
                if (right_lower < end) 
                {
                    // 取整前的数值减掉取整后的，得到右边不足一格的比例
                    const float right_scale = end - right_lower;
                    // 获取右边不足一格的值
                    quantize_distribution[i] += right_scale * distribution[right_lower];
                }

                // 获取中间的值
                for (int j=left_upper; j<right_lower; j++) 
                {
                    quantize_distribution[i] += distribution[j];
                }
            }

            // ************************ 将量化后的分布扩展回来 ************************
            // 初始化扩展后的分布为 0
            std::vector<float> expand_distribution(threshold, 0);

            // 对于每一个 bin, 将量化后的元素分配到扩展的分布
            for (int i=0; i<target_bin; i++) 
            {
                // 起始下标
                const float start = i * num_per_bin;
                const float end = start + num_per_bin;

                // 非零元素的个数，可以是半个
                float count = 0;

                // 左边的部分，向上取整
                const int left_upper = ceil(start);
                float left_scale;
                if (left_upper > start) 
                {
                    // 减掉取整前的数值，得到左边不足一格的比例
                    left_scale = left_upper - start;
                    // 如果非零，则加入到非零元素的个数
                    if (distribution[left_upper - 1] != 0) 
                    {
                        count += left_scale;
                    }
                }

                // 右边的部分，向下取整
                const int right_lower = floor(end);
                float right_scale = 0;
                if (right_lower < end) 
                {
                    // 取整前的数值减掉取整后的，得到右边不足一格的比例
                    right_scale = end - right_lower;
                    // 如果非零，则加入到非零元素的个数
                    if (distribution[right_lower] != 0) 
                    {
                        count += right_scale;
                    }
                }

                // 计算剩下的非零元素的值
                for (int j=left_upper; j<right_lower; j++) 
                {
                    if (distribution[j] != 0) 
                    {
                        count++;
                    }
                }

                // 扩展出来的值
                const float expand_value = quantize_distribution[i] / count;

                // 进行扩展
                if (left_upper > start) 
                {
                    if (distribution[left_upper - 1] != 0) 
                    {
                        expand_distribution[left_upper - 1] += expand_value * left_scale;
                    }
                }
                if (right_lower < end) 
                {
                    if (distribution[right_lower] != 0) 
                    {
                        expand_distribution[right_lower] += expand_value * right_scale;
                    }
                }
                for (int j=left_upper; j<right_lower; j++) 
                {
                    if (distribution[j] != 0) 
                    {
                        expand_distribution[j] += expand_value;
                    }
                }
            }

            // 计算 KL 散度
            float kl_divergence = compute_kl_divergence(t_distribution, expand_distribution);

            // 保留最小的 KL 散度对应的阈值
            if (kl_divergence < min_kl_divergence) 
            {
                min_kl_divergence = kl_divergence;
                target_threshold = threshold;
            }
        }

        return target_threshold;
    }

    float get_data_blob_scale()
    {   
        threshold_bin = threshold_distribution(histogram);
        threshold = (threshold_bin + 0.5) * histogram_interval;
        scale = 127 / threshold;
        return scale;
    }

public:
    std::string name;
    float max_value;
    float histogram_interval;
    int num_bins;
    std::vector<double> histogram;
    float threshold;
    int threshold_bin;
    float scale;
};

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::QuantNet squeezenet;

#if NCNN_VULKAN
    squeezenet.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    squeezenet.load_param("squeezenet_v1.1.param");
    squeezenet.load_model("squeezenet_v1.1.bin");  

    std::map<std::string,std::string> conv_bottom_blob_names = squeezenet.get_conv_bottom_blob_names();
    std::vector<std::string> conv_names = squeezenet.get_conv_names();
    std::map<std::string,std::vector<float> > weight_scales = squeezenet.get_conv_weight_blob_scales(); 


    FILE *fp=fopen("squeezenet.table", "w");

    // debug quantize weight
    // step 0 quantize weight
    printf("====> step 0\n");    
    for (size_t i=0; i<conv_names.size(); i++)
    {
        std::string layer_name = conv_names[i];
        std::string blob_name = conv_bottom_blob_names[layer_name];
        std::vector<float> weight_scale_n = weight_scales[layer_name];

        // fprintf(stderr, "%-20s :", layer_name.c_str());
        // for (size_t j=0; j<weight_scale_n.size(); j++)
        //     fprintf(stderr, "%f ", weight_scale_n[j]);
        // fprintf(stderr, "\n");
        fprintf(fp, "%s_param0 ", layer_name.c_str());
        for (size_t j=0; j<weight_scale_n.size(); j++)
            fprintf(fp, "%f ", weight_scale_n[j]);
        fprintf(fp, "\n");        
    }

    // debug quantize data
    std::vector<QuantizeData> quantize_datas;
    
    for (size_t i=0; i<conv_names.size(); i++)
    {
        std::string layer_name = conv_names[i];

        QuantizeData quantize_data(layer_name, 2048);
        quantize_datas.push_back(quantize_data);
    }    

    // step 1 count the max value
    printf("====> step 1\n");
    {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
        const float mean_vals[3] = {104.f, 117.f, 123.f};
        in.substract_mean_normalize(mean_vals, 0);

        ncnn::Extractor ex = squeezenet.create_extractor();
        ex.input("data", in);

        for (size_t i=0; i<conv_names.size(); i++)
        {
            std::string layer_name = conv_names[i];
            std::string blob_name = conv_bottom_blob_names[layer_name];

            // fprintf(stderr, "%-20s : %s \n", layer_name.c_str(), blob_name.c_str());

            ncnn::Mat out;
            ex.extract(blob_name.c_str(), out);  
            // fprintf(stderr, "[%d, %d, %d]\n", out.c, out.h, out.w);

            for (size_t j=0; j<quantize_datas.size(); j++)
            {
                if (quantize_datas[j].name == layer_name)
                {
                    quantize_datas[j].initial_blob_max(out);
                    break;
                }
            }
        }
    }

    // step 2 histogram_interval
    printf("====> step 2\n");
    for (size_t i=0; i<conv_names.size(); i++)
    {
        std::string layer_name = conv_names[i];

        for (size_t j=0; j<quantize_datas.size(); j++)
        {
            if (quantize_datas[j].name == layer_name)
            {
                quantize_datas[j].initial_histogram_interval();
                break;
            }
        }
    }    

    // step 3 kld
    printf("====> step 3\n");
    {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
        const float mean_vals[3] = {104.f, 117.f, 123.f};
        in.substract_mean_normalize(mean_vals, 0);
        ncnn::Extractor ex = squeezenet.create_extractor();

        ex.input("data", in);

        for (size_t i=0; i<conv_names.size(); i++)
        {
            std::string layer_name = conv_names[i];
            std::string blob_name = conv_bottom_blob_names[layer_name];

            ncnn::Mat out;
            ex.extract(blob_name.c_str(), out);  

            for (size_t j=0; j<quantize_datas.size(); j++)
            {
                if (quantize_datas[j].name == layer_name)
                {
                    quantize_datas[j].update_histogram(out);
                    break;
                }
            }
        }
    }

    // step4
    printf("====> step 4\n");
    for (size_t i=0; i<conv_names.size(); i++)
    {
        std::string layer_name = conv_names[i];
        std::string blob_name = conv_bottom_blob_names[layer_name];
        fprintf(stderr, "%-20s : ", layer_name.c_str());

        for (size_t j=0; j<quantize_datas.size(); j++)
        {
            if (quantize_datas[j].name == layer_name)
            {
                quantize_datas[j].get_data_blob_scale();
                fprintf(stderr, "bin : %-8d threshold : %-15f interval : %-10f scale : %-10f\n", \
                                                                quantize_datas[j].threshold_bin, \
                                                                quantize_datas[j].threshold, \
                                                                quantize_datas[j].histogram_interval, \
                                                                quantize_datas[j].scale);

                fprintf(fp, "%s %f\n", layer_name.c_str(), quantize_datas[j].scale);

                break;
            }
        }
    }    


    fclose(fp);
    // for (size_t j=0; j<quantize_datas.size(); j++)
    // {
    //     if (quantize_datas[j].name == "conv1")
    //     {
    //         printf("hist:\n");
    //         for (size_t i=0; i<quantize_datas[j].histogram.size(); i++)
    //         {
    //             printf("%f ", quantize_datas[j].histogram[i]);
    //         }
    //         printf("\n");
    //     }
    // }

    // debug quantize weight
    // for (size_t i=0; i<conv_names.size(); i++)
    // {
    //     std::string layer_name = conv_names[i];
    //     std::string blob_name = conv_bottom_blob_names[layer_name];
    //     std::vector<float> weight_scale_n = weight_scales[layer_name];

    //     fprintf(stderr, "%-20s :", layer_name.c_str());
    //     for (size_t j=0; j<weight_scale_n.size(); j++)
    //         fprintf(stderr, "%f ", weight_scale_n[j]);
    //     fprintf(stderr, "\n");
    // }

    // ncnn::Mat out0, out1;

    // ex.extract("pool1", out0);
    // ex.extract("fire9/concat_drop9", out1);

    // cls_scores.resize(out.w);
    // for (int j=0; j<out.w; j++)
    // {
    //     cls_scores[j] = out[j];
    // }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    // print_topk(cls_scores, 5);

    return 0;
}
