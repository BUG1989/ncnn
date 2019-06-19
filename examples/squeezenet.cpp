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
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <stdlib.h>
#include <algorithm>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "platform.h"
#include "net.h"
#include "cpu.h"
#include "benchmark.h"
#include "../src/layer/convolution.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

/*
 * Get the filenames from direct path
 */
int readFileList(const char *base_path, std::vector<std::string>& file_path)
{
    DIR *dir;
    struct dirent *ptr;

    if ((dir=opendir(base_path)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
        {
            continue;
        } 

        std::string path = base_path;
        file_path.push_back(path + ptr->d_name);
    }
    closedir(dir);

    return 1;
}

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


static int detect_squeezenet(const std::vector<std::string> filenames, const char* param_path, const char* bin_path)
{
    int size = filenames.size();

    ncnn::QuantNet squeezenet;

    squeezenet.load_param(param_path);
    squeezenet.load_model(bin_path);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();     

    std::map<std::string,std::string> conv_bottom_blob_names = squeezenet.get_conv_bottom_blob_names();
    std::vector<std::string> conv_names = squeezenet.get_conv_names();
    std::map<std::string,std::vector<float> > weight_scales = squeezenet.get_conv_weight_blob_scales(); 

    FILE *fp=fopen("squeezenet.table", "w");

    // debug quantize weight
    printf("====> Quantize the parameters.\n");    
    for (size_t i=0; i<conv_names.size(); i++)
    {
        std::string layer_name = conv_names[i];
        std::string blob_name = conv_bottom_blob_names[layer_name];
        std::vector<float> weight_scale_n = weight_scales[layer_name];

        // fprintf(stderr, "%-20s :", layer_name.c_str());
        // for (size_t j=0; j<weight_scale_n.size(); j++)
        //     fprintf(stderr, "%f ", weight_scale_n[j]);
        // fprintf(stderr, "\n");
        fprintf(fp, "%s_param_0 ", layer_name.c_str());
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
    printf("====> Quantize the activation.\n"); 
    printf("    ====> step 1 : fine the max value.\n");

    for (size_t i=0; i<filenames.size(); i++)
    {
        std::string img_name = filenames[i];

        if ((i+1)%100 == 0)
            fprintf(stderr, "          %d/%d\n", i+1, (int)size);

        cv::Mat bgr = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
        if (bgr.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
            return -1;
        }         

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
        const float mean_vals[3] = {104.f, 117.f, 123.f};
        in.substract_mean_normalize(mean_vals, 0);

        // double start = ncnn::get_current_time();
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

        // double end = ncnn::get_current_time();
        // double time = end - start;
        // printf("iter cost: %.8lf ms, %s\n", time, img_name.c_str());            
    }

    // {
    //     ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
    //     const float mean_vals[3] = {104.f, 117.f, 123.f};
    //     in.substract_mean_normalize(mean_vals, 0);

    //     ncnn::Extractor ex = squeezenet.create_extractor();
    //     ex.input("data", in);

    //     for (size_t i=0; i<conv_names.size(); i++)
    //     {
    //         std::string layer_name = conv_names[i];
    //         std::string blob_name = conv_bottom_blob_names[layer_name];

    //         // fprintf(stderr, "%-20s : %s \n", layer_name.c_str(), blob_name.c_str());

    //         ncnn::Mat out;
    //         ex.extract(blob_name.c_str(), out);  
    //         // fprintf(stderr, "[%d, %d, %d]\n", out.c, out.h, out.w);

    //         for (size_t j=0; j<quantize_datas.size(); j++)
    //         {
    //             if (quantize_datas[j].name == layer_name)
    //             {
    //                 quantize_datas[j].initial_blob_max(out);
    //                 break;
    //             }
    //         }
    //     }
    // }

    // step 2 histogram_interval
    printf("    ====> step 2 : generatue the histogram_interval.\n");
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

    // step 3 histogram
    printf("    ====> step 3 : generatue the histogram.\n");
    for (size_t i=0; i<filenames.size(); i++)
    {
        std::string img_name = filenames[i];

        if ((i+1)%100 == 0)
            fprintf(stderr, "          %d/%d\n", i+1, (int)size);

        cv::Mat bgr = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
        if (bgr.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
            return -1;
        }  

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
        const float mean_vals[3] = {104.f, 117.f, 123.f};
        in.substract_mean_normalize(mean_vals, 0);

        // double start = ncnn::get_current_time();        
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

        // double end = ncnn::get_current_time();
        // double time = end - start;
        // printf("iter cost: %.8lf ms, %s\n", time, img_name.c_str());          
    }

    // {
    //     ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
    //     const float mean_vals[3] = {104.f, 117.f, 123.f};
    //     in.substract_mean_normalize(mean_vals, 0);
    //     ncnn::Extractor ex = squeezenet.create_extractor();

    //     ex.input("data", in);

    //     for (size_t i=0; i<conv_names.size(); i++)
    //     {
    //         std::string layer_name = conv_names[i];
    //         std::string blob_name = conv_bottom_blob_names[layer_name];

    //         ncnn::Mat out;
    //         ex.extract(blob_name.c_str(), out);  

    //         for (size_t j=0; j<quantize_datas.size(); j++)
    //         {
    //             if (quantize_datas[j].name == layer_name)
    //             {
    //                 quantize_datas[j].update_histogram(out);
    //                 break;
    //             }
    //         }
    //     }
    // }

    // step4 kld
    printf("    ====> step 4 : using kld to find the best threshold value.\n");
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
    printf("====> Save the calibration table done.\n");

    return 0;
}

int main(int argc, char** argv)
{
    std::cout << "--- ncnn post training quantization tool --- " << __TIME__ << " " << __DATE__ << std::endl;   

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath] [parampath] [binpath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* parampath = argv[2];
    const char* binpath = argv[3];

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 2;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    ncnn::set_default_option(opt);
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(2);   

    std::vector<std::string> filenames;

    // parse the image file
    readFileList(imagepath, filenames);

    detect_squeezenet(filenames, parampath, binpath);

    return 0;
}
