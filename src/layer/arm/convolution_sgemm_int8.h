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

static inline short saturate2int16(int v)
{
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (short)v;
}

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
}

static void conv_im2col_sgemm_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_size)
{
    const signed char* kernel = _kernel;

    // kernel memory packed 4 x 4
    kernel_tm.create(4*kernel_size, inch, outch/4 + outch%4, (size_t)1u);
    
    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 2;
    remain_outch_start = nn_outch << 2;
    
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        const signed char* k0 = kernel + (p+0)*inch*kernel_size;
        const signed char* k1 = kernel + (p+1)*inch*kernel_size;
        const signed char* k2 = kernel + (p+2)*inch*kernel_size;
        const signed char* k3 = kernel + (p+3)*inch*kernel_size;

        signed char* ktmp = kernel_tm.channel(p/4);

        int q=0;
        for (; q+7<inch*kernel_size; q+=8)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k0[1];
            ktmp[2] = k0[2];
            ktmp[3] = k0[3];
            ktmp[4] = k0[4];
            ktmp[5] = k0[5];
            ktmp[6] = k0[6];
            ktmp[7] = k0[7];

            ktmp[8] = k1[0];
            ktmp[9] = k1[1];
            ktmp[10] = k1[2];
            ktmp[11] = k1[3];
            ktmp[12] = k1[4];
            ktmp[13] = k1[5];
            ktmp[14] = k1[6];
            ktmp[15] = k1[7];

            ktmp[16] = k2[0];
            ktmp[17] = k2[1];
            ktmp[18] = k2[2];
            ktmp[19] = k2[3];
            ktmp[20] = k2[4];
            ktmp[21] = k2[5];
            ktmp[22] = k2[6];
            ktmp[23] = k2[7];

            ktmp[24] = k3[0];
            ktmp[25] = k3[1];
            ktmp[26] = k3[2];
            ktmp[27] = k3[3];
            ktmp[28] = k3[4];
            ktmp[29] = k3[5];
            ktmp[30] = k3[6];
            ktmp[31] = k3[7];

            ktmp += 32;
            k0 += 8;
            k1 += 8;
            k2 += 8;
            k3 += 8;
        }

        for (; q<inch*kernel_size; q++)
        { 
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];

            ktmp += 4;
            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
        }           
    }

    for (int p=remain_outch_start; p<outch; p++)
    {
        const signed char* k0 = kernel + (p+0)*inch*kernel_size;
        signed char* ktmp = kernel_tm.channel(p/4 + p%4);

        int q=0;
        for (; q+7<inch*kernel_size; q=q+8)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k0[1];
            ktmp[2] = k0[2];
            ktmp[3] = k0[3];
            ktmp[4] = k0[4];
            ktmp[5] = k0[5];
            ktmp[6] = k0[6];
            ktmp[7] = k0[7];

            ktmp += 8;
            k0 += 8;
        }

        for (; q<inch*kernel_size; q++)
        {
            ktmp[0] = k0[0];

            ktmp++;
            k0++;
        }
    }
}

static void conv_im2col_sgemm_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, \
            const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // double start = ncnn::get_current_time();

    // im2row
    Mat bottom_im2row(kernel_h*kernel_w*inch, outw*outh, 1UL, opt.workspace_allocator);
    {
        signed char* ret = (signed char*)bottom_im2row;
        int retID = 0;
    
        for (int i=0; i<outh; i++)
        {
            for (int j=0; j<outw; j++)
            {
                for (int p=0; p<inch; p++)
                {
                    const signed char* input = bottom_blob.channel(p);
                    for (int u=0; u<kernel_h; u++)
                    {
                        for (int v=0; v<kernel_w; v++)
                        {    
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }                
                }
            }
        }
    }    

    // double end = ncnn::get_current_time();
    // printf("im2col : %8.3f ms\n", end - start);
    // start = ncnn::get_current_time();    

    // int kernel_size = kernel_w * kernel_h;

    // 4x1
    // sgemm(int M, int N, int K, float* A, float* B, float* C)
    {
        // int M = outch;  // outch
        int N = outw * outh; // outsize or out stride
        int K = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;
        
        for (int pp=0; pp<nn_outch; pp++)
        {
            int i = pp * 4;

            short* output0 = top_blob.channel(i);
            short* output1 = top_blob.channel(i+1);
            short* output2 = top_blob.channel(i+2);
            short* output3 = top_blob.channel(i+3);

            for (int j=0; j<N; j++)
            {
                signed char* vb = bottom_im2row.row<signed char>(j);
                const signed char* va = _kernel.channel(i/4);

                short sum0 = 0;
                short sum1 = 0;
                short sum2 = 0;
                short sum3 = 0;

                short sum0_s16[8] = {0};
                short sum1_s16[8] = {0};
                short sum2_s16[8] = {0};
                short sum3_s16[8] = {0};

                int k = 0;
                for (; k+63<K; k=k+64)
                {
                    short sum0_tmp[8] = {0};
                    short sum1_tmp[8] = {0};
                    short sum2_tmp[8] = {0};
                    short sum3_tmp[8] = {0};

                    // roll 0
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 1
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 2
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 3
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 4
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 5
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 6
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 7
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;                             

                    sum0_s16[0] = saturate2int16((int)(sum0_s16[0]) + sum0_tmp[0]);
                    sum0_s16[1] = saturate2int16((int)(sum0_s16[1]) + sum0_tmp[1]);
                    sum0_s16[2] = saturate2int16((int)(sum0_s16[2]) + sum0_tmp[2]);
                    sum0_s16[3] = saturate2int16((int)(sum0_s16[3]) + sum0_tmp[3]);
                    sum0_s16[4] = saturate2int16((int)(sum0_s16[4]) + sum0_tmp[4]);
                    sum0_s16[5] = saturate2int16((int)(sum0_s16[5]) + sum0_tmp[5]);
                    sum0_s16[6] = saturate2int16((int)(sum0_s16[6]) + sum0_tmp[6]);
                    sum0_s16[7] = saturate2int16((int)(sum0_s16[7]) + sum0_tmp[7]);

                    sum1_s16[0] = saturate2int16((int)(sum1_s16[0]) + sum1_tmp[0]);
                    sum1_s16[1] = saturate2int16((int)(sum1_s16[1]) + sum1_tmp[1]);
                    sum1_s16[2] = saturate2int16((int)(sum1_s16[2]) + sum1_tmp[2]);
                    sum1_s16[3] = saturate2int16((int)(sum1_s16[3]) + sum1_tmp[3]);
                    sum1_s16[4] = saturate2int16((int)(sum1_s16[4]) + sum1_tmp[4]);
                    sum1_s16[5] = saturate2int16((int)(sum1_s16[5]) + sum1_tmp[5]);
                    sum1_s16[6] = saturate2int16((int)(sum1_s16[6]) + sum1_tmp[6]);
                    sum1_s16[7] = saturate2int16((int)(sum1_s16[7]) + sum1_tmp[7]);

                    sum2_s16[0] = saturate2int16((int)(sum2_s16[0]) + sum2_tmp[0]);
                    sum2_s16[1] = saturate2int16((int)(sum2_s16[1]) + sum2_tmp[1]);
                    sum2_s16[2] = saturate2int16((int)(sum2_s16[2]) + sum2_tmp[2]);
                    sum2_s16[3] = saturate2int16((int)(sum2_s16[3]) + sum2_tmp[3]);
                    sum2_s16[4] = saturate2int16((int)(sum2_s16[4]) + sum2_tmp[4]);
                    sum2_s16[5] = saturate2int16((int)(sum2_s16[5]) + sum2_tmp[5]);
                    sum2_s16[6] = saturate2int16((int)(sum2_s16[6]) + sum2_tmp[6]);
                    sum2_s16[7] = saturate2int16((int)(sum2_s16[7]) + sum2_tmp[7]);

                    sum3_s16[0] = saturate2int16((int)(sum3_s16[0]) + sum3_tmp[0]);
                    sum3_s16[1] = saturate2int16((int)(sum3_s16[1]) + sum3_tmp[1]);
                    sum3_s16[2] = saturate2int16((int)(sum3_s16[2]) + sum3_tmp[2]);
                    sum3_s16[3] = saturate2int16((int)(sum3_s16[3]) + sum3_tmp[3]);
                    sum3_s16[4] = saturate2int16((int)(sum3_s16[4]) + sum3_tmp[4]);
                    sum3_s16[5] = saturate2int16((int)(sum3_s16[5]) + sum3_tmp[5]);
                    sum3_s16[6] = saturate2int16((int)(sum3_s16[6]) + sum3_tmp[6]);
                    sum3_s16[7] = saturate2int16((int)(sum3_s16[7]) + sum3_tmp[7]);
                }

                sum0 = saturate2int16((int)sum0 + sum0_s16[0]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[1]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[2]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[3]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[4]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[5]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[6]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[7]);

                sum1 = saturate2int16((int)sum1 + sum1_s16[0]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[1]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[2]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[3]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[4]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[5]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[6]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[7]);

                sum2 = saturate2int16((int)sum2 + sum2_s16[0]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[1]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[2]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[3]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[4]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[5]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[6]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[7]);

                sum3 = saturate2int16((int)sum3 + sum3_s16[0]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[1]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[2]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[3]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[4]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[5]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[6]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[7]);

                for (; k+7<K; k=k+8)
                {
                    short sum_tmp0 = 0;
                    short sum_tmp1 = 0;
                    short sum_tmp2 = 0;
                    short sum_tmp3 = 0;

                    sum_tmp0 = (short)va[0] * vb[0];
                    sum_tmp0 += (short)va[1] * vb[1];
                    sum_tmp0 += (short)va[2] * vb[2];
                    sum_tmp0 += (short)va[3] * vb[3];
                    sum_tmp0 += (short)va[4] * vb[4];
                    sum_tmp0 += (short)va[5] * vb[5];
                    sum_tmp0 += (short)va[6] * vb[6];
                    sum_tmp0 += (short)va[7] * vb[7];
                    va += 8;
                    sum_tmp1 = (short)va[0] * vb[0];
                    sum_tmp1 += (short)va[1] * vb[1];
                    sum_tmp1 += (short)va[2] * vb[2];
                    sum_tmp1 += (short)va[3] * vb[3];
                    sum_tmp1 += (short)va[4] * vb[4];
                    sum_tmp1 += (short)va[5] * vb[5];
                    sum_tmp1 += (short)va[6] * vb[6];
                    sum_tmp1 += (short)va[7] * vb[7];
                    va += 8;
                    sum_tmp2 = (short)va[0] * vb[0];
                    sum_tmp2 += (short)va[1] * vb[1];
                    sum_tmp2 += (short)va[2] * vb[2];
                    sum_tmp2 += (short)va[3] * vb[3];
                    sum_tmp2 += (short)va[4] * vb[4];
                    sum_tmp2 += (short)va[5] * vb[5];
                    sum_tmp2 += (short)va[6] * vb[6];
                    sum_tmp2 += (short)va[7] * vb[7];
                    va += 8;
                    sum_tmp3 = (short)va[0] * vb[0];
                    sum_tmp3 += (short)va[1] * vb[1];
                    sum_tmp3 += (short)va[2] * vb[2];
                    sum_tmp3 += (short)va[3] * vb[3];
                    sum_tmp3 += (short)va[4] * vb[4];
                    sum_tmp3 += (short)va[5] * vb[5];
                    sum_tmp3 += (short)va[6] * vb[6];
                    sum_tmp3 += (short)va[7] * vb[7];

                    va += 8;
                    vb += 8;
                    sum0 = saturate2int16((int)(sum0) + sum_tmp0);
                    sum1 = saturate2int16((int)(sum1) + sum_tmp1);
                    sum2 = saturate2int16((int)(sum2) + sum_tmp2);
                    sum3 = saturate2int16((int)(sum3) + sum_tmp3);
                }                

                for (; k<K; k++)
                {
                    int sum_tmp0 = 0;
                    int sum_tmp1 = 0;
                    int sum_tmp2 = 0;
                    int sum_tmp3 = 0;

                    sum_tmp0 += (int)va[0] * vb[0];
                    sum_tmp1 += (int)va[1] * vb[0];
                    sum_tmp2 += (int)va[2] * vb[0];
                    sum_tmp3 += (int)va[3] * vb[0];

                    sum0 = saturate2int16((int)(sum0) + sum_tmp0);
                    sum1 = saturate2int16((int)(sum1) + sum_tmp1);
                    sum2 = saturate2int16((int)(sum2) + sum_tmp2);
                    sum3 = saturate2int16((int)(sum3) + sum_tmp3);

                    va += 4;
                    vb += 1;
                }

                // dequant convert int32 to fp32
                output0[0] = sum0;
                output1[0] = sum1;
                output2[0] = sum2;
                output3[0] = sum3;

                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        for (int i=remain_outch_start; i<outch; i++)
        {
            short* output = top_blob.channel(i);

            for (int j=0; j<N; j++)
            {
                signed char* vb = bottom_im2row.row<signed char>(j);
                const signed char* va = _kernel.channel(i/4 + i%4);

                short sum = 0;
                short sum_s16[8] = {0};

                int k = 0;
                for (; k+63<K; k=k+64)
                {
                    short sum_tmp[8] = {0};
                    // roll 0
                    sum_tmp[0] = (short)va[0] * vb[0];
                    sum_tmp[1] = (short)va[1] * vb[1];
                    sum_tmp[2] = (short)va[2] * vb[2];
                    sum_tmp[3] = (short)va[3] * vb[3];
                    sum_tmp[4] = (short)va[4] * vb[4];
                    sum_tmp[5] = (short)va[5] * vb[5];
                    sum_tmp[6] = (short)va[6] * vb[6];
                    sum_tmp[7] = (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 1
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 2
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 3
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 4
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 5
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 6
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 7
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    sum_s16[0] = saturate2int16((int)(sum_s16[0]) + sum_tmp[0]);
                    sum_s16[1] = saturate2int16((int)(sum_s16[1]) + sum_tmp[1]);
                    sum_s16[2] = saturate2int16((int)(sum_s16[2]) + sum_tmp[2]);
                    sum_s16[3] = saturate2int16((int)(sum_s16[3]) + sum_tmp[3]);
                    sum_s16[4] = saturate2int16((int)(sum_s16[4]) + sum_tmp[4]);
                    sum_s16[5] = saturate2int16((int)(sum_s16[5]) + sum_tmp[5]);
                    sum_s16[6] = saturate2int16((int)(sum_s16[6]) + sum_tmp[6]);
                    sum_s16[7] = saturate2int16((int)(sum_s16[7]) + sum_tmp[7]);
                }

                sum = saturate2int16((int)sum + sum_s16[0]);
                sum = saturate2int16((int)sum + sum_s16[1]);
                sum = saturate2int16((int)sum + sum_s16[2]);
                sum = saturate2int16((int)sum + sum_s16[3]);
                sum = saturate2int16((int)sum + sum_s16[4]);
                sum = saturate2int16((int)sum + sum_s16[5]);
                sum = saturate2int16((int)sum + sum_s16[6]);
                sum = saturate2int16((int)sum + sum_s16[7]);                

                for (; k<K; k++)
                {
                    short sum_tmp = 0;
                    sum_tmp += (short)va[0] * vb[0];

                    sum = saturate2int16((int)(sum) + sum_tmp);

                    va += 1;
                    vb += 1;
                }

                // dequant convert int32 to fp32
                output[0] = sum;
                output++;
            }
        }
    } 

    // end = ncnn::get_current_time();
    // printf("sgemm  : %8.3f ms\n", end - start);    
}

static void conv_im2col_sgemm_int8_dequant_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, \
            const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Mat &_bias, std::vector<float> scale_dequant, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // const signed char *kernel = _kernel;
    const float* bias = _bias;

    // double start = ncnn::get_current_time();

    // im2row
    Mat bottom_im2row(kernel_h*kernel_w*inch, outw*outh, 1UL, opt.workspace_allocator);
    {
        signed char* ret = (signed char*)bottom_im2row;
        int retID = 0;
    
        for (int i=0; i<outh; i++)
        {
            for (int j=0; j<outw; j++)
            {
                for (int p=0; p<inch; p++)
                {
                    const signed char* input = bottom_blob.channel(p);
                    for (int u=0; u<kernel_h; u++)
                    {
                        for (int v=0; v<kernel_w; v++)
                        {    
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }                
                }
            }
        }
    }    

    // double end = ncnn::get_current_time();
    // printf("im2col : %8.3f ms\n", end - start);
    // start = ncnn::get_current_time();    

    // int kernel_size = kernel_w * kernel_h;

    // 4x1
    // sgemm(int M, int N, int K, float* A, float* B, float* C)
    {
        // int M = outch;  // outch
        int N = outw * outh; // outsize or out stride
        int K = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;
        
        for (int pp=0; pp<nn_outch; pp++)
        {
            int i = pp * 4;

            const float bias0 = bias ? bias[i]   : 0.f;
            const float bias1 = bias ? bias[i+1] : 0.f;
            const float bias2 = bias ? bias[i+2] : 0.f;
            const float bias3 = bias ? bias[i+3] : 0.f;

            const float scale_dequant0 = scale_dequant[i];
            const float scale_dequant1 = scale_dequant[i+1];
            const float scale_dequant2 = scale_dequant[i+2];
            const float scale_dequant3 = scale_dequant[i+3];

            float* output0 = top_blob.channel(i);
            float* output1 = top_blob.channel(i+1);
            float* output2 = top_blob.channel(i+2);
            float* output3 = top_blob.channel(i+3);

            for (int j=0; j<N; j++)
            {
                signed char* vb = bottom_im2row.row<signed char>(j);
                const signed char* va = _kernel.channel(i/4);

                short sum0 = 0;
                short sum1 = 0;
                short sum2 = 0;
                short sum3 = 0;

                short sum0_s16[8] = {0};
                short sum1_s16[8] = {0};
                short sum2_s16[8] = {0};
                short sum3_s16[8] = {0};

                int k = 0;
                for (; k+63<K; k=k+64)
                {
                    short sum0_tmp[8] = {0};
                    short sum1_tmp[8] = {0};
                    short sum2_tmp[8] = {0};
                    short sum3_tmp[8] = {0};

                    // roll 0
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 1
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 2
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 3
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 4
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 5
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 6
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 7
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;                             

                    sum0_s16[0] = saturate2int16((int)(sum0_s16[0]) + sum0_tmp[0]);
                    sum0_s16[1] = saturate2int16((int)(sum0_s16[1]) + sum0_tmp[1]);
                    sum0_s16[2] = saturate2int16((int)(sum0_s16[2]) + sum0_tmp[2]);
                    sum0_s16[3] = saturate2int16((int)(sum0_s16[3]) + sum0_tmp[3]);
                    sum0_s16[4] = saturate2int16((int)(sum0_s16[4]) + sum0_tmp[4]);
                    sum0_s16[5] = saturate2int16((int)(sum0_s16[5]) + sum0_tmp[5]);
                    sum0_s16[6] = saturate2int16((int)(sum0_s16[6]) + sum0_tmp[6]);
                    sum0_s16[7] = saturate2int16((int)(sum0_s16[7]) + sum0_tmp[7]);

                    sum1_s16[0] = saturate2int16((int)(sum1_s16[0]) + sum1_tmp[0]);
                    sum1_s16[1] = saturate2int16((int)(sum1_s16[1]) + sum1_tmp[1]);
                    sum1_s16[2] = saturate2int16((int)(sum1_s16[2]) + sum1_tmp[2]);
                    sum1_s16[3] = saturate2int16((int)(sum1_s16[3]) + sum1_tmp[3]);
                    sum1_s16[4] = saturate2int16((int)(sum1_s16[4]) + sum1_tmp[4]);
                    sum1_s16[5] = saturate2int16((int)(sum1_s16[5]) + sum1_tmp[5]);
                    sum1_s16[6] = saturate2int16((int)(sum1_s16[6]) + sum1_tmp[6]);
                    sum1_s16[7] = saturate2int16((int)(sum1_s16[7]) + sum1_tmp[7]);

                    sum2_s16[0] = saturate2int16((int)(sum2_s16[0]) + sum2_tmp[0]);
                    sum2_s16[1] = saturate2int16((int)(sum2_s16[1]) + sum2_tmp[1]);
                    sum2_s16[2] = saturate2int16((int)(sum2_s16[2]) + sum2_tmp[2]);
                    sum2_s16[3] = saturate2int16((int)(sum2_s16[3]) + sum2_tmp[3]);
                    sum2_s16[4] = saturate2int16((int)(sum2_s16[4]) + sum2_tmp[4]);
                    sum2_s16[5] = saturate2int16((int)(sum2_s16[5]) + sum2_tmp[5]);
                    sum2_s16[6] = saturate2int16((int)(sum2_s16[6]) + sum2_tmp[6]);
                    sum2_s16[7] = saturate2int16((int)(sum2_s16[7]) + sum2_tmp[7]);

                    sum3_s16[0] = saturate2int16((int)(sum3_s16[0]) + sum3_tmp[0]);
                    sum3_s16[1] = saturate2int16((int)(sum3_s16[1]) + sum3_tmp[1]);
                    sum3_s16[2] = saturate2int16((int)(sum3_s16[2]) + sum3_tmp[2]);
                    sum3_s16[3] = saturate2int16((int)(sum3_s16[3]) + sum3_tmp[3]);
                    sum3_s16[4] = saturate2int16((int)(sum3_s16[4]) + sum3_tmp[4]);
                    sum3_s16[5] = saturate2int16((int)(sum3_s16[5]) + sum3_tmp[5]);
                    sum3_s16[6] = saturate2int16((int)(sum3_s16[6]) + sum3_tmp[6]);
                    sum3_s16[7] = saturate2int16((int)(sum3_s16[7]) + sum3_tmp[7]);
                }

                sum0 = saturate2int16((int)sum0 + sum0_s16[0]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[1]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[2]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[3]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[4]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[5]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[6]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[7]);

                sum1 = saturate2int16((int)sum1 + sum1_s16[0]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[1]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[2]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[3]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[4]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[5]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[6]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[7]);

                sum2 = saturate2int16((int)sum2 + sum2_s16[0]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[1]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[2]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[3]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[4]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[5]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[6]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[7]);

                sum3 = saturate2int16((int)sum3 + sum3_s16[0]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[1]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[2]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[3]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[4]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[5]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[6]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[7]);

                for (; k+7<K; k=k+8)
                {
                    short sum_tmp0 = 0;
                    short sum_tmp1 = 0;
                    short sum_tmp2 = 0;
                    short sum_tmp3 = 0;

                    sum_tmp0 = (short)va[0] * vb[0];
                    sum_tmp0 += (short)va[1] * vb[1];
                    sum_tmp0 += (short)va[2] * vb[2];
                    sum_tmp0 += (short)va[3] * vb[3];
                    sum_tmp0 += (short)va[4] * vb[4];
                    sum_tmp0 += (short)va[5] * vb[5];
                    sum_tmp0 += (short)va[6] * vb[6];
                    sum_tmp0 += (short)va[7] * vb[7];
                    va += 8;
                    sum_tmp1 = (short)va[0] * vb[0];
                    sum_tmp1 += (short)va[1] * vb[1];
                    sum_tmp1 += (short)va[2] * vb[2];
                    sum_tmp1 += (short)va[3] * vb[3];
                    sum_tmp1 += (short)va[4] * vb[4];
                    sum_tmp1 += (short)va[5] * vb[5];
                    sum_tmp1 += (short)va[6] * vb[6];
                    sum_tmp1 += (short)va[7] * vb[7];
                    va += 8;
                    sum_tmp2 = (short)va[0] * vb[0];
                    sum_tmp2 += (short)va[1] * vb[1];
                    sum_tmp2 += (short)va[2] * vb[2];
                    sum_tmp2 += (short)va[3] * vb[3];
                    sum_tmp2 += (short)va[4] * vb[4];
                    sum_tmp2 += (short)va[5] * vb[5];
                    sum_tmp2 += (short)va[6] * vb[6];
                    sum_tmp2 += (short)va[7] * vb[7];
                    va += 8;
                    sum_tmp3 = (short)va[0] * vb[0];
                    sum_tmp3 += (short)va[1] * vb[1];
                    sum_tmp3 += (short)va[2] * vb[2];
                    sum_tmp3 += (short)va[3] * vb[3];
                    sum_tmp3 += (short)va[4] * vb[4];
                    sum_tmp3 += (short)va[5] * vb[5];
                    sum_tmp3 += (short)va[6] * vb[6];
                    sum_tmp3 += (short)va[7] * vb[7];

                    va += 8;
                    vb += 8;
                    sum0 = saturate2int16((int)(sum0) + sum_tmp0);
                    sum1 = saturate2int16((int)(sum1) + sum_tmp1);
                    sum2 = saturate2int16((int)(sum2) + sum_tmp2);
                    sum3 = saturate2int16((int)(sum3) + sum_tmp3);
                }                

                for (; k<K; k++)
                {
                    int sum_tmp0 = 0;
                    int sum_tmp1 = 0;
                    int sum_tmp2 = 0;
                    int sum_tmp3 = 0;

                    sum_tmp0 += (int)va[0] * vb[0];
                    sum_tmp1 += (int)va[1] * vb[0];
                    sum_tmp2 += (int)va[2] * vb[0];
                    sum_tmp3 += (int)va[3] * vb[0];

                    sum0 = saturate2int16((int)(sum0) + sum_tmp0);
                    sum1 = saturate2int16((int)(sum1) + sum_tmp1);
                    sum2 = saturate2int16((int)(sum2) + sum_tmp2);
                    sum3 = saturate2int16((int)(sum3) + sum_tmp3);

                    va += 4;
                    vb += 1;
                }

                // dequant convert int32 to fp32
                output0[0] = (float)sum0 * scale_dequant0 + bias0;
                output1[0] = (float)sum1 * scale_dequant1 + bias1;
                output2[0] = (float)sum2 * scale_dequant2 + bias2;
                output3[0] = (float)sum3 * scale_dequant3 + bias3;

                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        for (int i=remain_outch_start; i<outch; i++)
        {
            float* output = top_blob.channel(i);

            const float bias0 = bias ? bias[i] : 0.f;
            const float scale_dequant0 = scale_dequant[i];

            for (int j=0; j<N; j++)
            {
                signed char* vb = bottom_im2row.row<signed char>(j);
                const signed char* va = _kernel.channel(i/4 + i%4);

                short sum = 0;
                short sum_s16[8] = {0};

                int k = 0;
                for (; k+63<K; k=k+64)
                {
                    short sum_tmp[8] = {0};
                    // roll 0
                    sum_tmp[0] = (short)va[0] * vb[0];
                    sum_tmp[1] = (short)va[1] * vb[1];
                    sum_tmp[2] = (short)va[2] * vb[2];
                    sum_tmp[3] = (short)va[3] * vb[3];
                    sum_tmp[4] = (short)va[4] * vb[4];
                    sum_tmp[5] = (short)va[5] * vb[5];
                    sum_tmp[6] = (short)va[6] * vb[6];
                    sum_tmp[7] = (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 1
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 2
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 3
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 4
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 5
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 6
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 7
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    sum_s16[0] = saturate2int16((int)(sum_s16[0]) + sum_tmp[0]);
                    sum_s16[1] = saturate2int16((int)(sum_s16[1]) + sum_tmp[1]);
                    sum_s16[2] = saturate2int16((int)(sum_s16[2]) + sum_tmp[2]);
                    sum_s16[3] = saturate2int16((int)(sum_s16[3]) + sum_tmp[3]);
                    sum_s16[4] = saturate2int16((int)(sum_s16[4]) + sum_tmp[4]);
                    sum_s16[5] = saturate2int16((int)(sum_s16[5]) + sum_tmp[5]);
                    sum_s16[6] = saturate2int16((int)(sum_s16[6]) + sum_tmp[6]);
                    sum_s16[7] = saturate2int16((int)(sum_s16[7]) + sum_tmp[7]);
                }

                sum = saturate2int16((int)sum + sum_s16[0]);
                sum = saturate2int16((int)sum + sum_s16[1]);
                sum = saturate2int16((int)sum + sum_s16[2]);
                sum = saturate2int16((int)sum + sum_s16[3]);
                sum = saturate2int16((int)sum + sum_s16[4]);
                sum = saturate2int16((int)sum + sum_s16[5]);
                sum = saturate2int16((int)sum + sum_s16[6]);
                sum = saturate2int16((int)sum + sum_s16[7]);                

                for (; k<K; k++)
                {
                    short sum_tmp = 0;
                    sum_tmp += (short)va[0] * vb[0];

                    sum = saturate2int16((int)(sum) + sum_tmp);

                    va += 1;
                    vb += 1;
                }

                // dequant convert int32 to fp32
                output[0] = (float)sum * scale_dequant0 + bias0;
                output++;
            }
        }
    } 

    // end = ncnn::get_current_time();
    // printf("sgemm  : %8.3f ms\n", end - start);    
}

static void conv_im2col_sgemm_int8_requant_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, \
            const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Mat &_bias, std::vector<float> scale_requant, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // const signed char *kernel = _kernel;
    const float* bias = _bias;

    // double start = ncnn::get_current_time();

    // im2row
    Mat bottom_im2row(kernel_h*kernel_w*inch, outw*outh, 1UL, opt.workspace_allocator);
    {
        signed char* ret = (signed char*)bottom_im2row;
        int retID = 0;
    
        for (int i=0; i<outh; i++)
        {
            for (int j=0; j<outw; j++)
            {
                for (int p=0; p<inch; p++)
                {
                    const signed char* input = bottom_blob.channel(p);
                    for (int u=0; u<kernel_h; u++)
                    {
                        for (int v=0; v<kernel_w; v++)
                        {    
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }                
                }
            }
        }
    }    

    // double end = ncnn::get_current_time();
    // printf("im2col : %8.3f ms\n", end - start);
    // start = ncnn::get_current_time();    

    // int kernel_size = kernel_w * kernel_h;

    // 4x1
    // sgemm(int M, int N, int K, float* A, float* B, float* C)
    {
        // int M = outch;  // outch
        int N = outw * outh; // outsize or out stride
        int K = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;
        
        for (int pp=0; pp<nn_outch; pp++)
        {
            int i = pp * 4;

            const float bias0 = bias ? bias[i]   : 0.f;
            const float bias1 = bias ? bias[i+1] : 0.f;
            const float bias2 = bias ? bias[i+2] : 0.f;
            const float bias3 = bias ? bias[i+3] : 0.f;

            const float scale_requant_in0  = scale_requant[2*i];
            const float scale_requant_out0 = scale_requant[2*i+1];
            const float scale_requant_in1  = scale_requant[2*(i+1)];
            const float scale_requant_out1 = scale_requant[2*(i+1)+1];
            const float scale_requant_in2  = scale_requant[2*(i+2)];
            const float scale_requant_out2 = scale_requant[2*(i+2)+1];
            const float scale_requant_in3  = scale_requant[2*(i+3)];
            const float scale_requant_out3 = scale_requant[2*(i+3)+1];

            signed char* output0 = top_blob.channel(i);
            signed char* output1 = top_blob.channel(i+1);
            signed char* output2 = top_blob.channel(i+2);
            signed char* output3 = top_blob.channel(i+3);

            for (int j=0; j<N; j++)
            {
                signed char* vb = bottom_im2row.row<signed char>(j);
                const signed char* va = _kernel.channel(i/4);

                short sum0 = 0;
                short sum1 = 0;
                short sum2 = 0;
                short sum3 = 0;

                short sum0_s16[8] = {0};
                short sum1_s16[8] = {0};
                short sum2_s16[8] = {0};
                short sum3_s16[8] = {0};

                int k = 0;
                for (; k+63<K; k=k+64)
                {
                    short sum0_tmp[8] = {0};
                    short sum1_tmp[8] = {0};
                    short sum2_tmp[8] = {0};
                    short sum3_tmp[8] = {0};

                    // roll 0
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 1
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 2
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 3
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 4
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 5
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 6
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    // roll 7
                    sum0_tmp[0] += (short)va[0] * vb[0];
                    sum0_tmp[1] += (short)va[1] * vb[1];
                    sum0_tmp[2] += (short)va[2] * vb[2];
                    sum0_tmp[3] += (short)va[3] * vb[3];
                    sum0_tmp[4] += (short)va[4] * vb[4];
                    sum0_tmp[5] += (short)va[5] * vb[5];
                    sum0_tmp[6] += (short)va[6] * vb[6];
                    sum0_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum1_tmp[0] += (short)va[0] * vb[0];
                    sum1_tmp[1] += (short)va[1] * vb[1];
                    sum1_tmp[2] += (short)va[2] * vb[2];
                    sum1_tmp[3] += (short)va[3] * vb[3];
                    sum1_tmp[4] += (short)va[4] * vb[4];
                    sum1_tmp[5] += (short)va[5] * vb[5];
                    sum1_tmp[6] += (short)va[6] * vb[6];
                    sum1_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum2_tmp[0] += (short)va[0] * vb[0];
                    sum2_tmp[1] += (short)va[1] * vb[1];
                    sum2_tmp[2] += (short)va[2] * vb[2];
                    sum2_tmp[3] += (short)va[3] * vb[3];
                    sum2_tmp[4] += (short)va[4] * vb[4];
                    sum2_tmp[5] += (short)va[5] * vb[5];
                    sum2_tmp[6] += (short)va[6] * vb[6];
                    sum2_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    sum3_tmp[0] += (short)va[0] * vb[0];
                    sum3_tmp[1] += (short)va[1] * vb[1];
                    sum3_tmp[2] += (short)va[2] * vb[2];
                    sum3_tmp[3] += (short)va[3] * vb[3];
                    sum3_tmp[4] += (short)va[4] * vb[4];
                    sum3_tmp[5] += (short)va[5] * vb[5];
                    sum3_tmp[6] += (short)va[6] * vb[6];
                    sum3_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;                              

                    sum0_s16[0] = saturate2int16((int)(sum0_s16[0]) + sum0_tmp[0]);
                    sum0_s16[1] = saturate2int16((int)(sum0_s16[1]) + sum0_tmp[1]);
                    sum0_s16[2] = saturate2int16((int)(sum0_s16[2]) + sum0_tmp[2]);
                    sum0_s16[3] = saturate2int16((int)(sum0_s16[3]) + sum0_tmp[3]);
                    sum0_s16[4] = saturate2int16((int)(sum0_s16[4]) + sum0_tmp[4]);
                    sum0_s16[5] = saturate2int16((int)(sum0_s16[5]) + sum0_tmp[5]);
                    sum0_s16[6] = saturate2int16((int)(sum0_s16[6]) + sum0_tmp[6]);
                    sum0_s16[7] = saturate2int16((int)(sum0_s16[7]) + sum0_tmp[7]);

                    sum1_s16[0] = saturate2int16((int)(sum1_s16[0]) + sum1_tmp[0]);
                    sum1_s16[1] = saturate2int16((int)(sum1_s16[1]) + sum1_tmp[1]);
                    sum1_s16[2] = saturate2int16((int)(sum1_s16[2]) + sum1_tmp[2]);
                    sum1_s16[3] = saturate2int16((int)(sum1_s16[3]) + sum1_tmp[3]);
                    sum1_s16[4] = saturate2int16((int)(sum1_s16[4]) + sum1_tmp[4]);
                    sum1_s16[5] = saturate2int16((int)(sum1_s16[5]) + sum1_tmp[5]);
                    sum1_s16[6] = saturate2int16((int)(sum1_s16[6]) + sum1_tmp[6]);
                    sum1_s16[7] = saturate2int16((int)(sum1_s16[7]) + sum1_tmp[7]);

                    sum2_s16[0] = saturate2int16((int)(sum2_s16[0]) + sum2_tmp[0]);
                    sum2_s16[1] = saturate2int16((int)(sum2_s16[1]) + sum2_tmp[1]);
                    sum2_s16[2] = saturate2int16((int)(sum2_s16[2]) + sum2_tmp[2]);
                    sum2_s16[3] = saturate2int16((int)(sum2_s16[3]) + sum2_tmp[3]);
                    sum2_s16[4] = saturate2int16((int)(sum2_s16[4]) + sum2_tmp[4]);
                    sum2_s16[5] = saturate2int16((int)(sum2_s16[5]) + sum2_tmp[5]);
                    sum2_s16[6] = saturate2int16((int)(sum2_s16[6]) + sum2_tmp[6]);
                    sum2_s16[7] = saturate2int16((int)(sum2_s16[7]) + sum2_tmp[7]);

                    sum3_s16[0] = saturate2int16((int)(sum3_s16[0]) + sum3_tmp[0]);
                    sum3_s16[1] = saturate2int16((int)(sum3_s16[1]) + sum3_tmp[1]);
                    sum3_s16[2] = saturate2int16((int)(sum3_s16[2]) + sum3_tmp[2]);
                    sum3_s16[3] = saturate2int16((int)(sum3_s16[3]) + sum3_tmp[3]);
                    sum3_s16[4] = saturate2int16((int)(sum3_s16[4]) + sum3_tmp[4]);
                    sum3_s16[5] = saturate2int16((int)(sum3_s16[5]) + sum3_tmp[5]);
                    sum3_s16[6] = saturate2int16((int)(sum3_s16[6]) + sum3_tmp[6]);
                    sum3_s16[7] = saturate2int16((int)(sum3_s16[7]) + sum3_tmp[7]);
                }

                sum0 = saturate2int16((int)sum0 + sum0_s16[0]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[1]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[2]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[3]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[4]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[5]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[6]);
                sum0 = saturate2int16((int)sum0 + sum0_s16[7]);

                sum1 = saturate2int16((int)sum1 + sum1_s16[0]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[1]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[2]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[3]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[4]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[5]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[6]);
                sum1 = saturate2int16((int)sum1 + sum1_s16[7]);

                sum2 = saturate2int16((int)sum2 + sum2_s16[0]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[1]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[2]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[3]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[4]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[5]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[6]);
                sum2 = saturate2int16((int)sum2 + sum2_s16[7]);

                sum3 = saturate2int16((int)sum3 + sum3_s16[0]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[1]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[2]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[3]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[4]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[5]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[6]);
                sum3 = saturate2int16((int)sum3 + sum3_s16[7]);

                for (; k+7<K; k=k+8)
                {
                    short sum_tmp0 = 0;
                    short sum_tmp1 = 0;
                    short sum_tmp2 = 0;
                    short sum_tmp3 = 0;

                    sum_tmp0 = (short)va[0] * vb[0];
                    sum_tmp0 += (short)va[1] * vb[1];
                    sum_tmp0 += (short)va[2] * vb[2];
                    sum_tmp0 += (short)va[3] * vb[3];
                    sum_tmp0 += (short)va[4] * vb[4];
                    sum_tmp0 += (short)va[5] * vb[5];
                    sum_tmp0 += (short)va[6] * vb[6];
                    sum_tmp0 += (short)va[7] * vb[7];
                    va += 8;
                    sum_tmp1 = (short)va[0] * vb[0];
                    sum_tmp1 += (short)va[1] * vb[1];
                    sum_tmp1 += (short)va[2] * vb[2];
                    sum_tmp1 += (short)va[3] * vb[3];
                    sum_tmp1 += (short)va[4] * vb[4];
                    sum_tmp1 += (short)va[5] * vb[5];
                    sum_tmp1 += (short)va[6] * vb[6];
                    sum_tmp1 += (short)va[7] * vb[7];
                    va += 8;
                    sum_tmp2 = (short)va[0] * vb[0];
                    sum_tmp2 += (short)va[1] * vb[1];
                    sum_tmp2 += (short)va[2] * vb[2];
                    sum_tmp2 += (short)va[3] * vb[3];
                    sum_tmp2 += (short)va[4] * vb[4];
                    sum_tmp2 += (short)va[5] * vb[5];
                    sum_tmp2 += (short)va[6] * vb[6];
                    sum_tmp2 += (short)va[7] * vb[7];
                    va += 8;
                    sum_tmp3 = (short)va[0] * vb[0];
                    sum_tmp3 += (short)va[1] * vb[1];
                    sum_tmp3 += (short)va[2] * vb[2];
                    sum_tmp3 += (short)va[3] * vb[3];
                    sum_tmp3 += (short)va[4] * vb[4];
                    sum_tmp3 += (short)va[5] * vb[5];
                    sum_tmp3 += (short)va[6] * vb[6];
                    sum_tmp3 += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    sum0 = saturate2int16((int)(sum0) + sum_tmp0);
                    sum1 = saturate2int16((int)(sum1) + sum_tmp1);
                    sum2 = saturate2int16((int)(sum2) + sum_tmp2);
                    sum3 = saturate2int16((int)(sum3) + sum_tmp3);
                }                

                for (; k<K; k++)
                {
                    int sum_tmp0 = 0;
                    int sum_tmp1 = 0;
                    int sum_tmp2 = 0;
                    int sum_tmp3 = 0;

                    sum_tmp0 += (int)va[0] * vb[0];
                    sum_tmp1 += (int)va[1] * vb[0];
                    sum_tmp2 += (int)va[2] * vb[0];
                    sum_tmp3 += (int)va[3] * vb[0];

                    sum0 = saturate2int16((int)(sum0) + sum_tmp0);
                    sum1 = saturate2int16((int)(sum1) + sum_tmp1);
                    sum2 = saturate2int16((int)(sum2) + sum_tmp2);
                    sum3 = saturate2int16((int)(sum3) + sum_tmp3);

                    va += 4;
                    vb += 1;
                }

                // dequant convert int32 to fp32
                output0[0] = float2int8(((float)sum0 * scale_requant_in0 + bias0) * scale_requant_out0);
                output1[0] = float2int8(((float)sum1 * scale_requant_in1 + bias1) * scale_requant_out1);
                output2[0] = float2int8(((float)sum2 * scale_requant_in2 + bias2) * scale_requant_out2);
                output3[0] = float2int8(((float)sum3 * scale_requant_in3 + bias3) * scale_requant_out3);

                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        for (int i=remain_outch_start; i<outch; i++)
        {
            signed char* output = top_blob.channel(i);

            const float bias0 = bias ? bias[i] : 0.f;

            const float scale_requant_in0  = scale_requant[2*i];
            const float scale_requant_out0 = scale_requant[2*i+1]; 

            for (int j=0; j<N; j++)
            {
                signed char* vb = bottom_im2row.row<signed char>(j);
                const signed char* va = _kernel.channel(i/4 + i%4);

                short sum = 0;
                short sum_s16[8] = {0};

                int k = 0;
                for (; k+63<K; k=k+64)
                {
                    short sum_tmp[8] = {0};
                    // roll 0
                    sum_tmp[0] = (short)va[0] * vb[0];
                    sum_tmp[1] = (short)va[1] * vb[1];
                    sum_tmp[2] = (short)va[2] * vb[2];
                    sum_tmp[3] = (short)va[3] * vb[3];
                    sum_tmp[4] = (short)va[4] * vb[4];
                    sum_tmp[5] = (short)va[5] * vb[5];
                    sum_tmp[6] = (short)va[6] * vb[6];
                    sum_tmp[7] = (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 1
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 2
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 3
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 4
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 5
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 6
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;
                    // roll 7
                    sum_tmp[0] += (short)va[0] * vb[0];
                    sum_tmp[1] += (short)va[1] * vb[1];
                    sum_tmp[2] += (short)va[2] * vb[2];
                    sum_tmp[3] += (short)va[3] * vb[3];
                    sum_tmp[4] += (short)va[4] * vb[4];
                    sum_tmp[5] += (short)va[5] * vb[5];
                    sum_tmp[6] += (short)va[6] * vb[6];
                    sum_tmp[7] += (short)va[7] * vb[7];
                    va += 8;
                    vb += 8;

                    sum_s16[0] = saturate2int16((int)(sum_s16[0]) + sum_tmp[0]);
                    sum_s16[1] = saturate2int16((int)(sum_s16[1]) + sum_tmp[1]);
                    sum_s16[2] = saturate2int16((int)(sum_s16[2]) + sum_tmp[2]);
                    sum_s16[3] = saturate2int16((int)(sum_s16[3]) + sum_tmp[3]);
                    sum_s16[4] = saturate2int16((int)(sum_s16[4]) + sum_tmp[4]);
                    sum_s16[5] = saturate2int16((int)(sum_s16[5]) + sum_tmp[5]);
                    sum_s16[6] = saturate2int16((int)(sum_s16[6]) + sum_tmp[6]);
                    sum_s16[7] = saturate2int16((int)(sum_s16[7]) + sum_tmp[7]);
                }

                sum = saturate2int16((int)sum + sum_s16[0]);
                sum = saturate2int16((int)sum + sum_s16[1]);
                sum = saturate2int16((int)sum + sum_s16[2]);
                sum = saturate2int16((int)sum + sum_s16[3]);
                sum = saturate2int16((int)sum + sum_s16[4]);
                sum = saturate2int16((int)sum + sum_s16[5]);
                sum = saturate2int16((int)sum + sum_s16[6]);
                sum = saturate2int16((int)sum + sum_s16[7]);                

                for (; k<K; k++)
                {
                    short sum_tmp = 0;
                    sum_tmp += (short)va[0] * vb[0];

                    sum = saturate2int16((int)(sum) + sum_tmp);

                    va += 1;
                    vb += 1;
                }

                // dequant convert int32 to fp32
                output[0] = float2int8(((float)sum * scale_requant_in0 + bias0) * scale_requant_out0);
                output++;
            }
        }
    } 

    // end = ncnn::get_current_time();
    // printf("sgemm  : %8.3f ms\n", end - start);    
}