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

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static void pooling2x2s2_max_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    
    const int tailstep = w - 2*outw + w;
    
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<inch; q++)
    {
        const signed char* img0 = bottom_blob.channel(q);
        signed char* outptr = top_blob.channel(q);

        const signed char* r0 = img0;
        const signed char* r1 = img0 + w;

        for (int i = 0; i < outh; i++)
        {
            int remain = outw;

            for (; remain>0; remain--)
            {
                signed char max0 = std::max(r0[0], r0[1]);
                signed char max1 = std::max(r1[0], r1[1]);

                *outptr = std::max(max0, max1);

                r0 += 2;
                r1 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}