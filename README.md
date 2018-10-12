![](https://raw.githubusercontent.com/Tencent/ncnn/master/images/256-ncnn.png)
# ncnn

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://raw.githubusercontent.com/Tencent/ncnn/master/LICENSE.txt) 
[![Build Status](https://travis-ci.org/Tencent/ncnn.svg?branch=master)](https://travis-ci.org/Tencent/ncnn)
[![Coverage Status](https://coveralls.io/repos/github/Tencent/ncnn/badge.svg?branch=master)](https://coveralls.io/github/Tencent/ncnn?branch=master)


ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. ncnn is deeply considerate about deployment and uses on mobile phones from the beginning of design. ncnn does not have third party dependencies. it is cross-platform, and runs faster than all known open source frameworks on mobile phone cpu. Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient ncnn implementation, create intelligent APPs, and bring the artificial intelligence to your fingertips. ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

ncnn 是一个为手机端极致优化的高性能神经网络前向计算框架。ncnn 从设计之初深刻考虑手机端的部署和使用。无第三方依赖，跨平台，手机端 cpu 的速度快于目前所有已知的开源框架。基于 ncnn，开发者能够将深度学习算法轻松移植到手机端高效执行，开发出人工智能 APP，将 AI 带到你的指尖。ncnn 目前已在腾讯多款应用中使用，如 QQ，Qzone，微信，天天P图等。

---

### HowTo

[how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build)

[how to use ncnn with alexnet](https://github.com/Tencent/ncnn/wiki/how-to-use-ncnn-with-alexnet)

[ncnn 组件使用指北 alexnet](https://github.com/Tencent/ncnn/wiki/ncnn-%E7%BB%84%E4%BB%B6%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97-alexnet)

[ncnn low-level operation api](https://github.com/Tencent/ncnn/wiki/low-level-operation-api)

[ncnn param and model file spec](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure)

[ncnn operation param weight table](https://github.com/Tencent/ncnn/wiki/operation-param-weight-table)

[how to implement custom layer step by step](https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step)

---

### FAQ

[ncnn throw error](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-throw-error)

[ncnn produce wrong result](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result)

---

### Features

* Supports convolutional neural networks, supports multiple input and multi-branch structure, can calculate part of the branch
* No third-party library dependencies, does not rely on BLAS / NNPACK or any other computing framework
* Pure C ++ implementation, cross-platform, supports android, ios and so on
* ARM NEON assembly level of careful optimization, calculation speed is extremely high
* Sophisticated memory management and data structure design, very low memory footprint
* Supports multi-core parallel computing acceleration, ARM big.LITTLE cpu scheduling optimization
* The overall library size is less than 500K, and can be easily reduced to less than 300K
* Extensible model design, supports 8bit quantization and half-precision floating point storage, can import caffe/pytorch/mxnet/onnx models
* Support direct memory zero copy reference load network model
* Can be registered with custom layer implementation and extended
* Well, it is strong, not afraid of being stuffed with 卷   QvQ

### 功能概述

* 支持卷积神经网络，支持多输入和多分支结构，可计算部分分支
* 无任何第三方库依赖，不依赖 BLAS/NNPACK 等计算框架
* 纯 C++ 实现，跨平台，支持 android ios 等
* ARM NEON 汇编级良心优化，计算速度极快
* 精细的内存管理和数据结构设计，内存占用极低
* 支持多核并行计算加速，ARM big.LITTLE cpu 调度优化
* 整体库体积小于 500K，并可轻松精简到小于 300K
* 可扩展的模型设计，支持 8bit 量化和半精度浮点存储，可导入 caffe/pytorch/mxnet/onnx 模型
* 支持直接内存零拷贝引用加载网络模型
* 可注册自定义层实现并扩展
* 恩，很强就是了，不怕被塞卷 QvQ

---

### Example project

https://github.com/Tencent/ncnn/tree/master/examples/squeezencnn

### 技术交流QQ群：637093648(已满qaq) 853969140  答案：卷卷卷卷卷

---

### License

BSD 3 Clause

---

### 本分支说明

这个分支是用于ncnn int8新特的开发，不稳定版本，无法保证所有模型都能使用，我在这里测试好之后，才会Pull Request到ncnn master分支。

---

### Benchmark(不定时更新)

测试平台使用瑞星微的RK3399(Cortex-A72@1.8GHz x 2 + Cortex-A53@1.5GHz x 4)，android系统版本7.1.0，baseline参考Open AI Lab的Tengine推理框架，该框架支持Int8推理。

|                 | ncnn-A72x2 | Tengine-A72x2 | ncnn-A53x4 | Tengine-A53x4 |
| --------------- | ---------- | ------------- | ---------- | ------------- |
| SqueezeNet v1.1 |            |               |            |               |
| Float32         | 51.6       | 49.5          | 59.2       | 66.5          |
| Int8            | 35.4       | 36.5          | 44.8       | 55.5          |
| MobileNet_v1    |            |               |            |               |
| Float32         | 82.8       | 64.1          | 78.3       | 66.5          |
| Int8            | 50.4       | 44            | 60.7       | 58            |
| MobileNet SSD   |            |               |            |               |
| Float32         | 158        | null          | 163        | null          |
| Int8            | 101        | 89            | 120        | 127           |

