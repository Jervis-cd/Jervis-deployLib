# RKNN

## 1.文件准备

仓库地址：https://github.com/rockchip-linux/rknn-toolkit2?tab=readme-ov-file

整体框架：

![framework](./images/framework.png)

**框架解读：**要使用RKNPU进行模型推理，首先需要运行RKNN-Toolkit2工具，将训练好的模型转换为RKNN格式模型，然后使用RKNN C API 或Python API在开发板上进行推理

**仓库工具：**

* ***RKNN-Toolkit2***：软件开发套件，可在PC和瑞芯微NPU平台上运行，进行模型转换、量化、推理、性能和内存评估、量化精度分析、模型加密等功能
* ***RKNN-Toolkit-Lite2：***提供Python编程接口，部署RKNN模型

* ***RKNN Runtime：***提供C/C++编程接口，部署RKNN模型
