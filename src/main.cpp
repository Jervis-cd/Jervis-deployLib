#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <chrono>

#include <opencv2/opencv.hpp>

//宏函数，检查cuda runtime API是否正常运行，返回错误代码以及错误信息 
#define chechRuntime(op) __check_cuda_runtime((op),#op,__FILE__,__LINE__)

bool __check_cuda_runtime(cudaError_t code,const char* op,const char* file,int line){

    if(code!=cudaSuccess){

        const char* err_name=cudaGetErrorName(code);
        const char* err_message=cudaGetErrorString(code);
        printf("runtime error %s:%d %s failed.\ncode=%s, message=%s\n",file,line,op,err_name,err_message);
        return false;
    }
    return true;
}

//内联函数，返回日志代码对应的日志信息
inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

//数据集的label
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

// HSV->BGRd
static std::tuple<uint8_t,uint8_t,uint8_t> hsv2bgr(float h,float s,float v){
    const int h_i=static_cast<int>(h*6);
    const float f=h*6-h_i;
    const float p=v*(1-s);
    const float q=v*(1-f*s);
    const float t=v*(1-(1-f)*s);
    float r,g,b;
    switch(h_i){
    case 0:r=v;g=t;b=p;break;
    case 1:r=q;g=v;b=p;break;
    case 2:r=p;g=v;b=t;break;
    case 3:r=p;g=q;b=v;break;
    case 4:r=t;g=p;b=v;break;
    case 5:r=v;g=p;b=q;break;
    default:r=1;g=1;b=1;break;}
    return make_tuple(static_cast<uint8_t>(b*255),static_cast<uint8_t>(g*255),static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t,uint8_t,uint8_t> random_color(int id){
    float h_plane=((((unsigned int)id<<2)^0x937151)%100)/100.0f;
    float s_plane=((((unsigned int)id<<3)^0x315793)%100)/100.0f;
    return hsv2bgr(h_plane,s_plane,1);
}




