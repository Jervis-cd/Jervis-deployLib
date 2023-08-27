#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <string>
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
    return std::make_tuple(static_cast<uint8_t>(b*255),static_cast<uint8_t>(g*255),static_cast<uint8_t>(r * 255));
}

//随机获取颜色
static std::tuple<uint8_t,uint8_t,uint8_t> random_color(int id){
    float h_plane=((((unsigned int)id<<2)^0x937151)%100)/100.0f;
    float s_plane=((((unsigned int)id<<3)^0x315793)%100)/100.0f;
    return hsv2bgr(h_plane,s_plane,1);
}


// 构建体制机制记录器
class TRTLogger:public nvinfer1::ILogger{

public:
    virtual void log(Severity severity,nvinfer1::AsciiChar const* msg) noexcept override{

        if(severity<=Severity::kWARNING){

            if(severity==Severity::kWARNING){

                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;


//通过智能指针管理nv返回的指针，内存自动释放
template<typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr){

    return std::shared_ptr<_T>(ptr,[](_T* p){p->destroy();});
}

//判断文件是否存在
bool exists(const std::string& path){
    return access(path.c_str(), R_OK) == 0;
}

//解析onnx构建TensorRTmodel
bool build_model(){

    if(exists("yolov5s.trtmodel")){

        printf("yolov5s.trtmodel has exists.\n");
    }

    //定义nv日志记录器
    TRTLogger logger;

    //网络构建器以及其配置优化和网络结构文件
    auto builder=make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config=make_nvshared(builder->createBuilderConfig());
    auto network=make_nvshared(builder->createNetworkV2(1));

    //创建一个onnx文件解析器
    auto parser=make_nvshared(nvonnxparser::createParser(*network,logger));
    if(!parser->parseFromFile("yolov5s.onnx",1)){

        printf("Failed to parse yolov5s.onnx.\n");
        return false;
    }

    //设置maxbatchsize大小
    int maxBatchSize=10;
    //设置workspace大小,某些网络结构需要申请内存,所以在此先分配
    config->setMaxWorkspaceSize(1<<28);          //左移运算，256MB

    //当动态batch时，需要设置多个profile
    auto profile=builder->createOptimizationProfile();
    //从解析的onnx文件中获取输入张量
    auto input_tensor=network->getInput(0);
    //获取输入维度
    auto input_dims=input_tensor->getDimensions();

    //配置动态batch的范围
    input_dims.d[0]=1;
    profile->setDimensions(input_tensor->getName(),nvinfer1::OptProfileSelector::kMIN,input_dims);
    profile->setDimensions(input_tensor->getName(),nvinfer1::OptProfileSelector::kOPT,input_dims);
    input_dims.d[0]=maxBatchSize;
    profile->setDimensions(input_tensor->getName(),nvinfer1::OptProfileSelector::kMAX,input_dims);
    //将profile加入到配置文件中
    config->addOptimizationProfile(profile);

    //根据网络和配置文件构建engine
    auto engine=make_nvshared(builder->buildEngineWithConfig(*network,*config));
    if(engine==nullptr){

        printf("Build engine failed.\n");
        return false;
    }

    //序列化模型并保存
    auto model_data=make_nvshared(engine->serialize());
    FILE* f=fopen("yolov5s.trtmodel","wb");
    fwrite(model_data->data(),1,model_data->size(),f);
    fclose(f);

    printf("Build Done.\n");
    return true;
}

//打开文件
std::vector<unsigned char> load_file(const std::string& file){

    std::ifstream in(file,std::ios::in|std::ios::binary);
    
}




