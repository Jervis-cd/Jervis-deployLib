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
#define checkRuntime(op) __check_cuda_runtime((op),#op,__FILE__,__LINE__)

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

    std::ifstream in(file,std::ios::in|std::ios::binary);  //以二进制读的方式打开文件
    if(!in.is_open()){

        return {};
    }

    in.seekg(0,std::ios::end);          //从文件末尾计算偏移量
    //返回当前定位指针的位置
    size_t length=in.tellg();          //size_t=long unsigned int

    std::vector<uint8_t> data;
    if(length>0){

        in.seekg(0,std::ios::beg);          //beg表示从文件开始计算偏移量
        data.resize(length);

        in.read((char*)&data[0],length);
    }
    in.close();
    return data;
}

//TensorRT推理部分
void inference(){

    TRTLogger logger;
    //打开序列化文件
    auto engine_data=load_file("yolov5s.trtmodel");
    //构建runtime
    auto runtime=make_nvshared(nvinfer1::createInferRuntime(logger));
    //反序列化
    auto engine=runtime->deserializeCudaEngine(engine_data.data(),engine_data.size());

    if(engine==nullptr){

        printf("Deserialize cuda engine failed.\n");
        return;
    }

    if(engine->getNbBindings()!=2){

        printf("onnx导出有问题,必须一个输入和一个输出,当前有: %d个输出.",engine->getNbBindings()-1);
        return;
    }

    //创建一个cuda流
    cudaStream_t stream=nullptr;
    checkRuntime(cudaStreamCreate(&stream));         //检车cuda runtime API是否可用
    //创建执行上下文
    auto execution_context=make_nvshared(engine->createExecutionContext());

    //输入信息
    int input_batch=1;
    int input_channel=3;
    int input_height=640;
    int input_width=640;

    //输入元素个数
    int input_numel=input_batch*input_channel*input_height*input_width;
    float* input_data_host=nullptr;
    float* input_data_device=nullptr;
    //分配内存主机host空间以及device内存空间
    checkRuntime(cudaMallocHost(&input_data_host,input_numel*sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device,input_numel*sizeof(float)));

    //letter box
    auto image=cv::imread("car.jpg");
    //通过双线性差值对图像进行resize,计算缩放比例
    float scale_x=input_width/(float)image.cols;
    float scale_y=input_height/(float)image.rows;
    float scale=std::min(scale_x,scale_y);

    float i2d[6],d2i[6];
    //源图像和目标图像几何中心对齐
    i2d[0]=scale;i2d[1]=0;i2d[2]=(-scale*image.cols+input_width+scale-1)*0.5;
    i2d[3]=0;i2d[4]=scale;i2d[5]=(-scale*image.rows+input_height+scale-1)*0.5;

    cv::Mat m2x3_i2d(2,3,CV_32F,i2d);
    cv::Mat m2x3_d2i(2,3,CV_32F,d2i);
    cv::invertAffineTransform(m2x3_i2d,m2x3_d2i);               //获取放射变换矩阵

    cv::Mat input_image(input_height,input_width,CV_8UC3);
    //进行仿射变换获取输入图像
    cv::warpAffine(image,input_image,m2x3_i2d,input_image.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar::all(114));
    cv::imwrite("input_images.png",input_image);

    //计算图像所占的长度
    int image_area=input_image.cols*input_image.rows;
    unsigned char* pimage=input_image.data;

    //创建三个指针分别指向图像的bgr三个通道数据
    float* phost_b=input_data_host+image_area*0;
    float* phost_g=input_data_host+image_area*1;
    float* phost_r=input_data_host+image_area*2;

    //BGR->RGB，并对图像作归一化操作
    for(int i=0;i<image_area;++i,pimage+=3){
        *phost_r++=pimage[0]/255.0f;
        *phost_g++=pimage[1]/255.0f;
        *phost_b++=pimage[2]/255.0f;
    }
    
    //将host输出复制到device
    checkRuntime(cudaMemcpyAsync(input_data_device,input_data_host,input_numel*sizeof(float),cudaMemcpyHostToDevice,stream));

    //获取输出维度，yolov5输出维度(batch,25200,85)
    auto output_dims=engine->getBindingDimensions(1);       //获取输出维度
    int output_numbox=output_dims.d[1];                     //25200
    int output_numprob=output_dims.d[2];                    //85
    int num_classes=output_numprob-5;                       //获取类别总数

    //output元素个数
    int output_numel=input_batch*output_numbox*output_numprob;
    //创建输出在host
    float* output_data_host=nullptr;
    float* output_data_device=nullptr;
    //分配存储空间给网络输出
    checkRuntime(cudaMallocHost(&output_data_host,sizeof(float)*output_numel));
    checkRuntime(cudaMalloc(&output_data_device,sizeof(float)*output_numel));

    //明确当前推理使用的数据大小
    auto input_dims=engine->getBindingDimensions(0);        //获取输入维度
    input_dims.d[0]=input_batch;                            //输入维度与输出维度相同
    execution_context->setBindingDimensions(0,input_dims);  //context设置输入维度

    float* bindings[]={input_data_device,output_data_device};
    //启动cuda流异步推理
    bool success=execution_context->enqueueV2((void**)bindings,stream,nullptr);
    //将输出从device复制到host
    checkRuntime(cudaMemcpyAsync(output_data_host,output_data_device,sizeof(float)*output_numel,cudaMemcpyDeviceToHost,stream));
    //同步操作
    checkRuntime(cudaStreamSynchronize(stream));

    //decode box从不同尺度下的预测狂还原到原输入图上(包括:预测框，类被概率，置信度）
    std::vector<std::vector<float>> bboxes;
    float confidence_threshold=0.25;                //置信度
    float nms_threshold=0.5;                        //iou
    for(int i=0;i<output_numbox;++i){
        float* ptr=output_data_host+i*output_numprob;
        float objness=ptr[4];
        if(objness<confidence_threshold)
            continue;

        float* pclass=ptr + 5;
        int label=std::max_element(pclass, pclass + num_classes) - pclass;
        float prob=pclass[label];
        float confidence=prob*objness;
        if(confidence<confidence_threshold)
            continue;

        // 中心点、宽、高
        float cx=ptr[0];
        float cy=ptr[1];
        float width=ptr[2];
        float height=ptr[3];

        // 预测框
        float left=cx-width*0.5;
        float top=cy-height*0.5;
        float right=cx+width*0.5;
        float bottom=cy+height*0.5;

        // 对应图上的位置
        float image_base_left=d2i[0]*left+d2i[2];
        float image_base_right=d2i[0]*right+d2i[2];
        float image_base_top=d2i[0]*top+d2i[5];
        float image_base_bottom=d2i[0]*bottom+d2i[5];
        bboxes.push_back({image_base_left,image_base_top,image_base_right,image_base_bottom,(float)label,confidence});
    }
    printf("decoded bboxes.size = %d\n",bboxes.size());

    // nms非极大抑制
    std::sort(bboxes.begin(),bboxes.end(),[](std::vector<float>& a,std::vector<float>& b){return a[5]>b[5];});
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<std::vector<float>>box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const std::vector<float>& a,const std::vector<float>& b){
        float cross_left=std::max(a[0],b[0]);
        float cross_top=std::max(a[1],b[1]);
        float cross_right=std::min(a[2],b[2]);
        float cross_bottom=std::min(a[3],b[3]);

        float cross_area=std::max(0.0f,cross_right-cross_left) * std::max(0.0f,cross_bottom-cross_top);
        float union_area=std::max(0.0f,a[2]-a[0])*std::max(0.0f,a[3]-a[1]) 
                         +std::max(0.0f,b[2]-b[0])*std::max(0.0f,b[3]-b[1])-cross_area;
        if(cross_area==0||union_area==0) return 0.0f;
        return cross_area/union_area;
    };

    for(int i=0;i<bboxes.size();++i){
        if(remove_flags[i]) continue;

        auto& ibox=bboxes[i];
        box_result.emplace_back(ibox);
        for(int j=i+1;j<bboxes.size();++j){
            if(remove_flags[j]) continue;

            auto& jbox=bboxes[j];
            if(ibox[4]==jbox[4]){
                // class matched
                if(iou(ibox,jbox)>=nms_threshold)
                    remove_flags[j]=true;
            }
        }
    }
    printf("box_result.size=%d\n",box_result.size());

    for(int i=0;i<box_result.size();++i){
        auto& ibox=box_result[i];
        float left=ibox[0];
        float top=ibox[1];
        float right=ibox[2];
        float bottom=ibox[3];
        int class_label=ibox[4];
        float confidence=ibox[5];
        cv::Scalar color;
        std::tie(color[0],color[1],color[2])=random_color(class_label);
        cv::rectangle(image,cv::Point(left,top),cv::Point(right, bottom),color,3);

        auto name=cocolabels[class_label];
        auto caption=cv::format("%s %.2f", name, confidence);
        int text_width=cv::getTextSize(caption,0,1,2,nullptr).width+10;
        cv::rectangle(image,cv::Point(left-3,top-33),cv::Point(left+text_width,top),color,-1);
        cv::putText(image,caption,cv::Point(left,top-5),0,1,cv::Scalar::all(0),2,16);
    }
    cv::imwrite("result.jpg",image);

    //释放创建的流和分配的空间
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}

int main(){
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}





