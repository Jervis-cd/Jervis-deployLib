#include<NvInfer.h>
#include<cuda_runtime.h>

#include<stdarg.h>
#include<unordered_map>
#include<iostream>

#include "infer.hpp"


namespace trt{

// 定义宏函数检查cuda runtime api是否运行成功
#define checkRuntime(call)                                                                \
do{                                                                                       \
  auto ___call__ret_code__=(call)                                                         \
  if(___call__ret_code__!=cudaSuccess){                                                   \
    INFO("CUDA Runtime error %s # %s,code=%s [ %d ]",#call,                               \
         cudaGetErrorString(___call__ret_code__),cudaGetErrorName(___call__ret_code__),   \
         ___call_ret_code__),                                                             \
  }                                                                                       \
}while(0)


static std::string file_name(const std::string &path,bool include_suffix){
  /*返回输入路径的文件名称，并可指定是否返回后缀*/
  if(path.empty()) return "";

  int p=path.rfind('/');
  int e=path.rfind('\\');

  p=std::max(p,e);
  p+=1;

  if(include_suffix) return path.substr(p);

  int u=path.rfind('.');
  if(u==-1) return path.substr(p);

  if(u<=p) u=path.size();
  return path.substr(p,u-p);
}

// 宏函数定义，打印log
void __log_func(const char *file,int line,const char *fmt,...){
  va_list vl;
  va_start(vl,fmt);
  char buffer[2048];
  std::string filename=file_name(file,true);

  int n=snprintf(buffer,sizeof(buffer),"[%s:%d]:",filename.c_str(),line);
  vsnprintf(buffer+n,sizeof(buffer)-n,fmt,vl);
  fprintf(stdout,"%s\n",buffer);
}

class __native_nvinfer_logger:public nvinfer1::ILogger{
 public:
  virtual void log(nvinfer1::ILogger::Severity severity,const char *msg) noexcept override{
    if(severity==nvinfer1::ILogger::Severity::kINTERNAL_ERROR){
      INFO("NVInfer INTERVAL_ERROR:%s",msg);
      std::abort();
    }else if(severity==nvinfer1::ILogger::Severity::kERROR){
      INFO("NVInfer:%s",msg);
    }
  }
};

static __native_nvinfer_logger gLogger;

template <typename _T>
static void destroy_nvidia_pointer(_T *ptr){
  if(ptr) ptr->destroy();
}

class __native_engine_context{
 public:
  virtual ~__native_engine_context(){destroy();}

  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IRuntime> runtime_=nullptr;

  bool construct(const void *pdata,size_t size){
    destroy();
    if(pdata==nullptr || size==0) return false;

    runtime_=std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger),
                                                  destroy_nvidia_pointer<nvinfer1::IRuntime>);
    if(runtime_==nullptr) return false;

    engine_=std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(pdata,size,nullptr),
                                                    destroy_nvidia_pointer<nvinfer1::ICudaEngine>);
    if(engine_==nullptr) return false;

    context_=std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(),
                                                          destroy_nvidia_pointer<nvinfer1::IExecutionContext>);
    return context_!=nullptr;
  }

 private:
  void destroy(){
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }
};

class InferImpl:public Infer{
 public:
  std::shared_ptr<__native_engine_context> context_;
  std::unordered_map<std::string,int> binding_name_to_index_;

  virtual ~InferImpl()=default;

  bool construct(const void *data,size_t size){
    context_=std::make_shared<__native_engine_context>();
    if(!context_->construct(data,size)) return false;

    setup();
    return true;
  }

  void setup(){
    auto engine=this->context_->engine_;
    int nbBindings=engine->getNbBindings();

    binding_name_to_index_.clear();
    for(int i=0;i<nbBindings;i++){
      const char *bindingName=engine->getBindingName(i);
      binding_name_to_index_[bindingName]=i;
    }
  }
};

}



