#include<NvInfer.h>
#include<cuda_runtime.h>

#include<stdarg.h>
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

// 打印log
void __log_func(const char *file,int line,const char *fmt,...){
  va_list vl;
  va_start(vl,fmt);
  char buffer[2048];
  std::string filename=file_name(file,true);

  int n=snprintf(buffer,sizeof(buffer),"[%s:%d]:",filename.c_str(),line);
  vsnprintf(buffer+n,sizeof(buffer)-n,fmt,vl);
  fprintf(stdout,"%s\n",buffer);
}

}



