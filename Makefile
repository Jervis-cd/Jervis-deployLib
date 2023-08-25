cc        := g++										#编译器名称
name      := pro										#项目名称
stdcpp    := c++11										#c++标准
cuda_home := /usr/local/cuda-11.7						#cuda home目录
cuda_arch := 											#显卡计算能力设置
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)			#cu程序编译器

# 资源以及存储文件夹
workdir   := workspace
srcdir    := src
objdir    := objs

#系统以及第三方库根目录
sys_home	:= /usr/local
opencv_home	:= /home/jervis/Documents/opencv-4.6.0
trt_home	:= /home/jervis/Documents/TensorRT-8.5.1.7

# 定义opencv和cuda需要用到的库文件
link_cuda      := cudart cudnn
link_tensorRT  := nvinfer nvinfer_plugin nvonnxparser
link_opencv    := opencv_core opencv_imgproc opencv_imgcodecs
link_sys       := stdc++ dl protobuf
link_librarys  := $(link_cuda) $(link_tensorRT) $(link_sys) $(link_opencv)

# 定义库文件路径，只需要写路径，不需要写-L
library_paths := $(cuda_home)/lib64		\
				 $(sys_home)/lib		\
				 $(trt_home)/lib

#拼接导出路径，方便之后导出
empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

# 头文件路径
include_paths := src              	\
    $(cuda_home)/include     		\
	$(trt_home)/include				\
	$(sys_home)/include/opencv4

# 把库路径和头文件路径拼接起来成一个，批量自动加-I、-L、-l
run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

#编译器设置
cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread
cu_compile_flags  := -std=$(stdcpp) -w -g -O0 -m64 $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

# 如果头文件修改了，这里的指令可以让他自动编译依赖的cpp或者cu文件
ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

# 源文件搜索，以及依赖项和中间文件路径设置
cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

$(name)   : $(workdir)/$(name)

all       : $(name)
run       : $(name)
	@cd $(workdir) && ./$(name) $(run_args)

# 链接
$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

#编译
$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

# 生成依赖项
$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
    
$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

# 定义清理指令
clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/*.trtmodel $(workdir)/classifier.onnx

# 防止符号被当做文件
.PHONY : clean run $(name)

# 导出依赖库路径，使得能够运行起来
export LD_LIBRARY_PATH:=$(library_path_export)