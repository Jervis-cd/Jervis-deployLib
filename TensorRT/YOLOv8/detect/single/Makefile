exec_name := infer
workdir   := workspace
srcdir    := src

cc        := g++
stdcpp    := c++11
nvcc	  	:= /usr/local/cuda-11.6/bin/nvcc -ccbin=$(cc)
cuda_arch := 

syslib    := /usr/local/lib /usr/lib64
cuda_home := /usr/local/cuda-11.6
opencv	  := /usr/local
TensorRT  := /home/dockershared/TensorRT-8.6.1.6

cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

include_paths := src				\
  $(cuda_home)/include     	\
	$(TensorRT)/include 			\
	$(opencv)/include/opencv4

library_paths := $(cuda_home)/lib64 \
	$(syslib)													\
	$(opencv)/lib											\
	$(TensorRT)/lib

empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

link_sys       := stdc++ dl
link_cuda      := cuda cublas cudart cudnn
link_tensorRT  := nvinfer nvinfer_plugin nvonnxparser
link_opencv    := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs
link_librarys  := $(link_cuda) $(link_tensorRT) $(link_sys) $(link_opencv)

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread
cu_compile_flags  := -std=$(stdcpp) -w -g -O0 -m64 $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

$(exec_name)   : $(workdir)/$(exec_name)

all       : $(exec_name)
run       : $(exec_name)
	@cd $(workdir) && ./$(exec_name) $(run_args)

$(workdir)/$(exec_name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@$(cc) $^ -o $@ $(link_flags)

$(srcdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(srcdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

$(srcdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
    
$(srcdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

clean :
	@rm -rf $(srcdir)/*.o $(workdir)/$(exec_name) $(workdir)/*.jpg

.PHONY : clean run $(exec_name) all

export LD_LIBRARY_PATH:=$(library_path_export)