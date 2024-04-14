#ifndef __YOLO_HPP__
#define __YOLO_HPP__

#include<vector>
#include<infer.hpp>

namespace yolo{

struct Image{
const void *bgrptr=nullptr;
int width=0, height=0;

Image()=default;
Image(const void *bgrptr,int width,int height):bgrptr(bgrptr),width(width),height(height){};
};

struct Box{
  float left,top,right,bottom,confidence;
  int class_label;

  Box()=default;

  Box(float left,float top,float right,float bottom,float confidence, int class_label)
  :left(left),top(top),right(right),bottom(bottom),confidence(confidence),class_label(class_label){};
};

typedef std::vector<Box> BoxArray;

class Infer{
 public:
  virtual BoxArray forward(const Image &image,void *stream=nullptr)=0;
};

std::shared_ptr<Infer> load(const std::string &engine_file,float confidence_threshold=0.25f,float nms_threshold=0.5f);

}   // namespace yolo

#endif    // YOLO_HPP