#include<opencv2/opencv.hpp>

#include "yolo.hpp"

yolo::Image cvimg(const cv::Mat &image){
  return yolo::Image(image.data,image.cols,image.rows);
}

void single_inference(const std::string img_path,
                      const std::string engine_path,
                      const float confidence_tnreshold=0.25f,
                      const float nms_threshold=0.5f){
  cv::Mat image=cv::imread(img_path);

  auto model=yolo::load(engine_path,confidence_tnreshold,nms_threshold);
  if(model=nullptr) return;

  auto objs=model->forward(cvimg(image));
}

int main(int argc,const char* argv){

  return 0;
}


