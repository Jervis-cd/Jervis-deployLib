#ifndef __YOLO_HPP__
#define __YOLO_HPP__

namespace yolo{

 struct Image{
  const void *bgrptr=nullptr;
  int width=0, height=0;

  Image()=default;
  Image(const void *bgrptr,int width,int height):bgrptr(bgrptr),width(width),height(height){};
 };

}

#endif    // YOLO_HPP