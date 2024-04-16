#include<vector>
#include<memory>

#include "infer.hpp"
#include "yolo.hpp"

namespace yolo{

class InferImpl:public Infer{
 public:
  std::shared_ptr<trt::Infer> trt_;
  virtual ~InferImpl()=default;

  virtual BoxArray forward(const Image &image,void *stream=nullptr) override {
    auto output=forwards({image},stream);
    if(output.empty()) return {};
    return output[0];
  }

  virtual std::vector<BoxArray> forwards(const std::vector<Image> &images,void *stream=nullptr) override{
    int num_image=images.size();
    if(num_image==0) return {};

    auto input_dims=trt_->static_dims(0);
  }
};
}