# Swin Transformer + BiFPN + YOLO V5

Swin Transformer is one of the most powerful backbones in the image domain today. It is the first model that allowed ViT to be applied to dense prediction tasks such as object detection and segmentation. There are already other repositories adopting Swin Transformer as the backbone of the model, but most of them only implements in Faster R-CNN or Mask R-CNN structure, following the official. These two-stage detectors can guarantee high performance, but highly suffer from the latency. 

In this implementation, we present a new structure of one-stage detection model based on Swin Transformer. To remain the high performance of Swin Transformer and the speed of one-stage detection, and to overcome the drawbacks of one-stage detection, we changed it as follow:
* concatenate BiFPN to the backbone, instead of the FPN of the official implementation
* adopt YOLO V5-style bbox regression and prediction
* train the model with focal loss


하지만 그들 중 대부분은 오직 Faster R-CNN or Mask R-CNN 모델로만 구현한다, 오피셜을 따라서.  

Swin Transformer는 최근 이미지 도메인에서 가장 강력한 백본이다. 그것은 ViT가 object detection이나 segmentation과 같은 dense prediction을 할 수 있도록 제시한 모델이며, 그것은 이미 오피셜을 포함한 많은 구현에서 그것은 백본으로 채택되었다. 하지만 

In this repository, we present a new structure of one-stage detection model based on Swin Transformer. There are other repositories adopting Swin Transformer as the backbone of the model, but most of them are only implemented in Faster R-CNN or Mask R-CNN structure, following the official. These two- stage detectors can guarantee high performance, but highly suffer from the latency. 

The purpose of this implementation is to take advantage of the high performance of Swin Transformer and the speed of one-stage detection. 
...

* concatenate BiFPN to the backbone, instead of the FPN adopted in the official implementation
* adopt YOLO V5-style bbox regression and prediction
* train the model with focal loss


