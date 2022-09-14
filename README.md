# Swin Transformer + BiFPN + YOLO V5

In this repository, we present a new structure of one-stage detection model based on Swin Transformer. There are other repositories adopting Swin Transformer as the backbone of the model, but most of them are only implemented in Faster R-CNN or Mask R-CNN structure, following the official. These two- stage detectors can guarantee high performance, but highly suffer from the latency. 

The purpose of this implementation is to take advantage of the high performance of Swin Transformer and the speed of one-stage detection. 
...

* concatenate BiFPN to the backbone, instead of the FPN adopted in the official implementation
* adopt YOLO V5-style bbox regression and prediction
* train the model with focal loss


