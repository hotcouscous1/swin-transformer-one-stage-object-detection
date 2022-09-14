# Swin Transformer + BiFPN + YOLO V5

**[Yet Developing ⚙️]**

Swin Transformer is one of the most powerful backbones in the image domain today. It is the first model that allowed ViT to be applied to dense prediction tasks such as object detection and segmentation.  

There are already other repositories adopting Swin Transformer as the backbone of the model, but most of them only implements in Faster R-CNN or Mask R-CNN structure, following the official. These two-stage detectors can guarantee high performance, but highly suffer from the latency.  

In this implementation, we present a new structure of one-stage detection model based on Swin Transformer. To remain the high performance of Swin Transformer and get the speed of one-stage detection, we changed it as follow:
* concatenate Bi-FPN to the backbone, instead of the FPN of the official implementation
* adopt YOLO V5-style bbox regression and prediction
* train the model with Focal Loss



