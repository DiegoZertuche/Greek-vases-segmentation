# Final Project for "CS109b: Advanced Topics in Data Science" at Harvard University

## Members
* Javiera Astudillo
* Andrew Lee
* Diego Zertuche

# Motivation, context & framing of the problem

## Introduction
The study of ancient greek art facilitates the understanding of Greek society. Within its forms, pottery stands out due to its durability, its historical coverage and the vast amount of available samples. The extensive amount of vases obstructs their study due to the time consumption of the collection of relevant samples for a given research topic. It seems desirable to come up with some automatic object detection system for vase selection. In this work we focus on the detection of shields in vase images. Initially, we aim to detect the presence of shields within vase images. If this turns out successful, we will move on one step further to detect the location of the shield within the image.

## Related work

### Image classification and segmentation
Image classification and segmentation has greatly developed in the last couple of years with multiple applications on medical imaging analysis, object detection, recognition tasks, traffic control systems within others [1]. Currently, Deep Learning techniques models lead as the state-of-the-art models in the field [2]. Here, we describe a few, though it's not a comprehensive overview of all existing techniques but just a brief listing of some of them.

The first image classification breakthrough of this century came in 2012 with Alexnet [9], which reached an accuracy of 15.3% on ImageNet dataset, more than 10.8 percentage points lower than runner-up. It implemented multiple convolution layers, alternating with max pooling layers and ReLu activations. After that, multiple models came up that extended Alexnet work such as VGG [10], Inception [11], Resnet [12] and MobileNet [13].

Object recognition models share similar components as image classifications models, such as CNN layers, and extend them further, incorporating encoder and decoder parts, skip-connections and multi-scale analysis. A couple of prominent works include [3] U-Net, [4] DeepLab and [5] Mask R-CNN. U-Net improved its predecessor's resolution representations by creating information streams between high and low-resolution sections; DeepLab introduced a dilation rate that enabled enlarging the receptive field at no extra computational cost; Mask R-CNN incorporates bounding box candidates which improves traditional region proposal methods. 

### AI and ML for artworks
Different museums and artwork centres have expressed interest in developing automatic analysis of their pieces. For instance, in 2019 the Metropolitan Museum of Art in New York uploaded a challenge to Kaggle for the recognition of artwork attributes within their collection. Multiple works have addressed object detection within artworks such as [6], [7] and [8].They relied either on CNN feature extractors or more specifically in Image Segmentation models. They are all closely related to our setting, though they are mostly applied to paintings, all with well-delimited boundaries. In the current setting, greek vases present an additional challenge imposed by the variable zoom and angle of the paintings within each vase image.

## Data

### Getting Labeled Images from Labelbox

Given the problem that there were multiple images with different angles of the vases, we had to go through all the shield images manually and label the images that actually had a visible shield in the image. We used [Labelbox](https://labelbox.com/) for this task. We then downloaded the manually labeled data from Labelbox and loaded only the images that actually had the shields in the image and had around 1,800 images with shields. We took a random sample of the non-shield images that matched the size of our shield data, and this is how we created our working dataset for the classification model. We split the data in train and test partitions, taking into account the vase numbers, to ensure that different images from the same vase were not in the train and test set.

![Original Mask](imgs/seg_target.png?raw=true)
| <b>Image Credits - Fig.2 - 4K Mountains Wallpaper</b>|

## Results

### Classifier

Once we had our labeled dataset we fitted a shield classification model, where a 0 prediction is equivalent to "no shield" in the image while a 1 indicates the presence of a "shield" in the vase painting. We used an inception inspired model given it is one of the state of the art models of CNN models for image classification [11].

We see that our model classification model achieves 64% and 65% accuracy on train and test sets correspondingly; given that we are working on a balanced setting, this results are positive. We can conclude that, even though the dataset size is limited, the model is able to detect shields patterns within an image and generalize it to further vases. This leads us to develop a segmentation model for detecting the position of a shield within a vase image.

### Segmentation

![Predicted Mask](imgs/seg_pred.png?raw=true)

The resulting images on the validation set show that our model can detect the position of a shield within a vase image. It misses the exact shape, especially for paintings with thin strokes, which is not surprising. These results are promising and suggest that we could further improve our model if we increased our masking dataset. We conclude that classification and automatic model detection is feasible for greek vases images; this could be extended to other objects within greek vases.



# References

* [1] David A. Forsyth and Jean Ponce. 2002. Computer Vision: A Modern Approach. Prentice Hall Professional Technical Reference.
* [2] Minaee, S., Boykov, Y., Porikli, F., Plaza, A., Kehtarnavaz, N., and Terzopoulos, D. 2020. “Image Segmentation Using Deep Learning: A Survey”, arXiv e-prints.
* [3] Ronneberger, O., Fischer, P., and Brox, T. 2015. “U-Net: Convolutional Networks for Biomedical Image Segmentation”, arXiv e-prints.
* [4] Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., and Yuille, A. L. 2016. “DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs”, arXiv e-prints.
* [5] He, K., Gkioxari, G., Dollár, P., and Girshick, R. 2017. “Mask R-CNN”, arXiv e-prints.
* [6] S. Smirnov and A. Eguizabal, "Deep learning for object detection in fine-art paintings," 2018 Metrology for Archaeology and Cultural Heritage (MetroArchaeo), 2018, pp. 45-49, doi: 10.1109/MetroArchaeo43810.2018.9089828.
* [7] H. -J. Jeon, S. Jung, Y. -S. Choi, J. W. Kim and J. S. Kim, "Object Detection in Artworks Using Data Augmentation," 2020 International Conference on Information and Communication Technology Convergence (ICTC), 2020, pp. 1312-1314, doi: 10.1109/ICTC49870.2020.9289321.
* [8] Gonthier, N., Gousseau, Y., Ladjal, S., and Bonfait, O. 2018. “Weakly Supervised Object Detection in Artworks”, arXiv e-prints.
* [9] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2012. "ImageNet classification with deep convolutional neural networks". In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1 (NIPS'12). Curran Associates Inc., Red Hook, NY, USA, 1097–1105.
* [10] Simonyan, K. and Zisserman, A. 2014. "Very Deep Convolutional Networks for Large-Scale Image Recognition", arXiv e-prints.
* [11] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. 2015. "Rethinking the Inception Architecture for Computer Vision", arXiv e-prints.
* [12] He, K., Zhang, X., Ren, S., and Sun, J. 2015. "Deep Residual Learning for Image Recognition", arXiv e-prints.
* [13] Howard, A. G. 2017. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", arXiv e-prints.

