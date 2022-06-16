## Reproduction Blog: A ConvNet for the 2020s


**Authors**

Xinqi Li - 5478073 - X.Li-63@student.tudelft.nl

Haoran Wang - 5468175 - H.Wang-46@student.tudelft.nl


## Introduction
### Background
In the 2020s, Vision Transformers (ViT) began to surpass the ConvNets as the preferred choice for vision tasks. The widely held belief is that ViTs are more accurate, efficient, and scalable than ConvNets. The introduction of ViT completely changed the landscape of network architecture design. However, without the ConvNet inductive biases, a vanilla ViT model faces many challenges in being adopted in generic vision tasks. Hierarchical Transformers utilize a hybrid approach to overcome this problem. 
Swin Transformer[1] is a milestone work in this direction. It demonstrated that Transformers can be adopted as a generic vision backbone and achieve good performance across a range of computer vision tasks. The success of Swin Transformer also revealed that convolution remains desired. Under this situation, many of the advancements of Transformers for computer vision have been aimed at bringing back convolutions. 
In this paper, the authors investigated the architecture distinctions between ConvNets and Transformers and try to identify the confounding variables when comparing the network performance. They proposed ConvNeXts[2], a pure ConvNet model that can compete with state-of-the-art hierarchical vision transformers in multiple computer vision tasks.
To do this, the authors start with a standard ResNet (e.g. ResNet-50) trained with improved procedure. They gradually change the architecture to the construction of a hierarchical ViT (e.g. Swin-T), and discover several key components that contribute to the performance.
![](https://i.imgur.com/C0GMWPe.png)


### Dataset
In the paper, the authors use ImageNet-1K as the training dataset. In our reproduction, we decide to transfer the method to a different dataset, CIFAR-100[3]. It has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. It is a relatively small dataset compared to the ImageNet, which makes it feasible for our reproduction.

## Approach
In this section, we follow the trajectory of the paper which goes from a ResNet to a ConvNet. The starting point is a ResNet-50 model. Then we study a series of design decisions which can be summarized as 1)macro design, 2)ResNeXt, 3)inverted bottleneck, 4)large kernel size, 5)various layer-wise micro designs. 
<!-- Figure 1. is our reproduced process of this paper. -->

<!-- <figure>
  <img
    src="https://i.imgur.com/maMJIHH.jpg"
    alt="Process of building ConvNeXt"
    width="450">
  <figcaption>Figure 1. Results bar plot on DIFAR-100 dataset with 60 epoches</figcaption>
</figure> -->

Besides, in this paper, the authors use an improved ResNet-50 model as a baseline to compare the results. In our case, we use ResNet-50 model from PyTorch [ResNet-50]: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50.


### 1. Macro Design
According to the paper, Swin Transformers use a multi-stage design, and each stage has a different feature map resolution. There are two designs that can be adjusted in this part: the stage compute ratio, and the "stem cell" structure.

#### Changing stage compute ratio
The original design of the computation distribution across stages in ResNet was largely empirical. Swin-T, on the other hand, follows the same principle but with a different stage ratio of 1:1:3:1. And the ratio of larger Swin Transformers is 1:1:9:1. Therefore, we change the number of blocks in each stage from (3, 4, 6, 3) in ResNet-50 to (3, 3, 9, 3).
We use this stage compute ratio from now on.

#### Changing stem to "Patchify"
The stem cell in standard ResNet contains a 7x7 convolution layer with stride 2, followed by a max pool, which results in a 4x4 downsampling of the input images. While in ViTs, a "pachify" strategy is used as the stem cell, which corresponds to a larger kernel size. Swin Transformer uses a similar "pachify" layer, but with a smaller kernel size of 4. Therefore, we substitute the ResNet-style stem cell with a patchify layer implemented using a 4x4, stride 4 convolutional layer. 
We will use the "pachify" stem cell in the network.


### 2. ResNeXt-ify
In this part, the authors try to adopt the idea of ResNeXt[4], which has a better FLOPs/accuracy trade-off than a vanilla ResNet. The core component of the ResNeXt is grouped convolution, where the convolutional filters are separated into different groups. More precisely, ResNeXt employs grouped convolution for the 3×3 convolutional layer in a bottleneck block, which can significantly reduce the FLOPs. The network width is expanded to compensate for the capacity loss.
In this case, the authors use depthwise convolution, a special case of grouped convolution where the number of groups equals the number of channels. The combination of depthwise Conv and 1x1 Convs leads to a separation of spatial and channel mixing. The use of depthwise convolution effectively reduces the network FLOPs and also the accuracy. Therefore, the authors decide to increase the network width to the same number of channels as Swin-T's (from 64 to 96). 


### 3. Inverted Bottleneck
One important design of the Transformer block is that it creates an inverted bottleneck. Here, the authors want to explore the effect of inverted bottleneck design. This change reduces the FLOPs of the whole network.
We will use inverted bottleneck afterward.


### 4. Large Kernel Sizes
In this part, the paper focuses on the effect of large convolutional kernels. While large kernel sizes have been used in the past with ConvNets, the best way is to stack small kernel-sized (3x3) Conv layers, which are more efficient on modern GPUs[5]. Although Swin Transformers reintroduced the local window to the self-attention block, the window size is at least 7x7, larger than the ResNet kernel size of 3x3. 

#### Moving up depthwise conv layer
To explore large kernels, one prerequisite is to move up the position of the depthwise Conv layer. As we have an inverted bottleneck block, the complex/inefficient models will have fewer channels, while the efficient 1x1 layers will do the heavy lifting.

#### Increasing the kernel size
With all the preparations, the benefit of using larger kernel-sized convolutions is significant. The authors experimented with several kernel sizes, including 3, 5, 7, 9, and 11. 


### 5. Micro Design
In this section, the authors investigate several other architecture differences at a micro level, focusing on specific choices of activation functions and normalization layers.

#### Replacing ReLU with GELU
There are many activation functions, but the Rectified Linear Unit (ReLU)[6] is still extensively used in ConvNets. The Gaussian Error Linear Unit (GELU)[7] is used in the most advanced Transformers. Therefore, the authors decide to substitute the ReLU with GELU, 

#### Fewer activation functions
Transformer blocks usually have fewer normalization layers as well. In this paper, researchers remove two BatchNorm (BN) layers, leaving only one BN layer before the Conv 1x1 layers.

#### Substituting BN with LN
BatchNorm is an important component in ConvNets because it improves convergence and reduces overfitting. However, BN also has many disadvantages. The simpler Layer Normalization (LN) has been used in Transformers, resulting in good performance across different application scenarios. Therefore, substituting BN with LN is a reasonable experiment. 

#### Separating downsampling layers
In ResNet, the spatial downsampling is achieved by the residual at the start of each stage, using 3x3 Conv with stride 2. In Swin Transformers, a separate downsampling layer is added between stages. The authors explore a similar strategy in which they use 2x2 Conv layers with stride 2 for spatial downsampling. 


## Result
Our reproduced result is slightly different from the paper. We show the original results and reproduced results in the following graph to compare them. 

Original results           |  Reproduced results
:-------------------------:|:-------------------------:
![](https://i.imgur.com/MNweAjT.png)  |  ![](https://i.imgur.com/maMJIHH.jpg)

We can see that the differences are mostly after the step changing ReLU to GELU. The paper's results keep increasing till the end. However, our results start to decrease after changing ReLU to GELU.

We got our best result after step 11 in 100 epochs, replacing ReLU with GELU. Detials of accuracy is shown is Figure 2. The following steps did not help us improve the performance. The best top 1 accuracy after 100 epoches is 80.2%. 

The difference might be caused by several reasons. 1) The difference between the dataset. The original paper is using the ImageNet dataset and we are using the CIFAR-100 dataset. The data complexity and variation is different. Using fewer activations and fewer norms might cause the overfitting in the test data. 2)Training time. Due to the computational limitation, the previous result are only trained on 60 epochs. Some model might converge slower as we can also in Figure 2. 

<figure>
  <img
    src="https://i.imgur.com/TRF2pvE.png"
    alt="Detailed results"
    width="700">
  <figcaption>Figure 2. Top 1 Accuracy of CIFAR-100 Dataset on Different Model Disign</figcaption>
</figure>


## Conclusion
We followed the steps of constructing ConvNeXt from ConvNet in the paper, and reproduced the results on a different dataset, CIFAR-100. Our reproduced result is slightly different with the paper's. But consider we are using a another dataset, these differences are acceptable. This project also inspires us to think the importance of convolution in Computer Vision. 


## References
1. Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
2. Liu, Zhuang, et al. "A ConvNet for the 2020s." arXiv preprint arXiv:2201.03545 (2022).
3. Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009): 7.
4.  Xie, Saining, et al. "Aggregated residual transformations for deep neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
5.  Lavin, Andrew, and Scott Gray. "Fast algorithms for convolutional neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
6.  Nair, Vinod, and Geoffrey E. Hinton. "Rectified linear units improve restricted boltzmann machines." Icml. 2010.
7.  Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." arXiv preprint arXiv:1606.08415 (2016).
