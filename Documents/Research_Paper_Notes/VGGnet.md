- Increase the depth of ConvNet architecture
- Use of very small (3x3) convolution filters in all layer

**Architecture**
- Input 224 x224 RGB image. 
- Subtract mean RGB value of a training set from each pixel.
- In convolution layer, filter with receptive field 3x3. 1x1 convolution filter also used
- Convolution stride- 1 pixel
- Padding is 1 pixel for 3x3 conv. layer. •Spatial pooling is carried out by 5 max-pooling layers. max pooling performed over 2x2 pixel window with stride 2..
- 3 fully-connected layer. first 2 have 4096 channel each, 3rd has 1000 channels. final layer is softmax layer.
- ReLU used. only one network contains.
- Local Response Normalisation (LRN) : doesnot improve performance but leads to increased memory consumption and computational time.

A stack of two 3x3 conv layer has effective receptive field of 5x5. three 3x3 conv layer has 7×7 effective receptive field.

- In 3 3x3 conv layer makes decision function more discriminative. 
- decreases number of parameters 
- c-number of channels : 3(3x3 xcxc) = 27c²
- for a single 7x7 conv layer. 7x7XCXC=49
- 1x1 conv. layer increases non-linearity.

**Training**.
At test time given a trained ConvNet and an input image, it is classified in the following way.
- it is isotropically rescaled to a 
Q- pre-defined smallest image side. 
We note that Q is not necessarily equal to the training scale S. Then, the network is applied densely over the rescaled test image  Namely, the fully-connected layers are first converted to convolutional layers (the first FC layer to a 7 x 7 conv. layer, the last two FC layers to 1x1 conv. layers). 
The resulting fully- convolutional net is then applied to the whole (uncropped) image. The result is a class score map with the number of channels = number of classes, and a variable spatial resolution, dependent on the input image size. 
Finally, to obtain a fixed-size vector of class scores for the image, the class score map is spatially averaged (sum-pooled). We also augment the test set by horizontal flipping of the images; the soft-max class posteriors of the original and flipped images are averaged to obtain the final scores for the image.

- Minibatch gradient descent used. 
- Batch size - 256, momentum-0.9,
- Regularised by weight decay and dropout regularisation for first two fully connected layers.

Weights were initialized randomly with zero mean and 10^-2 variance.

To obtain size 224x224 ConvNet input image, we randomly crop rescaled training images. For data augmentation, crops were randomly flipped horizontally and random RGB colour shift.

**Training** **image** **size**
- S-smallest side of isotropically rescaled training image
   S=224,whole image is not cropped 
   S>> 224 small part of image considered

To set S- 
single-scale training. fixed at one scale 
multi-scale training- a range is set
[ Smin, Smax] Smin = 256 Smay=-512

**Testing** 
image input,
Q-  pre-defined smallest image Side
Q not equal to S
We use several value of Q for each S to improve performance.
multi-crop evaluation is complementary to dense evaluation due to different convolution boundary conditions: 
- when applying a ConvNet to a crop, the convolved feature maps are padded with zeros
- while in the case of dense evaluation the padding for the same crop naturally comes from the neighbouring parts of an image (due to both the convolutions and spatial pooling), which substantially increases the overall network receptive field, so more context is captured. 
  
**Single Scale Evaluation.**
fixed s
Test image size for Q = S 
(Scale jittering at training time) 
for S in range S belong to [Smin, Smax]. 

Q=0.5 (Smin + Smax)

**Effects** 
1) on model A, local response normalization (A-LRN network) does not improve without any normalisalim layers.
2) Classification error decreases from A to E.C (1x1 conv layers) performs worse than D (3x3 conv layers). C is better than B.

- Deep net with shallow filters outperform shallow net with larger filters. 
- S in range performs well than S fixed

**Multiscale Evolution.**
Running a model over several rescaled Version of a test image followed by averaging the resulting class posteriors
when s is fixed. 
model evaluated over 3 test image sizes
Q= {5-32, S, S+32} 

Scale jittering at training time, networks applied to wider range of Scales at test time
S belongs to [Smin; Smax]
Q = { Smin, 0.5 (Smin + Smax), Smax}
better performance than single scale D and E deeper config performs best

  
- Multi-GPU training exploits data parallelism, and is carried out by splitting each batch of training images into several GPU batches, processed in parallel on each GPU. 
- After the GPU batch gradients are computed, they are averaged to obtain the gradient of the full batch. 
- Gradient computation is synchronous across the GPUs, so the result is exactly the same as when training on a single GPU.
The classification performance is evaluated using two measures: 
the top-1 and top-5 error.  
- top-1 is a multi-class classification error, i.e. the proportion of incorrectly classified images; 
- top-5 is the main evaluation criterion used in ILSVRC, and is computed as the proportion of images such that the ground-truth category is outside the top-5 predicted categories.
We also assess the complementarity of the two evaluation techniques by averaging their soft- max outputs. 
As can be seen, using multiple crops performs slightly better than dense evaluation, and the two approaches are indeed complementary, as their combination outperforms each of them. As noted above, we hypothesize that this is due to a different treatment of convolution boundary conditions