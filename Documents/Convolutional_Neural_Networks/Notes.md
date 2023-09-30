- Deep learning computer vision is helping self-driving cars, face recognition, and apps that show pictures of food, hotels, and scenery
- Examples of computer vision problems: image classification, object detection, and neural style transfer
- Challenge of computer vision problems: inputs can get very large, leading to too many parameters and difficulty preventing overfitting

![](https://lh4.googleusercontent.com/kjyUWkdKniQUbIqs0sDmSVz1NdXkfeoGmksiRIOJ-AtS8Dtqu5kjakJn50BvyagErW-IIQKj4kid4D3SYvq4rXf0Q7O-QacX3R3OoNmpcmbGSDtdNJ_Rk_pV6N_N0ZiqVvK0d1iAdY50sPIsKPog1Rc)

- Convolution operation is a fundamental building block of a convolutional neural network
- Edge detection is used as an example to show how the convolution operation works
- Early layers of a neural network detect edges, while later layers detect objects and face
- To detect edges in an image, a 6x6 grayscale image is used and a 3x3 matrix (filter) is constructed
- The convolution operation is denoted by an asterisk and the output is a 4x4 matrix
- To compute the first element of the 4x4 matrix, the 3x3 filter is pasted on top of the 3x3 region of the original input image and element wise product is taken
- The output of the convolution operator is interpreted as a 4x4 image
- This turns out to be a vertical edge detector
- To illustrate this, a simplified 6x6 image is used where the left half is 10 and the right half is 0
- When convolved with the 3x3 filter, a 4x4 matrix is obtained which is interpreted as an image with a strong vertical edge down the middle
- The convolution operation gives a convenient way to specify how to find vertical edges in an image
- The convolution operator is implemented in different programming languages using different functions
__![](https://lh5.googleusercontent.com/yxhFBbjBlWZGmJTkINOjU1DXUjBUobU7wCHMIWAySuCFnNxsF7twJ44lIPtRWzhsxrJ4xwGeX5PRGfQ96H5eejbjWX7aIPF__WySqcyhy2RaXidRaKFNcnKNR7HPay1eSZOcLgfe8Tz9AjHv3zDRabE)![](https://lh3.googleusercontent.com/ATNgk0RFQVo3_oROgIB2GDcyVkcrdH1uG_KC5HuSdtsgZOQcA4M7D6DwMyx-BF0eU8p_PmX0MegR7WMfwnl-hvQ0P_cuhKOn_vR6mEU4Wt4si3pKSZ3iyRuI7Lj4cPeywSmctsDb_ChYUBi097qmk4A)![](https://lh4.googleusercontent.com/lrhzukHuMCK038HSJc81Y6EtaHu17U7mi6cZfGbU_UlPE6HnT4_Mxj-B2VHEeWV50N1U_S7199eZotIHnL9a4--IGVvBus8Xr3JS61mRfkxnYss4pKLYrakl1jJzW7hh1-47qHtUuv9wontuGdFp1qI)

- Convolution operation allows for implementation of vertical edge detector
- Difference between positive and negative edges (light to dark vs dark to light)
- Examples of other types of edge detectors 
- Algorithm can learn edge detector rather than hand coding. Example of 6x6 image with light on left and dark on right, convolving with vertical edge detection filter results in detecting vertical edge down middle
- Example of image with colors flipped, convolving with same filter results in negative 30s
- Absolute values of output matrix can be taken if don't care which of two cases it is - 3x3 filter allows for detection of vertical and horizontal edges
- Example of more complex image with 10s in upper left and lower right corners, convolving with horizontal edge detector results in 30s and 30s
- Intermediate values (e.g. -10) reflect blending of positive and negative edges. Different filters can be used (e.g. Sobel, Scharr)
- With rise of deep learning, nine numbers of matrix can be treated as parameters and learned using back propagation
- Neural networks can learn low level features (e.g. edges) more robustly than computer vision researchers -Next two videos discuss padding and different strides for convolutions
- Padding is a modification to the basic convolutional operation used to build deep neural networks - Without padding, the output of a convolutional operation on a n by n image with an f by f filter is n minus f plus one by n minus f plus one
- Downsides to not using padding include shrinking output size and throwing away information from the edges of the image
- Padding adds an additional border of pixels around the image, usually with zeros
- The output size of a padded convolution is n plus 2p minus f plus one by n plus 2p minus f plus one, where p is the padding amount - Common choices for padding are valid convolutions (no padding) and same convolutions (padding to keep output size the same as input size)
- For same convolutions, the padding amount is f minus one over two, where f is the filter size - By convention, f is usually odd in computer vision
- Stride convolutions is a basic building block of convolution neural networks. Example of convolving a 7x7 image with a 3x3 filter using a stride of 2
- Output dimensions are governed by the formula n+2p-f/s, where n is the size of the image, f is the size of the filter, p is the padding, and s is the stride
- If the fraction is not an integer, round down to the nearest integer
- Cross-correlation is sometimes used instead of convolution, but in machine learning literature it is usually just called convolution

__![](https://lh5.googleusercontent.com/iuGzyngH23nhDUK7DM8rYKT-h9OFLYhSdQzU-HfLj2_Sh8wgZCbJ7lJKoDc0I2Pznit_TQz6dkKP3hHOQ0JECjo6x5XO7HiUCUFDYzh-V8HIwHk6F7nEqradNSNLS0AFzizj7-jDfU5AvYhA1bKsHRM)![](https://lh4.googleusercontent.com/g2bQfhiMLOp54qUFeXNLWkp0ezv6ZXqHD3UH2bnmjJ5xW12cqqKocMgqqVt0pVGnBuFenUmYLjwhKu5IqMdryXqeQMCRvE4KCsBXy9XC87Hqt8NCh1j98FS45vXjSgfavrxVcOxii_B7Zov89eHcknY)

- Convolutional operations can be applied to 3D volumes, such as RGB images, as opposed to 2D -A 3D filter is used, which has the same number of images channels as the image
- The output of the convolutional operation is a 4x4x1 image
- Different parameters can be used to detect different features in the image Multiple filters can be used to detect multiple features, resulting in a 4x4x2 output volume
- The number of channels in the output volume is equal to the number of filters used


__![](https://lh5.googleusercontent.com/LdKZfdh_f_edlvVSfNTQWsP6gZWRR6vlkxcXjmistgaL7X-RNkVcQxc0l-mrdl3Uc6euzhvwM_PFG-GdLC9FqHRhNjtCa-SGW5vkKXWhUKtmn_1fLUzVbTCFK4vXOZBoICvVEdreKPVkD4voPB2z864)![](https://lh4.googleusercontent.com/4Il1VEcmmZqOkcrt0IySakGAft05-huwCG4T6471qrDvf9qlpafHgU1h7HGyU1ztGxYfewTZ05NAAOEgpCpugO93Zr6oswztwmEh6isW7E6xokucT3J39Ifq0s64k4V0Xr70UXDHIF6u11vwjOo6sFY)![](https://lh4.googleusercontent.com/PFvMB2aFluZ2VYGGuzX96TR5gXfuI86OcGoLZKkKRW2BnrDM1IgYcOGb3tG6VYarmsP5t7K6dyjeQ1oTz_i0LHA2jdipc3UQg3jy4i-jwI2cj75mOXpfujTadaxOCqSY3SxV0iqBUZiIdj18VLdol4M)![](https://lh6.googleusercontent.com/Jm2xy1ARBPviH2_8rZCr6J1I8JMh5djDdM5ro4D3eVuSe4-RQDg01keQ3r1uRJOZycl13JmdQTVTgRP0PFOuUjuV4GG_FG4JDgzySi6lUQM4Hkzk_svrXbfkOXUPXxYquBgBf9zkSzveAnrIEdDfmYI)

- Convolutional neural networks are composed of layers, and this lesson focuses on one layer of a convolutional network - An example is given of two filters being used to convolve a 3D volume, resulting in two 4x4 outputs - To turn this into a convolutional neural net layer, a bias is added to each of the 16 elements and a non-linearity is applied - The output of the convolution operation is a 4x4 matrix, which plays a role similar to w1 times a0 in a non-convolutional neural network
- The bias is added and a non-linearity is applied, resulting in a 4x4x2 output
- This computation of going from a 6x6x3 to a 4x4x2 is one layer of a convolutional neural network
- If 10 filters are used instead of 2, the output volume will be 4x4x10 - The parameters of this layer are calculated as 28 parameters per filter, multiplied by the number of filters (10 in this example), resulting in 280 parameters
- The notation used to describe one layer of a convolutional neural network is: f[l] for the filter size, p[l] for the amount of padding, and s[l] for the stride
- One layer of a convolutional network is composed of an input volume, a set of filters, a bias parameter, and an output volume
- The size of the output volume is given by the formula n+2p-f over s + 1, where n is the size of the input volume, p is the padding, f is the filter size, and s is the stride
- The number of channels in the output volume is equal to the number of filters used in the layer -The size of each filter is f[l] by f[l] by nc[1-1], where nc[1-1] is the number of channels in the input volume - The output of the layer is the activations of the layer, al, which is a 3D volume of size nHI by nwl by ncl
- The weights of the layer are all of the filters put together, and have a dimension of f[l] by f[l] by nc[1-1] times the total number of filters
- The bias parameters are a vector of size nc[l]
- The ordering of the variables in the programming size is the index and the trailing examples first, followed by the three variables of height, width, and number of channels

  

__![](https://lh6.googleusercontent.com/oudLGkQc0FranWcAS_Y4Bo0QatUGUw2lMHPLwws4FQgcTANz2cvhkX_7ZDYx_cFNk1ftSBM-IVEjbuTbQTbIDyVoctlxbnMCPcfTMArjyihENTWpvPG7WvsAeB82WvmZzx5ckANVDwny-XTdaWK03Mc)![](https://lh6.googleusercontent.com/vyyrgZLcgQALqICvgnST2igCN_-JTF6wqqbyPzsjxOUS02Yd07gZ4iUVJVaIrmz6ptPX8XoHe_Uc1xEeR_mTcq9xSStVZ0qQ6DVNIIBz8w-1qlz3sJqJHTCWF6Xb34v5D2mw5iekTyA5KtOgWiCFWvE)

- A 39 x 39 x 3 image is used as an example for image classification
- The first layer uses 3 x 3 filters with a stride of 1 and 10 filters
- The activations in the next layer are 37 x 37 x 10 -The second layer uses 5 x 5 filters with a stride of 2 and 20 filters
- The output of this layer is 17 x 17 x 20
- The third layer uses 5 x 5 filters with a stride of 2 and 40 filters
- The output of this layer is 7 x7x40 -The 7 x 7 x 40 volume is flattened into a vector of 1,960 units - This vector is fed into a logistic regression or softmax unit to make a prediction -The hyperparameters of the network are chosen based on the size, stride, padding, and number of filters 
- There are three types of layers in a ConvNet: convolutional, pooling, and fully connected -The next video will discuss how to implement a pooling layer for the ConvNet -Training will be done using back propagation

__![](https://lh6.googleusercontent.com/YmW9k2WFWbByUcxOTm-8bKjplB3N-9e625sV61C7l93ppppaVXqISOCWPz36Z3reWAvHDRIDpyacQJLj2-p3NBibEWSCHnsWBvdLZamdKZI8sEcgVjgqydr4cFwXJjzhP2MQuQ9PC-YtB4KI2GLnXts)![](https://lh6.googleusercontent.com/abg3e6bqzxsyy5qZ0eDRxULwP7ZWdj9cspNeN_9HD19rQ2k5bwD9Hlk_dY_3KdHl2qj6elTHZaY2jMyfdCqemYBDACggWS5cQQa5s2LaSVUDUHZgsnSBKqbbcg0C9AVGQZMEd5ZIgXTqcKMyAphLtqA)![](https://lh5.googleusercontent.com/PtTVWcCu4uJl0Df9e8m7VHk-1BUIm2PFEgGUMbsjYX2Uc464NEaTnfMcMJiIdoxP04bPj5G_ck3-jZaIiZFAP1bMTImZjYH-P2LjSdLJhnOqoODj5SomtqZapbLVrhEPkbvPm4yVyQaTkcDk4zPIXKg)![](https://lh3.googleusercontent.com/8NSACZcPWL1N8LcBj-Pm9iFDJhGVLXQRfadHYcQ5Kk2VYOxIAo_DEu5yLxNUcclmgDV0t-cd2oqExQthjfeZbTHeen2-q7zWu7Gscdy5IciGRTXo0wmswUeCZ_hlSBUT_xp0_9kbsJL4bKcQJ3pTTw4)

- Pooling layers are used in ConvNets to reduce the size of the representation, speed up computation, and make features more robust -Max pooling is a type of pooling that takes a 4x4 input and produces a 2x2 output -The output is determined by taking the max of each 2x2 region of the input
- This is equivalent to a filter size of 2 and a stride of 2 - The intuition behind max pooling is that if a feature is detected anywhere in the filter, the output will remain high -Max pooling has hyperparameters (filter size and stride) but no parameters to learn
- An example of max pooling with a 5x5 input and a filter size of 3 and a stride of 1 is given, with the output size and each element of the output calculated
- Pooling layers are used to reduce the size of the input representation
- Max pooling is the most commonly used type of pooling, where the maximum value within a filter is taken
- Average pooling is less commonly used, where the average value within a filter is taken
(stride)
- Hyperparameters for pooling are f (filter size) and s - Common choices of parameters are f=2, s=2, which shrinks the height and width of the representation by a factor of two N_C
- Pooling is done independently on each of the channels -There are no parameters to learn in pooling, only hyperparameters to set -Input of max pooling is a volume of size N_H by N_W by N_C, and output is a volume of size N_H minus f over s by N W minus f over s by N_C

__![](https://lh3.googleusercontent.com/uFXUuYoGRwEqgs-z9IOHZEKfM3nM1uin3anoFRiAff2o0VDBIev2hBMxtlx5EM_cU_Ras2a38MwXCzoB-hl9z9vQK5zrwYOge9fTsR_ofPbl1s_K_6PL8VOgYfbMJxvWPK1xQldWscFmuQNPVB1VSnw)

- Course lesson is on building a full convolutional neural network (CNN)
- Example of inputting an image of 32 x 32 x 3 (RGB image) to recognize which of the 10 digits from 0-9 is present 
- Inspired by classic neural network LeNet-5 - First layer uses 5 x 5 filter, stride of 1, no padding, 6 filters, bias, non-linearity, output is 28 x 28 x 6 (Conv 1) 
- Second layer is max pooling with f=2, s=2, no padding. output is 14 x 14 x 6 (Pool 1)
- Third layer is convolutional layer with 5 x 5 filter, stride of 1, 10 filters, output is 10 x 10 x 10 (Conv 2)
- Fourth layer is max pooling with f-2, s=2, output is 5 x 5 x 10 (Pool 2)
- Fifth layer is convolutional layer with 5 x 5 filter, stride of 1, 16 filters, output is 5 x 5 x 16 (Conv 3) Sixth layer is max pooling with f=2, s=2, output is 5x5 x 16 (Pool 3)
- Seventh layer is flattening Pool 3 into 400 x 1 dimensional vector
- Building a convolutional neural network (CNN) with layers FC3 (120 units) and FC4 (84 units)
- Weight matrix W3 is 120 x 400 and bias parameter is 120-dimensional - Output is 84 real numbers that can be fed to a softmax unit
- Common pattern in neural networks is conv layers followed by pooling layer, then one or more conv layers followed by pooling layer, then fully connected layers, and then a softmax
- Activation shape of input is 32 x 30 x 3 (3072) -Max pooling layers have no parameters
- Conv layers have relatively few parameters -Fully connected layers have most parameters
- Activation size decreases gradually as you go deeper in the neural network

__![](https://lh5.googleusercontent.com/4M38eXHr8iNSojIBFBwudWY10IAxY3OmbSTjKd5_EFBmhfATo_ewpOZ6QbTqGKVt9WptuTJJv9QwRgeGBCp_eJmPsBwSE9qxTlN7d8AUgp0TBq855MDm8jOv689Dkn19Q7EpeKOp3pCKTOD27IHWMQI)![](https://lh5.googleusercontent.com/4xjdlf8bsVJ_3BWKLYYQ8RAwOChh5BUjSDSx9Kxd6Z1Er5rbvC7xSRtTlDiDWVlfopp-4_aOw5cuN-sHy8ztxRsMk_zSjX6ZzboRhs0vY0igeVZUPND822st6DicpxCyZsKAD_PjCqEbbKlXeizm25s)![](https://lh4.googleusercontent.com/Cl2-C52aH6OKKahdnGX8kJHNL64b6F0ORQe1N7GfeOVX0SZMhUcPxP5TTxR5Wysp0ICkmVYUPTQOf_fhSw9Z_0omITHVXbPAbvL4xiFOJvR8zd5vnpMtQKC9x3VKlHk0beP8-g6H0GmxDFb4ZIbcEMA)![](https://lh4.googleusercontent.com/jmLZM7H2o1SFSHOHMQg4u1D7Nv1X7ViSF9Ty-KXninLi-W1pYXwJDyigZRGfPoFw7T6PX_XLCxyVMt-z-TkyYS-Rh5Q0o8R-jHKunQoc_ee5ZCK8k0j33RdbRNFA9XUsJ1dq3U3TtmbZX4v7z24lvPA)

- Convolutional layers have two main advantages over fully connected layers: parameter sharing and sparsity of connections
- Parameter sharing is motivated by the observation feature detectors such as vertical edge detectors useful in different parts of an image
- Sparsity of connections means that each output depends only on a few input features
- This reduces the number of parameters in a convolutional layer, allowing it to be trained with smaller training sets and be less prone to overfitting - To train a convolutional neural network, one needs to use gradient descent or similar algorithms to optimize the parameters of the neural network to reduce the cost function
- Next week, more concrete examples of convolutional neural networks will be discussed, as well as other computer vision applications such as object detection and neural style transfer

__![](https://lh6.googleusercontent.com/uBZNrg9RlAjodrVwD-hSOVt5ycwX_PlUUwhAbBkMRPNXs67uoXkFJU6I3QODRkmt1eq4rBT2RjRK9dCRihNJtTtlK7wiDFx2FHrVF2JHmDskcDzfJS_wm1L1ISarKNkEUnhMqnz2sdpYCYN5593E3YU)![](https://lh6.googleusercontent.com/rk8H6DhNfoOQtLUF0SJ_c70lGqcmgEmZm2Z9lllHmXNCvn-TQf2O7tUPr6HURcVIhVKuW0a0iLRBhxETnc1biCEDLOOjzHOxeKSpfW4_2EYnn30mnZkHIftJR5YI1HaGGv1P_yZ4KDkz7oj-9XTj8P0)

- LeNet-5 architecture starts with an image of 32 by 32 by 1 and is used to recognize handwritten digits
- Uses 6 5 by 5 filters with a stride of one, reducing the image dimensions from 32 by 32 to 28 by 28
- Uses average pooling with a filter width of two and a stride of two, reducing the dimensions to 14 by 14 by 6 - Uses 16 5 by 5 filters, reducing the dimensions to 10 by 10
- Uses another pooling layer, reducing the dimensions to 5 by 5 by 16 (400 nodes)
- Fully connects each of the 400 nodes with 120 neurons
- Uses 84 features with one final output (ŷ) -ŷ takes on 10 possible values corresponding to recognising each of the digits from 0 to 9 -Has about 60,000 parameters
- As you go deeper in the network, the height and width tend to go down while the number of channels increases - Common arrangement of layers is one or more conu layers followed by pooling layer, one or more conu layers followed by a pooling layer, some fully connected layers and then the outputs
- AlexNet input starts with 227 by 227 by 3 images
- Uses a set of 96 11 by 11 filters with a stride of four, reducing the volume to 55 by 55
- Applies max pooling with a 3 by 3 filter, reducing the volume to 27 by 27 by 96
- Applies 5 by 5 same convolution, same padding, resulting in 27 by 27 by 276
- Applies 3 by 3, same convolution, resulting in 13 by 13 by 384 filters
- Applies 3 by 3, same convolution, resulting in 6 by 6 by 256
- Unrolls this into 9216 nodes -Has a few fully connected layers
- Uses a softmax layer with a 10 way classification output
- AlexNet was a neural network with 60 million parameters, compared to LeNet-5's 60,000 parameters - AlexNet used a ReLU activation function
- AlexNet used two GPUs to train, with a thoughtful way for the two GPUs to communicate
- AlexNet also used a Local Response Normalization (LRN) layer, which is not used much today - AlexNet was the paper that convinced the computer vision community to take a serious look at deep learning
- VGG-16 network simplified the neural network architecture by using conv-layers with 3x3 filters, stride of 1, and same padding, and max pooling layers with 2x2 filters and stride of 2
- VGG-16 has 16 layers with weights and 138 million parameters
- VGG-16 architecture is uniform, with a few conv-layers followed by a pooling layer, and the number of filters in the conv-layers doubling each time

__![](https://lh6.googleusercontent.com/XiXVpGHww7UgK-D3OIqJwJU6BmS74ivFc3uYpj1SsxfZ3YahTRQbtufOryAtvavv7krW0QykzNbEkBnY3DhK6Fi29CK2kS-JwSBV-OWPCkI75ZN6f-MxQSUmhGctyF_vy1-BHRAYuRs5jYcny04j7gg)

- Very deep neural networks are difficult to train due to vanishing and exploding gradient problems 
- Skip connections allow activations from one layer to be fed to another layer deeper in the neural network - ResNets are built out of residual blocks, which take the activation from one layer and add it to the output of the next layer before the ReLU non-linearity
- This allows information to flow from one layer to another without needing to go through all the steps in between
- ResNets enable the training of very deep networks, sometimes even networks of over 100 layers - The inventors of ResNet, Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun, found that using residual blocks allows for deeper neural networks to be trained

__![](https://lh5.googleusercontent.com/zwwH8X94DddQTbvkvX5J9jtatvEyZbXvv50UCdOHaD8elYGyvWritl4a0cwVbRoJn1f2K506Qo56Gnn4_TOh00zZ2OosXlPaSwiBkx6Mu5oMY3zDxSd8H_S3prZnPm82ovlFLPVXen1VSi8Q5tQnwZs)

- A one-by-one convolution is a type of content architecture used to design networks -It involves taking a 6x6x1 image and convolving it with a 1x1x1 filter, which multiplies each element by the number in the filter
- If the image is 6x6x32 instead of 6x6x1, the convolution with a 1x1x1 filter can do something more useful - It looks at each of the 36 positions in the image and takes the element-wise product between the 32 numbers in the image and the 32 numbers in the filter, then applies a ReLU nonlinearity
- This is like having one neuron taking 32 inputs and multiplying them by 32 weights, then applying a ReLU nonlinearity and outputting the corresponding result - If there are multiple filters, it is like having multiple units taking all the numbers in one slice and building them up into an output
- This can be used to shrink the height and width of a 28x28x192 volume by using 32 1x1 filters, or to keep the number of channels the same or even increase it
- It allows for a more complex function of the network by adding another layer, and can be used to help build up to the inception network

__![](https://lh4.googleusercontent.com/-RklU5sJNqL_YKOnRje8HpTrB9X2VbhVYu7WrHxuARRmI5E_K85SKVytZwQz6VM8ovHux-y-uLEYt8cbwxNb4upQ9s2FmSgaw9gMJhh9LGUNYTNad9JkZhxPf5-2U5uuYS4j9P59-PHin2D6MMobC64)

- Inception Network Motivation:Instead of picking one filter size or pooling layer, the
 inception network says to do them all - Example: 28x28x192 input volume, 1x1 convolution
 outputting 28x28x64, 3x3 convolution outputting 20x20x128, 5x5 convolution outputting 28x28x32, and pooling outputting 28x28x32
- To make the dimensions match, padding and stride of one for pooling is used
- Computational cost of 5x5 filter is 120 million multiplies
- Alternative architecture using 1x1 convolution reduces computational cost to 12.4 million multiplies
- Shrinking representation size does not hurt performance, but saves computation

__![](https://lh3.googleusercontent.com/T0CeLCEQjwXoMYvnwuNN2YY6dvGs4-DPZo_hdw8tgGuy2fwtI5p5TfQ60V7GfK2pKbPk4zyUhSe6OIRJq0Z1X4VQMwIvHwGEK2bSfDffHDjXBxyDS-vMSFGU0OA5KRXjbBcrSND2w7aUmlI19Z65J1k)![](https://lh4.googleusercontent.com/MOa1RRrF5ZCtEymVVr_XtX3qOg2BWTrS_KLgzKJ8vf_fK7vxHesTK9GGH29Lgge7ooEq5C7Pe0Ee5ND7FuyJziK70-507gGB9zlmvcjHBLEArukT3QDwtcpIIppKlg_VZYDJX48HGL3Itdjr2_axKCQ)![](https://lh6.googleusercontent.com/K4sHVxUZjSfSL8PIdkKG7Nlan3hIOUzemNojlSpRwDkfx5FbbdHSh74ftMQttXMZyIbcCt9wWcabZHGwSvf-bqlkmmlP1hz-oEWIINJf_0BPdBFPchxBM_IdWVeXxOD5jkdoZf8tAbomNdLWqNijGHo)

- Inception network is composed of building blocks from a previous video - Input is 28x28x192
- 1x1 convolution with 16 channels followed by 5x5 convolution with 32 channels
- 3x3 convolution with 28x28x128 output
- 1x1 convolution with 28x28x64 output
- Pooling layer with 28x28x192 output - 1x1 convolution with 28x28x32 output
- Concatenation of 64, 128, 32, and 32 channels for 28x28x256 output
- Inception network is composed of multiple inception modules
- Optional side branches for making predictions at intermediate layers
- Developed by Google and called GoogleNet
- Name comes from meme "We need to go deeper" 
- Variations of inception network such as Inception v2, v3, and v4 
- Skipped connections can be used to improve performance

__![](https://lh4.googleusercontent.com/ZPrKOMQLS8j_WZMT5N-el9TtcibI5OWlU0T72PzQ_NNt-fZEEZZOxvaP3Pb0M58FmwOjNq_n1EySEGkbZt_ykQYa6f8VB_GYXJiJ-34kO6K8M0UZtYhNfsXZB2iR2RVGGcGloEMkCP-NOupdF35xAvw)![](https://lh5.googleusercontent.com/tKOTbMoDCniSRqZuMyALVh3agy0s5Bf84ZjG_7xWZMBZE92jXYCoV7WPd6oVX9uJPSxUU2csOQUY8j-NnBvrI-30qWe9_LdRWh-8lTyxoUHitApfcL1mQb74m4VuujIsnoAcbPTy1MaBup2xt7lgYV4)

- MobileNets are a convolutional neural network architecture used for computer vision that can work in low compute environments, such as a mobile phone 
- Normal convolution involves taking a filter (f by f by n_c) and placing it over an input image (n by n by n_c) and carrying out 27 multiplications to compute the output (n out by n out)
- Depthwise separable convolution has two steps: depthwise convolution and pointwise convolution -Depthwise convolution involves taking a filter (f by f) and placing it over an input image (n by n by n_c) and carrying out 9 multiplications to compute the output (n out by n out by n_c prime)
- The computational cost of the normal convolution is 2,160, while the computational cost of the depthwise separable convolution is 432
- MobileNet is a convolutional neural network (CNN) that uses depthwise separable convolution as a building block.
- Depthwise separable convolution is composed of two steps: depthwise convolution and pointwise convolution. -In the example given, the input is a 6x6x3 matrix and the output is a 4x4x5 matrix. -The computational cost of the normal convolution is 2160 multiplications, while the depthwise separable convolution is 672 multiplications.
- This is a 31% savings in computational cost. -In general, the ratio of the cost of the depthwise separable convolution compared to the normal convolution is equal to 1/nc' + 1/f^2.
- The depthwise convolution is represented by a stack of 3 filters, even if nc is much larger.
- The pointwise convolution is represented by a pink set of filters.

__![](https://lh5.googleusercontent.com/-wCVMul6apIofZ6FiihHKAo_Ow9DhWvfhfVHQSOX9i4rgYyoMLD4pS5CdmkVZiETL-fHMIjwUaNHNC4UWcfsc18o3-BID99R47A3Bi8QzUQBkjoW2ocsgEDbs6uN7_yWB08qf0OpvsaLYfcZ6U3k_jM)![](https://lh3.googleusercontent.com/YRcii3D075VVnNbBXxj2ZReROEY46TL9KO95mVEEUVqozTCXxv1oF0cpvjlVDbn-5oIqQLdcpEDxKJpXMOB16aCS4cnjJMri-WOT5G9BSLgS9yUqJZje7FSvgW7AhlbVI4NI2EHj9pM5SBgbC5s7Qos)![](https://lh6.googleusercontent.com/AKG5S4ERqeWD9F7iz1TRrw-zNTTFaoxf2aV-f47sNoC7zDDE7jqmSJIa-c4J7KBmMBDxpab_CBUO_HOso0jDRRrPVp7UCTvEu_S5E4dvqWEbmK2xENWu9Z_15m2i-ruP2UHkojMQ_AA8jOKGx3nsbUc)![](https://lh3.googleusercontent.com/xbfcWHPLhvB186_9Hb01lF939D60UIhaHqyLVZAsGeiCqq-gdsBSDq5ySqj2FZuen76uiL4fDy03_4HwkKrEOgeRd9lHIq5kYRQkSh1YzzIwdHp00bL6Tzukc8PicV9gBKY7i39WhSC0gKzhAdeKV8o)![](https://lh3.googleusercontent.com/DANMMdUmTZiH_NRzFvseI-yAJ5_ce5fEWnPiZ2L7uYAkaKcqRWitK70o2lMQuc-DRb1klC81K60Uexm4oQzwjYs-6-4tFm5UZ-j8WL0yvG35h5Zi6vqwcp3MzQhMI2D_dFHNYCXBKIhpBnZsPnp8rNY)![](https://lh3.googleusercontent.com/FkOPUxjQvbd4RHUXGiTrq4tw55bvvtQgQJUIJJe8hY2J3zzTDvH2ut0JuDmqiIFmfcoZbn3mcXWW-8AUNL48bQY45UPwW-vTFCMp6WfqpjJCS7PUGFGOjO_feRIb4x4vKZgy3H6kkpuJqYOijasO0GM)

- MobileNet is a neural network architecture that uses. depthwise separable convolutional operations instead of expensive convolutional operations
- MobileNet v1 paper had a specific architecture with 13layers that went from the raw input image to making a classification prediction - MobileNet v2 has two main changes: addition of a residual connection and an expansion layer before the depthwise convolution
- MobileNet v2 uses a bottleneck block 17 times and then ends with the usual pooling, fully-connected, and softmax layers
- The bottleneck block increases the size of the representation within the block, allowing the neural network to learn a richer function, while also reducing the memory needed to store the values
- Efficient nets is another idea for building efficient neural networks.
- Given an input n x n x 3, the MobileNet v2 bottleneck will pass that input via the residual connection directly to the output, just like in the Resnet. Then in this main non-residual connection parts of the block, you'll first apply an expansion operator,  1 x 1 x n_c. In this case, 1 x 1 x three-dimensional filter. 
- A factor of expansion of six is quite typical in MobileNet v2 which is why your inputs goes from n x n x 3 to n x n x 18, and that's why we call it an expansion as well, it increases the dimension of this by a factor of 6
- the next step is then a depthwise separable convolution. With a little bit of padding, you can then go from n x n x 18 to the same dimension. 
- Finally, you apply a pointwise convolution, which in this case means convolving with a 
  1 x 1 x 18-dimensional filter.
- If you have 3 filters and 3 prime filters, then you end up with an output that is n x n x 3 because you have three such filters. 
- In this last step, we went from n x n x 18 down to n x n x three, and in the MobileNet v2 bottleneck block. This last step is also called a projection step because you're projecting down from n x n x 18 down to n x n x 3. 

_![](https://lh6.googleusercontent.com/Oz3eGHGZsNIDaFQUx4poBcbJX-5ftrRkvBwGphZgv1hQ28GdzNWr-QU_sjfPgznDIoqS0vz6DY5rtlj78i352DOYrg1tNGBXFsp-8AMIr3w4whrHLCELGjrO4sAp_RQxl0FMkid6sFuPrM4Wa-0mQ9c)![](https://lh3.googleusercontent.com/Qzd2cCL_6tg0XA3nlBstHLbw2Ok5nugzVpnlzFschNA8V_Lv4x_ZXXGQUkQl6DmCAtw3H6MqugUNaIGWPDmp5ts_ubfqomGxd9F7cUoflbAk89ZsOLn2UsohvtDhIkmrbfPbSJdej-AcNAReI1chc2k)

- MobileNet V1 and V2 gave a way to implement a neural network that is more computationally efficient 
- EfficientNet gives a way to automatically scale up or down neural networks for a particular device 
- EfficientNet observed that three things you could do to scale things up or down are: use a high resolution image, vary the depth of the neural network, and vary the width of the layers
- Given a particular computational budget, what is the good choice of r, d, and w? - Compound scaling can be used to simultaneously scale up or down the resolution of the image, and the depth, and the width of the neural network
- EfficientNet helps to choose a good trade-off between r, d, and w to scale up or down the neural network to get the best possible performance within the computational budget
- With MobileNet and EfficientNet, skills are needed to build neural networks for mobile devices, embedded devices, and other devices with limited computation and memory

![](https://lh3.googleusercontent.com/M6DDxUCi-BpdV-KqZP-KfIBRhtsouSc9LIkMKiiAd5hRUsU00E_pZDC3kIeNlVnlnU4nTM1qi3N8TtYctTpAbtvZC0q8OIiRG_bezHIXWOfjArOv0_mh3J_gUgBImeoEEKLuDQHvrpIfKcQ99dUFDqg)