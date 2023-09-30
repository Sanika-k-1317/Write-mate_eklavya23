- top 1 & top-5 error rates of 37.5% 17%. 60 million parameter, 6 50000 neuron 
- 5 convolution layer + max-pooling layer 3 fully connected layer with softmax
   1000-way
- to avoid reduce overfitting in fully- connected layers. dropout regularisation used

**Introduction**

- To learn about thousands of objects from millions of image, we need a model with large learning capacity
- Immense complexity of object recognition task doesnot help even when we have lots of inpul data CNN has much fewer connection & parameter
- 5 convolution & 3 fully- connected layes.

**dataset**
- 15 millim labeled high-resolution dataset with 22,000 category.
- 256x256 raw RGB values of pixels
- 8 learned layers → 5 convolution layer ; 3 fully connected

**ReLU Non-linearity**

- faster than tanh
- four-layer convolutional neural network with RelUs reaches a 25% training error rate on CIFAR-10 dataset. Six times faster than an equivalent network with tanh neurons (dashed line)
![](https://lh5.googleusercontent.com/x1-Z3XRYOrbFgNhcK21deh1YYZp-UAVaA8s8f4WFlotY2eV4YOQ3ZfUXTaN9sYUhwDB64ZWrnaJrDsU2dzIOQZ8uO4a6CTZWsaMG2fUacyTWi4SolXryrvAgquqXzJPQ1F8L4uVj7wPY1IzYmQqy4Xo)
  

**GPU's**
- To train 1.2 million training example in network, 2 GPU's are used. Current GPUs can perform GPU parallelization as they read from and write to one another's memory directly without going through host machine memory.
- half neurons on each GPU
- GPU communicate only in certain layers. kernel is layer 3 take at input from all kernel maps in layer 2.
- Kernel in layer 4 take input only from kernel in layer 3 belonging to same GPU. This precisely tunes amount of communication until it is acceptable fraction of amount of computation

**Local response Normalisation**

- K, n, a, B are hyper-parameters K = 2, n=5, 2=10-4 B = 0.75
- applied this normalisation after applying ReLU nonlinearity in certain layers.
- reduces type-1 error rate by 1.2% and type-5 error rate by 1.4%. 

**Overlapping pooling**
- Pooling layer in CNN summarise the output of neighbouring groups of neuron in same neuron map.
- consists of grid of pooling units spaced 5 pixels apart summarising neighbourhood of
   size zxz 
   s = z || traditional local pooling. 
   s<z || overlapping pooling.
- reduces top- 1 & top-5 error rates by 0.4% and 0.3%

**Overall architecture.**
8 layers with weights
- first 5 : convolutional layer.
- remaining 3 : fully connected
the output of last fully connected layer fed to 1000-way softmax, produces distribution. Over 1000 class labels.

![](https://lh4.googleusercontent.com/_pCuAsyYe46hu2BDQJVNzOpF8Y1KfBErW0oP6Hv2273pzRyLlsnBcji3F8YsiHBZbOtF36XwJbs1-yrFwK-_n74muzuvE-ZJfIP0IIO9xc_ugVi9O_SpX4HF5i4yo6Q2V7TMYvA9Xu4ZWE8Qw9L9EHQ)

Kemels of 2nd, 4th, 5th convolution layer connected only to kernels map in previous fully connected layer -4096 neuron layer residing in same GPU. Kernels of 3rd layer are connected to all kernel map in 2nd layer. Neurons in fully connected layer are connected to all neuron in previous layer

| layers    | Input               | filter  | stride | no  |
|:---       | :---:               | :---:   | :---:  | ---:|
| 1st layer | 224 X224X3          | 11x11x3 | 4      | 96     |
|2nd layer  | output of 1st layer | 5x5x48  |        | 256    |
| 3rd layer    | output of 2nd layer | 3x3x256 |        | 384    |
|4th layer     | output of 3rd layer | 3x3x192 |        | 384    |
|5th layer      | output of 4th layer | 3x3x192 |        | 256   |

**Reducing Over-fitting** 
or we will need to use smaller network.

1) **Data Augmentation**
     flipping an image, requires less computation transformed images are generated in Python. code on CPU while GPU is training previous batch of image.
     Image translation & horizontal reflections

     2nd form
     altering intensity of RGB channels in training images. Performing PCA on Set of RGB pixel values.
     to each train reduces top error nate by 1%.

2) **Dropout**
    Setting to zero the output of each hidden neuron with probability 0.5. Neurons which are dropped out donot contribute to forward. pass and in back propogation
    reduces complex co-adaption of neuron
    Learning.
    U(i+1) = 0.9 v(i) -0.0005 w(i)- dL /dw (wi, Di)
    w(i+1) = w(i) + U(i+1)
    i- iteration index; v is momentum variable average over ith batch.
    dL /dw (wi, Di) - average over ith batch
    Di - derivative of objective wrt w

**Results**
- Network has learned variety of frequency- and orientation selective kernels as well as various coloured blobs.
- Kernels of GPU 1 are largely colour-agnostic
- Kernels of GPU 2 are largely colour specific The specialization occurs every time independent of random weight initialisation

  --
  If two image produce feature activation vector with small euclidean separation, higher levels of neural network consider them to be similar.
   
**Details of learning**
- trained the models using stochastic gradient descent with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005. they found that this small amount of weight decay was important for the model to learn. Weight decay here is not merely a regularizer: it reduces the model's training error. The update rule for weight w was
   v(i+1) =0.9 . Vi -0.0005 .. wi- dL/dw (wi, Di)
   Wi+1= Wi + V(i+1)
   i- iteration index; v is momentum variable average over ith batch.
   dL /dw (wi, Di) - average over ith batch
   Di - derivative of objective wrt w


![](https://lh6.googleusercontent.com/e8OT895XWpgmlDNhNJSepoTyWufreMAjiWBFAvpWNDshgirK8uH7st-XDjQKcbNKSFIKpBHtdyd16UxsCabvKGw9BQagELLmPmtDsAf8NUM9mtsWbzQvVpo3y4UDE21jha3QZ7SDzXkPz_Ww7zkFmX4)

  
96 convolutional kernels of size 11x11x 3 learned by the first convolutional layer on the 224×224× 3 input images. The top 48 kernels were learned on GPU 1 while the bottom 48 kernels were learned on GPU 2.

- initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01. 
- initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1. 
- This initialization accelerates the early stages of learning by providing the ReLUs with positive inputs. 
- initialized the neuron biases in the remaining layers with the constant 0.
- used an equal learning rate for all layers, which was adjusted manually throughout training. The heuristic which was followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 