**WEEK 1**

**DEEP LEARNING:** BRANCH OF AI,TRAINING NEURAL NETWORKS

**NEURAL NETWORKS:** Depicts human brain,have many hidden layers between input and output, for training we give a set of input and output and the network  sets its hidden nodes functions.

In neural network each node is called as a **NEURON,** takes input from previous step and computes some function and gives output to further neurons

**ReLU:** rectified linear unit: rectified means taking minimum value as 0

Every input is connected to every neuron in the next hidden layer

**SUPERVISED LEARNING:** use of labeled datasets to train algorithms to classify data or predict outcomes accurately. As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.001.png)

Data can be structured or unstructured data(image,audio,text)

For different training algorithms 

Performance increases with more hidden layers/neurons/amount of data

FACTORS LEADING TO BETTER FUNCTIONING OF NEURAL NETWORKS:

1)Increased data:better training

2)Computational speed: With faster speed,correction and optimization becomes faster and productivity increases

3)Better algorithms: increase computational speed

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.002.png)

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.003.png)

**WEEK 2**

LOGISTIC REGRESSION IS AN ALGORITHM OF BINARY CLASSIFICATION

In binary classification output(Y) is either 1(yes) or 0(no)

When an image is an input,computer stores 3 matrices of size pixel vertical x horizontal (64 x 64) for red green and blue respectively

And as input image is represented as **x** ,a column matrix storing all values of RGB linearly hence **dimension of x will be 3x64x64(n)**

The training example number is represented as **m**

**And** X denotes matrix of n x m storing all input data for training

**X** as n x m is better than m x n

**Y** represents output of each training data and is a row matrix

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.004.png)


**y cap:** estimate/probability of x being a cat picture.

**PARAMETERS OF LOGISTIC REGRESSION: w: nx dimensional vector, b:single  real number.**

In linear regression y=w(transpose)x +b (y may not be between 0 to 1 here)

In logistic regression:  y=sigma(w(transpose)x +b) **sigmoid function** range of output is only 0 to 1.

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.005.png)

**Loss Function:error for a single training example**(measure how good the model is working)

**Cost function(J(w,b)): average of loss functions of all training examples.**

L=0.5\*(y cap -y)^2 (squared error) is not appropriate in logistic regression for optimization.(optimization curve becomes non convex,will have multiple local optima)

Hence loss fn formula with logs is used 

Here y is true label and y cap is prediction output

For y=1,0 y cap needs to be close to 1,0 for minimum error as per the formula

Cost fn is fn of w and b,the values of w and b are set to get minimum value of cost function

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.006.png)


**GRADIENT DESCENT ALGORITHM:** used to find values of w and b for which J(w,b) is minimum.

J(w,b) is a surface above plane of w and b.(convex function,has only one global optimum

W and b can be initialized with any value and will give the same result,normally initialized as 0

Gradient descent consists of iterations taking downward steps from initial value until the lowest point is reached.

Alpha: learning rate,controls how big step we take during each iteration


![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.007.png)

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.008.png)

Computations in NN are forward(finding output) and backward propagation(finding derivatives,gradient descends) 

**Computation graph** organizes the computation from left to right

Final Output value is found  by **forward propagation** (used for finding cost function)

**Backward propagation** is used to find derivative of output wrt to various variables of network  

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.009.png)

For  finding derivative wrt inner var of network chain rule is used

**LOGISTIC REGRESSION GRADIENT DESCENT:**

Considering the example of 1 set of data for dining derivatives

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.010.png)

For  training set examples:

Initially values of J,dw1,dw2,db(cummulators) are set to 0..

A for loop  is used to iterate over all examples and J,dw1,dw2,db are summed up by finding them for each case and later  they are divided by m to get average of all values

These are used to change values of w1,w2,b

This is one step of gradient descent consisting of two for loops one over m examples and for each example for loop over all input features(below shows example of 2 features)

Hence **VECTORIZATION** technique is used to eliminate these explicit for loops.

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.011.png)

IN LOGISTIC REGRESSION WE NEED TO COMPUTE VECTOR MULTIPLICATION OF w TRANSPOSE AND x ,IF WE USE FOR LOOPS IT WILL CONSUME A LOT OF TIME.

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.012.png)

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.013.png)

USING NUMPY FUNCTION (PARALLELISM) FOR DOT PRODUCT OF VECTORS IS AROUND 300 TIMES FASTER THAN USING FOR LOOP

NUMPY WOULD TAKE 1 MIN FOR TASK FOR WHICH FOR LOOP TAKES 5HR

THE CODE MUST USE AS LESS AS EXPLICIT FOR LOOP AS POSSIBLE 

BUILT IN FUNCTION(FASTER THAN FOR LOOPS):

For applying exponential fn to all elements of vector: 

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.014.png)

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.015.png)

Using vectorization we can initialize dw as a vector and  calculate dw=x.dz instead of calculating dw1,dw2,..... And hence the inner for loop for all features is eliminated

For eliminating the outer for loop over m training set we assemble all input x[nx,1] as columns of X[nx,m] and wT[1,nx] and then Z=[z1,z2,z3,.....,zm] 

So **Z=wT.X + b** here b is a real number but when added to matrix wT.X[1,m] python converts it into matrix b[1,m]=[b,b,b,b…..(m times)..] .This expanding of b as a [1,m] vector is **broadcasting**

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.016.png)

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.017.png)

Here forward propagation is vectorized

For backward propagation, dZ=A-Y where these are matrices[1,m] of dz,a,y for each training example.

also total dw can be calculated by multiplying X with transpose and dividing its values by m

db is the sum of dZ matrix elements divided by x. 

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.018.png)

We eliminated two for loops but we need for loops for iterating over number of times gradient descend needs to be performed.

**ITERATING OVER NO OF GRADIENT DESCEND CASES**

**Z=np.dot(wT,X) + b**

**A=sigmoid(Z)**

**dZ=A-Y**

**dw=1/m\* np.dot(X,dZT)**

**db=1/m\* np.sum(dZ)**

**b=b-alpha\*db**

**w=w-alpha\*dw**

**BROADCASTING:** technique python uses to resize matrices t  +/-/\*/divide elements of a matrix with another row/column  matrix or a real number

Here matrix A is initialized and then its column wise sum is calculated with axis=0

For row wise sum axis=1 has to be written

Then perc matrix is created by dividing A by cal

Here by broadcasting cal[1,4] is converted to [3,4] and then each element of matrix A is divided by each element of matrix cal(reshape was not needed as size is already [1,4])


![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.019.png)

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.020.png)

**SUGGESTIONS:** 

Below 1st initialization is a rank 1 array(neither row nor column vector) and can cause error at places,hence avoid using rank 1 arrays

Instead use initialization 2,3 showing column and row vectors respectively

a.shape shows the size/shape of a given vector.

assert statements can be used to verify that given v is a column/row vector

Also rank 1 arrays can be converted into row/column vectors using resize function. 

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.021.png)

Practical implementation of logistic regression

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.022.png)



**WEEK 3**

Neural network is a repetition of logistic regressions,output of first acts as input for second calculation

Layer number is written as [i] in superscript and example number is written as (i)

Subscript number represents feature(node) number.

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.023.png)

**a denotes activation which is a set of values that each layer passes to the next layer**

Input layer: zeroth layer 

Hidden layer after input: first layer

Each layer has its own weights,baises. Dimension of w is [i,j]

i represents no of features(nodes) of the layer and j represents the no of features of previous layer, b has dimensions [i,1]

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.024.png)

Rather than using for loop for all hidden units of a layer,we can stack them into vectors

W[1] vector consists of wT vectors of all nodes as rows, b[1] is a column vector with all biases of nos as rows. X[n,1] consists of all input features.Z[1],a[1] are [i,1] vectors of z and a for all nodes of layer 1. Now a[1] acts as X for layer 2 with W[2] and b[2]giving a[2] as output.![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.025.png)

For m training examples, X is [n,m] and Z[1] and A[1] are also rectangular matrices.

Like logistic regression instead of looping over training set we use vectorization   

For m training examples

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.026.png)

**ACTIVATION FUNCTIONS:** function converting z to a in general **a=g(z)**

Examples: sigmoid function,tanh fn

**Tanh fn is better than sigmoid fn** as it centralizes the mean of activation around 0 rather than 0.5,makes it better for next layer to learn

Activation function can be different for different layers

Incase of final binary classification layer,sigmoid is better than tanh as it gived value from (0,1)

**ReLU:rectified linear unit is a=max(0,z)** is popularly used activation function

For tanh,sigmoid the slope is marginal for very small or large values of z and hence learning becomes slow,hence use ReLU

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.027.png)

Linear identity activation fn: a=g(z)=z 

Composition of two linear fn is another linear fn so using linear fn for activation in the hidden layer is useless But activation fn of final output can be linear if we want output to not be [0,1] but a Real number.(in that case ReLU can also be useful)

**Derivatives(da/dz) : sigmoid=a(1-a)**

**Tanh = 1-a^2             ReLU                     leaky ReLU**

`                                   `**=0 (z<0)                 =0.01(z<0)**

`                                   `**=1(z>0)                  =1(z>0)**



**GRADIENT DESCENT:**

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.028.png)

**In neural network we can't initialize W as zero vectors:** as they will contribute same fn to next layer and any every iteration of gradient descent they will still have same values row wise so having multiple hidden layers will be meaningless

We use **W1=np.random.rand((n[1],n[2]))\*0.01**

We multiply w by 0.01 as we dont want very big values of z as they will result in less derivative and slow learning for sigmoid/tanh fn.

**b can be initialized to 0: b=np.zeros((n[1],0))**

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.029.png)

**WEEK 4:**

**L**: number of layers in a neutral network

n[l]: number of hidden units in l th layer

a[l]: activation of l th layer similarly we have z[l],W[l],b[l]

VECTORIZING FORWARD PROPAGATION:

**Z[l]=W[l].A[l-1]+b[l]**

**A[l]=g[l](Z[l])**

**W[l]=[ n[l],n[l-1] ]  b[l]=[ n[l],1 ]**

**dW and db have same dimensions as W and b**

**A and Z have same dimensions**

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.030.png)

` `**WHY USED DEEP NEURAL NETWORKS**

Deep neural networks break the problem into several layers replicating human brain first finding basic level intuitions and later complex intuitions 

Also total number of hidden units required for shallow is exponentially greater than deep ones for particular problems 

**FORWARD AND BACKWARD PROPAGATION(ONE ITERATION) OF DEEP NET:**

A[l-1] is passed as input to each layer and a[l] is calculated and z,w,b[l] are stored in cache and during back prop da[l] is input and da[l-1] is calculated giving dz,dw,db[l] in each step.

![](/assets/Aspose.Words.e8122f05-957e-40b9-9220-9676a5c95191.031.png)

**COMPUTATIONS FOR BACK PROPAGATION**

**dZ[l]= dA[l]\*g`[l](Z[l])** element wise multiplication

**dW[l]=(1/m)\*dZ[l].A[l-1].T**

**db[l]=(1/m)\*np.sum(dZ[l],axis=1,keepdims=True)**

**dA[l-1]=W[l].T.dZ[l]**

**HYPERPARAMETERS** control the parameters of net

**learning rate,iterations , hidden layers/units,activation fn**

Values of hyperparameters need to be found by trying many values over the time,no direct method.

*Deep learning is said to be analogous to human brain but how a neuron in brain computes a decision and how it learns is still uncertain, receiving electrical signals from other neurons and sending signals ahead.*














