The data available can be divided into train/dev/test set

**Train set:** Data on which the model trains

**Dev set:** development set/Hold out cross validation set

Used to check efficiency of each model architecture and change hyperparameters accordingly

**Test set:** After a model with appropriate hyperparameters is selected, it is used to test the model, gives unbiased estimate of model

For small data, it can be divided as 60/20/20%

But for data in millions, few thousands is enough for dev and test data

So it can be 98/1/1% or 99.5/0.4/0.1%

**BAIS/VARIANCES: ![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.001.png)**

IN HIGH BIAS ERROR IS MORE IN TRAINING SET AND IN HIGH VARIANCE THERE IS A HIGH DIFFERENCE BETWEEN ERRORS OF TRAINING SET AND DEV SET.

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.002.png)

HIGH VARIANCE AND BIAS BOTH IS THE WORST CASE 

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.003.png)  

HERE THE MODEL IS LINEAR AND SHOWS HIGH BIAS FOR MOST OF THE PART BUT AT PLACES IT CURVES TO GIVE HIGH VARIANCE.

IF WE HAVE HIGH BIAS: bigger network/longer training might help

IF WE HAVE HIGH VARIANCE: more data/regularization may help

Earlier there was bias variance tradeoff(decreasing one increases other) but now with modern techniques,decreasing one wont affect the other much.

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.004.png)

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.005.png)

To reduce high variance we use regularization which does not increase bias much

To the loss fn we add **lambda/2m \* (sum of [w 2 norm]^2 for all layers)** where w2 square is sum of squares of elements of w. Adding b regularization is not very imp as most of parameters come from w

b does not affect much

**Lambda** is regularization (hyper)parameter found from running dev test on model,prevents overfitting **in dw lambda/m\*w is added.**

**L2 regularization** is also called weight decay as w is multiplied by (1-alpha\*lambda/m) 

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.006.png)

We use regularization as bigger values of lambda will decrease w values so effect of many hidden units decreases making the network work more like linear.

As w decreases z decreases and a=tanh(z) appears more linear so it cant take in complicated cases and hence reduces overfitting 

*Cost computation*

**L2\_regularization\_cost =(lambd/(2\*m))\*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3))**

**cost = cross\_entropy\_cost + L2\_regularization\_cost**

*Back prop*

**dWn = (1/m) \* np.dot(dZn, An-1.T) + (lambd/m) \* Wn**

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.007.png)

**DROP OUT(INVERTED)** another regularization technique where we nullify few hidden units randomly for each example differently so network gets much simpler

Else for different iterations we can randomly zero out hidden units  

We create a d3 matrix of shae of a3(3th layer) and randomly generate no and if its more than 0.8, make them 0.

This will zero out 20% hidden units and then element wise multiply d3 and a3 

Later z4 will be reduced to 0.8 times so to keep the same magnitude we divide by keep\_prob t=so z4 has same value

**We don't apply drop out to test/dev set**

*Forward prop*

**D1=np.random.rand(A1.shape[0],A1.shape[1])**

**D1=(D1<keep\_prob).astype(int)**

**A1=A1\*D1**

**A1=A1/keep\_prob**

*Back prop*

**dA2=dA2\*D2**

**dA2=dA2/keep\_prob**


![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.008.png)

**Drop out** randomly remove few inputs so the next layer node doesn't reply on specific feature and the w is more spread among all features and hence prevents overfitting 

For w with more parameters we keep keep\_prob lower and for w with less features we use high keep\_prob.using drop out the cost fn doesn't decrease monotonically. 

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.009.png)

**DATA AUGMENTATION** can reduce variance without paying for getting more training data, by flipping/zooming cat photos,you can double the training data and train model for more cases

The number 4 can be modified to train model for more cases. 

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.010.png)

**Early stopping** ,we plot error/cost fn for both training and test data and stop the training at a point where test set error is minimum

Advantage: no need to try many lambda values as in L2 regularization

Disadvantage: mixes two steps of optimizing cost and regularization so the w and b are not fully optimized

ML PROBLEMS ARE SOLVED FOLLOWING ORTHOGONALIZATION: following an order of tasks

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.011.png)

**To normalize input**,we subtract mean of all values to have 0 mean and then to reduce variance divide by standard deviation

We use same mu and sigma on test set

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.012.png)

If range of diff features is different, similar will be with w1 w2 and cost fn will be elongated and for gradient descent we would have to take small learning rate but for similar range,cost fn is circular and gradient descent is better.

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.013.png)

For deep networks ,we face the problem of vanishing/exploding gradients as if w is smaller or greater than one then whose values raised to L result in extremely small/large values affecting gradient descent.similar happens with derivatives of parameters.if yhat/derivatives gets very small gradient descent would work very slow. 

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.014.png)we can tune w to be more much greater or smaller than one using variance during initialization

For **relu sqrt(2/n[l-1)**  for **tanh sqrt(1/n[l-1)** as n[l-1] terms contribute to increasing z. 

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.015.png)

**GRADIENT CHECKING:** USED FOR DEBUGGING BACK PROP

Add all w and b into one big vector **theta** and dw and db in **d'theta** 

For each term of theta i,we iterate and find derivative on both sides using epsilon as small as 10^-7 . Now we check if dtheta is the correct derivative wrt J by using L2 norm of diff betw the two(theres no 2 over norm so we need to take sqrt) if diff is 10^-5 check i and there might be some big diff for particular i.if diff is greater than 10^-3 there is a serious bug.  

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.016.png)

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.017.png)

1 checking is slow so don't run it in every iteration of training

3\. Regularization term should also be encountered in d'theta or dw

4\. Dropout cost fn is complicated so turn off drop out run checking and then run drop out

5\. Initially w and b are close to 0 so model might raise diff for bigger values of w and b 

So run checking initially,iterate the training and then run checking again

**WEEK 2:** 

**Mini batch:** During one step of gradient descent we compute over full training set 

For millions of example, it gets very slow (batch gradient descent)

So we divide m examples into mini batches 

And compute gradient descent for each mini batch. This makes the learning faster

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.018.png)

Using mini batch gradient descent can be applied multiple times in a single iteration over a training set. The plot of cost fn wont be smooth as some set might be simpler and other may be more complex  

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.019.png)

**Size of mini batch:** m→ batch gradient descent : too long to execute

1 → stochastic gradient descent: no vectorization,too slow,zigzag fashion cost does not converge well towards minimum 

In between → fastest,vectorization

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.020.png)

Exponentially weighted (moving)average is distribution of data considering its behavior in the past **Vt=B(Vt-1)+(1-B)theta t** 

Vt averages over past 1/(1-B) data

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.021.png)

EMA: weight of previous values decreases exponentially

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.022.png)

Ema consumes less memory,only one var can be used for v of all values(overwritten) and is also computationally better as consists of only one line of code

As Vo=0, initially we don't get proper estimate of theta as V and values is low

So we use **BIAS CORRECTION as Vt/(1-B^t)** for big t it is negligible but for small t it improves Vt

**GRADIENT DESCENT WITH MOMENTUM:** speeds up gradient descent 

We use ema in gradient descent to reduce oscillations as it would take the average path moving horizontally and almost zero vertically 

*Momentum takes into account the past gradients to smooth out the update. The 'direction' of the previous gradients is stored in the variable v. Formally, this will be the exponentially weighted average of the gradient on previous steps. You can also think of v as the "velocity" of a ball rolling downhill, building up speed (and momentum) according to the direction of the gradient/slope of the hill.*


![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.023.png)

Instead of using dw we use Vdb for gradient descent. Vdw can be calculated by following two formulae. So now gradient descent has two hyperparameters alpha and beta and most common value of B is 0.9

The larger the momentum 𝛽 is, the smoother the update, because it takes the past gradients into account more. But if 𝛽 is too big, it could also smooth out the updates too much.

Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.


![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.024.png)

**RMS(ROOT MEAN SQUARE) PROP: Sdw** we first square the derivatives(element wise) and then divide dw by sqrt of Sdw during gradient descent.Derivatives with higher values will have higher S so divided by larger value to reduce oscillations in that dimension.we add a small number epsilon to avoid dividing dw/db by 0/near to 0 

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.025.png)

**ADAM(adaptive moment estimation)**  mixture of momentum and rmsprop algo

Alpha is hyperparameter and needs to be tuned,

` `**B1=0.9**(FIRST MOMENT)

` `**B2=0.999**(SECOND MOMENT)** 

**epsilon=10^-8**

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.026.png)

**LEARNING RATE DECAY:** as our cost approaches min we need to converge it rather than running zigzag,so we decrease the learning rate.

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.027.png) here alpha0 and decay rate are hyperparameters

Epoch: one complete pass over entire training set

Other learning rate decay formulae

1)exponential decay 3)discrete staircase,reducing alpha to half at instances 4)manual decay when model trains for hours/days 

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.028.png)

IN HIGHER DIMENSIONS CHANCES OF GETTING STUCK IN A LOCAL OPTIMA(MIN COST) IS VERY LESS AS ALL PARAMETERS WON'T HAVE THE SAME SHAPE 

ALTHOUGH SADDLE POINT IS COMMON WHERE LEARNING RATE DECREASES AND DERIVATIVE ARE CLOSE TO ZERO 

Here momentum rmsprop adam algos can be useful 

Adam trains a lot faster than gd and momentum.

Learning rate decay shouldn't be on each example as alpha will soon get close to zero so it should be on specific time intervals

**Learning \_rate=leaning\_rate0/(1+decay\_constant\*(math.floor(epoch\_num/time\_interval))** 

**WEEK 3**

**HYPERPARAMETER TUNING: alpha,beta,minibatch\_size,hidden units,layers,learning\_rate\_decay,beta1,beta2,epsilon** (as per the priority of tuning)

- Try random values rather than using grid(setting different values of hp2 for same value of hp2) as this would allow limited values of hp1
- Course to fine: check diff values and increase no of values near better results and focus on that region

For l,number of hidden units, linear search for hp is fine but for beta/alpha tuning we cant do linear search as changes are more sensitive in range close to 1 and 0 respectively

So we use log and exponent

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.029.png)

We need more values in smaller range so cant search uniformly(linearly)

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.030.png)

There are two approaches for tuning hyperparameters

1. Babysitting one model:less computational power,reviewing and changing model every day for weeks to find good choice of hp(large data/big models,online advertising/computer vision)
1. Training multiple models in parallel: more computational power,running many models simultaneously and checking which works better. 

To make training of w and b easier we apply normalization to input as well as hidden layers

Instead of a we normalize z but we don't necessarily want z to have mean 0 and variance 1 so then we add learnable parameters gamma and beta

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.031.png)

It can also be applied for mini batch,momentum,rmsprop,adam

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.032.png)

When we subtract mean from z the effect of b[l] is nullified sso we eliminate b

Batch norm makes learning faster.

During subsequent back prop the input of layer l is z[l]/a[l] which depends on w[l-1] which changes during back prop. So for layers to learn independent of changes in earlier layer batch norm keeps same mean and variance for input coming from earlier layers

It also has slight regularization effect(like dropout) as mean and standard deviation are noisy as they are calculated on different mini batches so doesnt let unit depend on specific input feature

Larger mini batch size will be lower regularization effect.

Can be used with dropout

During test,we dont have a set of examples so theres no point of calculating mean and std deviation of single value. So we use ema to store mean and std deviation of training set and use that

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.033.png)

Softmax regression is used to classify multiple classes,giving the probability of each class as an output.![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.034.png)

*In hardmax max number gets value 1 all else gets value 0*

the softmax identification function generalizes the logistic activation function to C classes rather than just two classes

![](\assets\deep learning course2\Aspose.Words.f8b84fb3-b60b-4005-a599-eefb87a16e11.035.png)

**DEEP LEARNING FRAMEWORKS:** helps you with abstraction and there's no need to write code from scratch.they ms be easy to develop and deploy,high speed and open source with good governance




