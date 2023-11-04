**Binary classification**
- looks at an input and predicts which of two possible classes it belongs to.
- In it we learn how a classifier classify an image represented by feature vector and predict if the image is cat image or not (only two possible output)

 **Notations**
 ![](https://lh5.googleusercontent.com/q5ZGLw9pMX0gQvjBiZ_pU0F3cyLcPBfDbR6QCzasi6qUe_bpS8thH3iyF5yycdTueX-NX5Qj0Ru5k0AmI_Igq-T3OH6zTSA20gx1daKcCWtlf55AV0sMlpGizj-sxYwmkfltHbyApg41vFCFnM-WAEg) 
**Logistic Regression**

![](https://lh3.googleusercontent.com/DXouptxfIXGWgyHqu8r5HjrD8jW-GR6hSc9uxW-1iGbdiDv4jv5oQ0lRHMpE5M7TjWY2tus8u2jcfntdSsK8zb3cFy4rX7awstb2OzdUTekTGljBpRc9MJOF6u18e_6VIOUWsa19tiFnxsacjC65Bms)

Sigmoid function is applied to bring value of y between 0 and 1. it converts the linear function into non-linear function.

**Error (lost) function** defined to measure how good our output is

![](https://lh6.googleusercontent.com/cwxSbQ6y6N9XkHFS7YDn65WB73FZpwgrpQoRTlY_w8NKcEh1rgCGif1oake1Iyaim0TgE44xXYfqen1HXNC0Nj8p0WWx_hrSvvp4eouZk9CpmNA_mqWWu2DU0SS17x0U-GfWesI3iSq8YmkcxggFiAg)

**Loss function** computes error for a single training example; **cost function** is average of the loss function of the entire training set

**Gradient function** aims to find local minimum of a function
- When we know the gradient of function we also know the correct step to take in what direction ,the amount by which you want to update w, b
- A convex function always has one local optima
![](https://lh3.googleusercontent.com/WeGcB3558dF_62m_qkWNm1wcLA0M5KlUnMO7yXw056a2wKyiW1m_X8k5x8KjcZl5ogTbKDUYQJImBl0EjiJnzbpHohloMSnPNzH57P2QgbDGAz0oey21SPFR-u0268vJkI7zh8ro2YGT_AC2BJcHZLo)

A computation graph the computation with blue arrowed left to right computation
- One step of backward propagation on a computational graph yields derivative of final output variable
- The coding convention dvar represents the derivative of a final output variable with respect to various intermediate quantities.
- A-Y is the simplified formula for the derivative of the loss function with respect to z (A is the final output and Y is the correct output value of function)
![](https://lh6.googleusercontent.com/wPIe53tGwKJ2vG2WgE3fQMJEuu4yZT05YhPMiBTBe0MFRnoYtPWGyO8htHhxQJDBmX3FTM_3G4jVnFXhcGuf8sNodwdJsnQ7YGBFQKjjG0dMLBb_kUk6Tod7C3O1vNJdWf9nA8vp5olNnH6UgLN8Rbk)

Logistic regression on m examples
![](https://lh3.googleusercontent.com/4EHm9F4FfOEnOwVIlT08RiBV54asyraWfbePlqXOhYNLe3jg0qJEulUIksobCf9h6U-XAQYA3Z-nYD12dAe4eWIM7kPs9f3kP-kll5-A_rT8YSCnaX1URRJP1OD-NSu3q6pDYS2yVH6KITTfDURYCv8)

In this the dw code is cumulative 
![[Pasted image 20230911104316.png]]

we would move to a bigger datasets, and so being able to implement the algorithms without using explicit for loops is important and will help to scale to much bigger datasets.

**Vectorisation** is used for the process.
![](https://lh6.googleusercontent.com/ZDdq-lsQGBWfuiO6cPdvKKCNDSATGuB2b3joe8PxkIzENlIlLllPj8oViW0cG5sVZ56sfehnVrIyz9zudDn3jciIxUMkA5HHK3YKWUPBoAUarQhjoN4a4itMy0yewAvZIYUl5viQMgbFos7CAHsKHxg)

There are a set of techniques called vectorization techniques that allow you to get rid of these explicit for-loops in your code

![](https://lh3.googleusercontent.com/GadzjkI0g3qDIOMyRbiwbwzqhq3It3QZXnlYDIRqGB5_lXs77wXKnQbu6iXnCFdW_4hkNTrLC3Qp6Lre_g4ZZCVV-sDccLnjtBXfdsVflDWBoa5Jd1Qita-raNh_n2TonRmBfbSRjm0gmKT2AaO3NKk)

Tricks on how to convert loops into vectors

![](https://lh5.googleusercontent.com/ZXEBS5trfK6z5kg0zsfAWHSB-uq1XmVjll4ECylefdU5QzpYVwXkzUth61jZNa3urPKTC-0FWpUAB-Xxh5ISkKrdEvmNhTBCEoQREejEOdbp4fJsRyCvKb-mbyG7WwUK__1Oz7NOXIMZqd3D6ANV3e8)

Removing for loops in logistic regression code
![](https://lh6.googleusercontent.com/fv7u9uw7YZRafsacFc90EM1KHF5qAv1QOJnqayPQ0sCKTtKbc2IpoPSgmidG-b_z9RErmQgpkOrF-RM5g_GoByrPhSIXvBX9c78XToMast3Qosv8LQoWpefgMB6r0ZDWixn9OQm6ybY0hY5nfQeyRTQ)

on this slide is that instead of needing to loop over M training examples to compute z and lowercase A, you can implement the one line of code, to compute all  Z's at the same time.

![](https://lh5.googleusercontent.com/GBFr34-yMTvtZtrVmNsZpUtgsYqVskAUBKTJiompZjM_pz17aTkGyFuxRCp81jNrCpX44ML43MCZ46Qz-iHRRNONwW5UnzIheJVvzyGDTizvj3pWZCRsvQzZ2zJU65zEehFy7furo2dq8E3GeoL3NJc)

  
if you want to have a thousand iterations of gradient descent, you might still need a for loop over the iteration number. There is an outermost for loop like that then there is no way to get rid of that for loop.

**Broadcasting**

![](https://lh3.googleusercontent.com/tHV-o_7f48rxzdI_5rvbdrR28-NPW50qNu3txVuA1x0y-SxjwDfK4YOfKMkzBBIrHeBlvBKktFSO8-Vcez9kxWu7-tLe3KqsEoXxwHBwlrhNRrlF1kQ_hxzf4cBPjC9btXzEg6A4fqMqCWWXE4AZYuc)

**General suggestions**

![](https://lh4.googleusercontent.com/kkrkAX1qHsjhe3n-YQyMIJynRnd0dJEj5UEieov4RW5_L8EnOVakdolsiAwnyMdAOOiO3OKxfcSdsXl7-k2e5OhNjGvxfBkYDJGwpwsFBoeuaYlrHqBO-mHm7sgX9Mg2vjnygl8wjzBoEnoAoIUhNfA)

to simplify your code, don't use rank 1 arrays. Always use either 
(n,1) matrices - column vectors
(1,n) matrices - row vectors.

Week 3

![](https://lh6.googleusercontent.com/fRWAC7R1Dc1mKkznfLmL_wUOIm90V31MSEPvDnf02TbqJakCPrv_y4_aotja8s3v4iaw4D5AWsOjLdNXeP8Bsmn8AkPZGzH8B1pS4WpKRmOTd86yDwFH91LI9rcazuOW6Y-au1idrJu-9pp4n2ter7c)

- superscripts [1] to indicate that these are parameters associated with layer one with the hidden layer. w will be a 4 by 3 matrix and b will be a 4 by 1 vector in this
- Where the first coordinate 4 means we have 4 nodes of our hidden units and a layer, and 3 means have 3 input features.
![](https://lh3.googleusercontent.com/pGoYnP5KODHDEVl7xw5yHqAOm996cfr0zwmbKXYDRMbpWRmAXaJ5REV4w7BZYmjNdutv_h58hFU34n9wxreJYwzGnOhPqrWS1mZtdpSkvz-sAfT_KkdqZwO53pNLu-5hFI-xTwk_wSLlArqGCVrwauI)


**Neural Network and Deep Learning how to build and get to work**
To predict the price of housing we draw a straight line through readings. Such line is called ReLU function that is rectified linear unit

![](https://lh5.googleusercontent.com/u_jmPgfAyfgnyTEuYT1aKf1lHVQ4glVPOl9Tao5zZzmbib_Uxz7_bWT0KkzZi6C752jy29hB0AA5oW0sIqaBSU_8Y6mYikQYKps-VpWga9d7uRRus9NL6ArJkMiSR1h8KYhvT00bfYSm4Nd07s5Z3bM)

The structure consists of neurons that processes the given data and gives out output
They consists of hidden layers of neurons which are connected to every input information

**Supervised learning**
![[Pasted image 20230911122316.png]]
We have an output x and we make a function to get output y. 
**RNN** are recurrent neural network, very useful for 1 D sequence data that has a temporal component **CNN** used for image data

**Structured data database of data**
![](https://lh4.googleusercontent.com/P7-68qxgmEzmO2GNf0eQ1LGM_aeqi4iLKD3BqwCcQmLvM6g9g8EPJAHIyY3ardq6ysJzl8SPC51OEptlE0e5ms1rqXg_JYZRkzPpV-0SonZ9hGxiG52ZPM3BCUybbc7aoi2rIgLOz1MLtQ2-CUZpfJA)

Week 4

![](https://lh4.googleusercontent.com/3iW95_58pBRFtdtpcw34bhHA7FbaD0H08KzHuy1rCdYBk9mowsRuYD81wYR3uwsWl2FhnhU7eH7FFDIJxE7coCMgeUwY1dH8Gl28C7N5gnqJFuCDcZ-Dul72bJhj5XA0UGFWF3jlbSMhdesOFQqPVxI)

  

![](https://lh6.googleusercontent.com/RyHDSS1HVOvDsIGxupWiV2NzZogtAUqwuYCcP3gfnK_y0tk1EjozHO1SWfG6J6cNZUVppSsZOTTkKeHuYw4yXQ0JICjmyGgFfbBwDufQlOmr-_7zqmJ8p4x4ofqiOPF63DBLxq3T63U6qVksT0WVbmI)

  

![](https://lh6.googleusercontent.com/gVOJ-j9J_48UwcrXVgLYsOl4dRl4s9w1Q_bL6_3o5RtYcu54yEJOVSkmdPwKt7qdZygkMaGwtH36fodCc89I_qLmdh9mOgGrNiHM2nIsctu5Ca2la4KMGt38Z4PRdLFqCxF7hJE6UvTPEyMhaB-JTEc)

When you implement back propagation for your neural network, you need to either compute the slope or the derivative of the activation functions.

In Sigmoid activation function for any given value of z, this function will have some slope or some derivative corresponding to it
So, if g(z) is the sigmoid function, then the slope of the function is dg(z)/dz, and so we know from calculus that it is the slope of g'(x) at z.

While summing horizontally we use keepdims, and what keepdims does is, it prevents Python from outputting one of those rank one arrays. Where the dimensions was (N,). So, by having keepdims equals true, this ensures that Python outputs for DB a vector that is (N,1).

![](https://lh5.googleusercontent.com/49ydEZq_56Ei8ng-2SVTitaIod9uPpOp31_s_tM2nFxVAe6FNpcXjs7vCjGUQpiOhfRtp8cKAwgB1G-1hu6b2nTH1rUFwv9X5RfJNh5mB0b03Q7cgAva1qaHjJt0olfaR8dmGN317-5HocdWNW1p-8E)

But now as you continue to run back propagation, you will compute this, dZ2 * g'(Z1).
This quantity g'() is the derivative of whether it was the activation function you use for the hidden layer, and for the output layer.

![](https://lh6.googleusercontent.com/mXqSO_BXfP-P-OnLsYZxTwfvqHU5l5q6HZkFWFbNwLVIM9LI9UT85NnW15Pv9MDrdcyxZa1NAt7BUbZ0dbcflFg42tqaFo3sotmL8HcOKrilRC0S8XPxtS203h147UsPt2lcEXcpNW_dEv47SDuifMs)

It turns out initializing the bias terms b to 0 is okay, but initializing w to all 0's is a problem.
The activation will be the same of two neurons eg 11,12, because both of these hidden units are computing exactly the same function. And then, when you compute back propagation, it turns out that dz(11) and dz(12) will also be the same colored by symmetry.

![](https://lh3.googleusercontent.com/FDAM-8t7dsyHSgj6IkVGJayjTvOuR2i76EOJ9j5c9B3zlH0emCcH8GTnw694dSz7FZSk_IAUnkMAElHUskOJK0qLQyKCQhcyTsIfR3VkUGYIfEK9retytTx4IjuYvWYu-BM8B1IhFAIoVAuFPwbFZBg) 

- We perform a weight update.
    w1=w1-alpha *dw.
- we usually prefer to initialize the weights to very small random values.Because if w is very big, z will either be very large or very small. And so in that case, we more likely to end up at  fat parts of the tanh function or the sigmoid function, where the slope or the gradient is very small. So gradient descent will be very slow. So learning was very slow. 

  

![](https://lh3.googleusercontent.com/aVtDzpSBaBF1E_4hrd8WVh7PyrGc74wINT6-zRCfwkvmqFEsMFGEG5sE8auOAShfzXfBzU9qZAYxuPsXyjS2O_aVDfn-nY24mLXUqpQt3v3XtRl73ZKrW3XLC4RNnBSJVuPt6HzAAmwVZxX9rzqgdHQ)![](https://lh5.googleusercontent.com/mJKbs8B1Sdc1EwrlnJouxKUXp1iFLzB-yxQ5qWbCWHma_TLlxcKBVBfNEW4Qj75w_SsT7DLmr_oZmeK1oJkyRk7mlMRA6aO_7oLy-Z_4KwHLGE00sa0WoJkkDOAC_1sGTHEq-gCur6oiN5ksy3hJlAo)![](https://lh5.googleusercontent.com/23vJEOReVJl5bNY-cd4dP2LIwZgG9NR-AkSfHvnSXlRqGs67FpXd8M_2hzCvOEiFUYY2qJiZM1lQEkq9CXix3n_8uys5lkAr64hyx_yvhblcfP_xEuxPtOX-59mkQt__4HPTtZgSJv37AlrVClELJ-A)

- Then you have to compute the activations for layers for which we use for loops.
- All the training examples are column vectors stacked left to right. 
- When implementing back-propagation, dimensions of dw should be the same dimension as w, and db should be the same dimension as b.

![](https://lh3.googleusercontent.com/Brb8W01UoLklZsqcbFCgx6Sm_2MnG99zOcX4oBwUOgnX7yfW8bIY_JFm7nIQVf9n4lTCxQb21RUdDG-t2FriYQ9JpGcGw0Vxyfj1Q-HLRXiBB8L0X7C9RwjwLhUUGmTwbwvjEEB4yWh3G68iE9wsLl8)
- Even for a vectorized implementation the dimensions of w, b, dw, and db will stay the same.

![](https://lh4.googleusercontent.com/XDKmLpfnQY5fYa-UlIK9jTm53zUNpDb7w9DKWbRnmSLFlKb3Glc7II2Fe29tE_YC3EMrKlw3D1_uyqhFP-EetG6fqnnWXoK8BdX5oGF9VC2B3kJuO3mG6zAm0Piio-Mun5ukFmWxV-xUcrM_vieIWs8)
![](https://lh4.googleusercontent.com/0UPVvvJINXhB-_YAZ5wrFpi7HrtXtMVISaZw9JoMTWCcd_XroxdkneUIgYMAIMTveP07G0BIGgxwuio85wwHxdCK5FjnmwdVdKodEzdB2H3oPZTlnl9HkRcik2adWAdh9cOnSNrjPV__T4A5_YmSaWY)

On left hand side
- When building an XOR tree ,  it computes the XOR of X1 and X2, then takes X3 and X4 and computes their XOR.
- When using AND or NOT gate, we need a couple layers to compute the XOR function rather than just one layer, but with a relatively small circuit, we compute the XOR, and so on.
- To compute XOR, the depth of the network will be on the order of log N.
- if you are not allowed to use a neural network with multiple hidden layers with,  so you have all these things going into the hidden units. And these things then output Y.
- Then in order to compute this XOR function, this hidden layer will need to be exponentially large, because essentially, you need to exhaustively enumerate 2^N possible configurations.
- There are mathematical functions that are much easier to compute with deep networks than with shallow networks.
- When starting out a new problem, start out with even logistic regression then try something with one or two hidden layers and use that as a use that as a parameter or hyper parameter that you tune in order to try to find the right depth for your neural network.
![](https://lh6.googleusercontent.com/4mPNkIcDE_CGfzoPNDsm2CmpybmRZZ4dHKKfiieD1s7exfVohdm_SsCheRma78m6xcX1YCZ0qpIE8TDB1oYKWICB8VGQK0OOsCe3vJ-aRxvVLNvrwlc2lXTdAvN9MBo_KImAqzXJN2aoKNTqVXOTphk)
![](https://lh3.googleusercontent.com/2It32R_LnToUay5V44LzN6lzGaG_DyhZ7dGiUfpNzAUsYZhMRc_kYfjiWE3lGUvqZQLzEsfH16KP2QWD7Wm_dMm76gUqkYg_S4JrH5dpvq7AMKfeTmNerg98UZT_f8qrzgVUegb7gIygtkNIjL8EApY)- We take the input features a[0], feed that in, and that would compute the activations of the first layer, a[l1] and to do that, you need w[1] and b[1] and cache away z[1]
- Now having done that, we feed that to the second layer and then using w[2] and b[2], we compute deactivations in the next layer a[2] and so on. Until eventually, we outputting a[l] which is equal to y-hat. And along the way, we cache all of these values z.
- That's the forward propagation step. Now, for the back propagation step, what we're we do will be a backward sequence of iterations in which you move backwards and compute gradients .
- We feed in here, da[i] and then the box will give us da[l- 1] and so on until we get da[2] ,da[1] inside these boxes we end up computing the dz's as well.
- We store z in cache as well as w and b. So this stores z2, w2, b2. But from an implementation standpoint, it is a convenient way to copy parameters to where you need to use them later when you're computing back propagation.

![](https://lh5.googleusercontent.com/437fIUhRZEvS4DdEyBYNdnpj9WXSLYby8axSpk4TPzILoQxDXhDqLFleEmn_Wt4EKMnldMBZsQbaPy7DT00A7-_k_pdKZludeykE3Sw3pomiFasEQkA9ktjw2QjYewd5MzVwoSotn8WcpnYwf9pFTt4)![](https://lh6.googleusercontent.com/vdxhy8B3UZ1r2uO3bBWt6zYyQd8ubSgi3jzrzXo5kyoyJTdCFw80s7uhTete-opnUX2yvEwwqhkG8rW8QTmcdO4t8uADHyjppsOG87HiH9TdUvAyM7bEducJeZ_7SujQSr1OxjWYf9qxtBSf2pFc98w)
![](https://lh3.googleusercontent.com/00hQPGClZFnl7nex-VUDFzErWijwPiYin6aeipjSwur-0bJAGy4FsT9xpa2MmSSbVdjGUj8HUZvuVc4vYL4o2EGTn58Ow_avcjG0a_ssiruQ92fLFSdVdn0TBmTgfnx6F9Xvd5TjWn1BpMdEb1KG7lo)
![](https://lh6.googleusercontent.com/p7MsafiAvWxlPhds1Xu8KQMWrmjcm2b7W19Z12CiCd6pCRsnXtKZmvKabVMJQ6OvsVZsJgEVOnbuvxgIlfEDk8gBflwpGfSJkJ3QH4waOY6wRM3JrxF7udprxnGfmt3zTnCHboRYycwDle420mqCDyA)
![](https://lh6.googleusercontent.com/WNuzXLKrNRh8U1mOfcNNOI1VZl1KEPpqq9cu8616HaNXjFrJ7Dmndtt6vQkGgFbdTu8V8GlKf0j6IOco1BwRaVdDa-xEvYWCYH5kIJf6Xh3Qd1aQN7RSR-qKrBdZgkPL20m8APsx0_GqR5lOk6LQr5A)

The parameters like alpha, the learning rate, the number of iterations, number of hidden layers that control the ultimate parameters W and B and so we call them as hyper parameters.