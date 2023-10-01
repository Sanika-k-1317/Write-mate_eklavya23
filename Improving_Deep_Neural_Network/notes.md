
**dev-set and test set**
- The goal of the **dev set** or the development set is that you're going to test different algorithms on it and see which algorithm works better.
- we'll have to have 98% train, 1% dev, 1% test or 99.5% train and 0.25% dev, 0.25% test. Or maybe a 0.4% dev, 0.1% test.
- make sure that the dev and test sets come from the same distribution.
- it might be okay to not have a test set : the goal of the test set is to give an unbiased estimate of the performance of your final network, of the network that you selected. But if we don't need that unbiased estimate, then it might be okay to not have a test set.
- if you have only a dev set but not a test set, you train on the training set and then you try different model architectures. Evaluate them on the dev set, and then use that to iterate and try to get to a good model.

when you have just a train and a dev set but no separate test set, use dev set or your hold-out cross validation set to see which of many different models performs best on your dev set.
you can take the best model you have found and evaluate it on your test set in order to get an unbiased estimate of how well your algorithm is doing.

![](https://lh5.googleusercontent.com/Gs_3UifJNAIK1fiYHGe5mcQVXWxyQpa7IKP45T6dZ7uDhxDX4zn9uPxPplhsVH-gXyzLw-Kn5NyYBS9kjzs4nmezUtBguz_BHJVsg5tafOHSyiE3TWmnir6UIzarktAaUEFccwEGJRm3hot7oWc21cg)

The first one is not a very good fit to the data, and so this is a class of high bias. So we say that this is underfitting the data.

The third one where you fit an incredibly complex classifier, you can fit the data perfectly. This is a classifier of high variance and this is overfitting the data. 

The classifier in between, with a medium level of complexity, that  looks like a much more reasonable fit to the data, so we call it, just right.

![](https://lh5.googleusercontent.com/ZUHcxsQc5_LLDicUQYxNhxnLuBLucNjOTooWJpp5daSDmy32eOEd4W8ejyNzAcDlB6Ir8UlY5443ovy3bGlffVcJ7ToEB7saPTbnmF0zXGwWY32rUAEwYxOzcBJHydDSYyXvqlKty750liFpmzoQ9EE)

By looking at your training set error, you can get a sense of how well you are fitting, and so that tells you if you have a bias problem.

And then looking at how much higher your error goes, when you go from the training set to the dev set gives you a sense of how bad the variance problem is.

All this is under the assumption that the Bayes error is quite small and that the training and dev sets are drawn from the same distribution. 

![](https://lh4.googleusercontent.com/bWZrAqFQt7l4K2vC3O_Kiu98Y_cMfvVPrObWwTKpYd0rhDx-Kf-11x2EmAzV_0c745Ulr5nvgb1K7IYw3BEo3-rFGKND2UcFSeJ1KtYMPOmBD_QaeK_Ng2yhKv1DFI9W59OuvKwPKBhbEhxYec2Wnnc)
So the classifier drawn in purple, has both high bias and high variance.
- high bias, because, by being a mostly linear classifier, it's not fitting the quadratic line shape that is w.
- high variance, because it had too much flexibility to fit. 

![](https://lh5.googleusercontent.com/OjxLD9e9zlhCRGs4v8MT8wyPFCPFVKqJQ7TZkaypc8Fvc8GPXU3_nRCdbSrKQ522hEjdvcFN7aLfkCxFUHtkKsZBr_zbJ3S5MC0kaBt7TTyZZ2pixSzWEenFwMTd-9MhMTIjrdIapjmGpxhrWrnSE_A)

- getting a bigger network  reduces your bias, without necessarily hurting your variance
- getting more data, regularization etc pretty much always reduces your variance and doesn't hurt your bias much.

If you suspect your neural network is over fitting your data, that is, you have a high variance problem, one of the first things you should try is probably regularization. 
![](https://lh3.googleusercontent.com/SdMCYwasrWiBUFBuVMXBbJbGo-pCu4h7PDscawhLYiibeLpKhr4a279FfOMdorBAwDJw3xyjmUVDFYiGGu2CAyhILbA1ZrfoOXG4VsLCv8A0tC5qN-_Eq-pz2uqLT0pDVALVXkFuNiA2Cj7PVUr4I-I)

![](https://lh6.googleusercontent.com/MEQgRHS8MORzBeXn8VBiTvoOhYsdP7cYHtyo_-3elBUcHX1SRdCdfBx-uUy1mqCC1fpHfE-v_yyWSVTSC0lylHUWVq3hNGbWhrqZccrcouGuX9qzjtHUATmdPgtWeE2rN_SZ80Comr_30rWw-JOhQ3s)

And so what we did for regularization was add the extra term that penalizes the weight matrices from being too large. And we said that was the Frobenius norm. 

if we crank the regularization lambda to be really, really big, W becomes reasonably close to zero that's basically zeroing out a lot of the impact of hidden units. Then, this much simplified neural network becomes a much smaller neural network that will take you from this overfitting case, much closer to the left, to the other high bias case.

if lambda, the regularization parameter is large, then the  parameters will be relatively small, because they are penalized being large in the cost function.
And so if the weights, W, are small, then because z =W+b. But if W tends to be very small, then z will also be relatively small.

if z ends up taking relatively small values, just in this little range, then g(z) will be roughly linear. So it's as if every layer will be roughly linear, as if it is just linear regression.
So it's not able to  fit those complicated decision, very non-linear decision boundaries that allow it to overfit  to data sets

![](https://lh6.googleusercontent.com/ZFOnP2-7DuNXK_Hh-ULWs7v5A6oPrwjDprgj0hPfsbUt44TTplXXEmpXURcNDXAVimrVcuAVS2y1fpwcZ-PqBIQ8djlM9gMwOgYc4HF57_Jo3gTHyMt9-9UuJxLwGYJwTXd4e--bTflrYd6TxE43D8Y)

![](https://lh6.googleusercontent.com/goLdTPhGiPRFIX5Nxi2HfCimP8U2jm6pfQF2wGYk1LnVLlI9tzH053Cg-K-WKwz7ssNRiZvYiQdy66lmPXziE76oG9nu_b5ICiDrzsKBeZO-4G7_8Lg6h-a6JOMyiC9dn0aw4zywjkE48xMXrKGzW9w)

**Dropout regularisation**

- d3- decides which node will be cancelled
- To not reduce the value of z4 we have to divide a3 by keep prob
- We multiply a3 by d3 so if one element ends up zeroing out, the other vanishes as well
the number of neurons in the previous layer gives the number of columns of the weight matrix, and the number of neurons in the current layer gives us the number of rows in the weight matrix.
With dropout, we go through each of the layers of the network and set some probability of eliminating a node in neural network.

Let's say that for each of these layers, we're going to- for each node, toss a coin and have a 0.5 chance of keeping each node and 0.5 chance of removing each node.
So, after the coin tosses, maybe we'll decide to eliminate those nodes, then what you do is actually remove all the outgoing things from that no as well. So you end up with a much smaller, really much diminished network. And then you do back propagation training.

There's one example on this much diminished network. And then on different examples, you would toss a set of coins again and keep a different set of nodes 

![](https://lh4.googleusercontent.com/5aFN7pmqG4K-86EttXTSUzBHpwONponMEcvRHbBG47wtXBKZYRlILTY_AQMf9kyO89PvVwwEUVwmSGpeQl1yBcjmuAmIgk1ZfIbIvNRlli3xyzWJajUWbovC3XsCy-UQvl8UUusVSShzKlzuFIPDcDo)

So it's as if on every iteration you're working with a smaller neural network. And so using a smaller neural network seems like it should have a regularizing effect. 

The ways were reluctant to put too much weight on anyone input because it could go away. So the units will be more motivated to spread out this weights and by spreading out the weights this will tend to have an effect of shrinking the squared norm of the weights, and so similar to L2 regularization.

The effect of implementing dropout is that its strength the ways and similar to L2 regularization, it helps to prevent overfitting

![](https://lh5.googleusercontent.com/-nwHgkHkbirsCY2cL9uzlUl42ZxK68_HJJ32-AuLLb-y2Ox26WbdvY1fQRUPQ3N9ZVsvSK7vyU_P22eSx6V8Vhw2xnQDtb4c3FU73sk7ULnxUkr_PIAzQbLLpGC6rF1KLNwRluzqGsKGDW0lUpU2nn4)

One big downside of drop out is that the cost function J is no longer well defined or it's certainly hard to calculate. So you lose this debugging tool.
So turn off drop out or if you will set keep-propped = 1 to calculate J

**Data Augmentation**
- getting more training data can be expensive 
by flipping the images horizontally, you could double the size of your training set.
by taking random distortions and translations of the image you could augment your data set and make additional fake training examples. But these extra fake training examples they don't add as much information as they were to call they get a brand new independent example of a cat.

**Early Stopping**
 with random initialization you probably initialize w to small random values so before you train for a long time, w is still quite small.
And as you iterate, as you train, w will get bigger and bigger and bigger until here maybe you have a much larger value of the parameters w for your neural network. So what early stopping does is by stopping halfway you have only a mid-size rate w.

![](https://lh3.googleusercontent.com/2xzcrSCJvFoZd5wWQdEFMIvqEhlQBeIxQvgZKx4cpe9-n8AZ8QybbfKTJqufvFCTUWdiZHDPZN3SyY6VTTYGcCwsLOzS3vxchnQMJ4CGHgKTaAWeJJFeGBhjNW9ln5q6b6tmmwP4IJDQfb6Nep4dRLA)

main downside of early stopping is that this couples these two tasks.

By stopping gradient decent early, you're breaking whatever you're doing to optimize cost function J, because now you're not doing a great job reducing the cost function J. 

And then you also simultaneously trying to not over fit. So instead of using different tools to solve the two problems, you're using one that kind of mixes the two. And this just makes the set of things you could try are more complicated to think about.

Rather than using early stopping, one alternative is just use L2 regularization then you can just train the neural network as long as possible.
But the downside of this though is that you might have to try a lot of values of the regularization parameter lambda. And so this makes searching over many values of lambda more computationally expensive.

And the advantage of early stopping is that running the gradient descent process just once, you get to try out values of small w, mid-size w, and large w, without needing to try a lot of values of the L2 regularization and hyperparameter lambda. 

**Normalisation**
![](https://lh4.googleusercontent.com/gAbFpejIvgtr0N_F89QCzmUao-xXwbjGc6sDDa8TzTdct-4t_Yb5NIKqjA8euVwBNNJHKGsSkQ6omS-anVWcFnHvnuJCbLmA-I1QQIXsCLAFiJFl3KTEpIQL-JrB3A2VA7kBVMff_ufBRpDqRTmjAo8)

It turns out that if you use un-normalized input features, it's more likely that your cost function will look like this, like a very squished out bar, very elongated cost function where the minimum is hard to find ad will  have to use a very small learning rate

![](https://lh6.googleusercontent.com/upItxe01A_gR_GrqypfltetBuZk_HRHrrr6SqwBtoByqmDq5MedD7-PEySbJCKA0gnFA5Vr-HaOxDCX5Ts__8s5Y0_pocmldPONINrPBen2b3RxPImNFcRkW2eNh2Ar7Y4rGSFTIWL2tyngYrzasG5w)

if you normalize the features, then your cost function will on average look more symmetric and gradient descent can go straight to the minimum. You can take much larger steps where gradient descent need, rather than needing to oscillate around like the picture on the left.

**Exploding /Vanishing gradients**
![](https://lh4.googleusercontent.com/LrvlJ3K2pPfgTBbR81feimIcivajYQKU_gFvg_4DhkHsleEEfG-wavvp1KpCczAhkyNFQPxTbQeAQ5oeli8fBEJ5odSiQRGo8Qx5rIgjtQv_CH9ruxXwVVmaYy0HpTV7jC7UhUY5i1eT4fHtKMw8jIg)

![](https://lh3.googleusercontent.com/IOTTbMajdiGvFOzXd93qbu6XTQWVzMovFTGRL46Q2sEVkfkaRTWTytfy9_NsIKAMKNZWTFwzSgiErFRqe-1okfB_oX2RuDcuwADHe3LcYfksRB3mt8y4_qAlD7xv88mhGifTW9Js1zBdIkOf8dkPXXU)

if the input features of activations are roughly mean 0 and standard variance and variance 1 then this would cause z to also take on a similar scale and this doesn't solve, but it definitely helps reduce the vanishing, exploding gradients problem, because it's trying to set each of the weight matrices w,  so that it's not too much bigger than 1 and not too much less than 1 so it doesn't explode or vanish too quickly.

![](https://lh4.googleusercontent.com/HY-rhBSqLwKAYlfre9tE2MCt7OUk5ru8SSzBkopYVVhaNc1N-vPZR9KMC1pdPgqgD2nI2q8Xu_82YX3dbTD7z9_w3qyUls_cbJOUsM6oqYLS13a-HSkxCndXCRQKmt8iz-0x8vMkdx9Q8brnb0P7x_U)

two-sided difference formula is much more accurate.
![](https://lh6.googleusercontent.com/wTNWctW4Qa6NuAkwkYaVDIUTWms2-aqH0K132Tqgkb4lZvcZ90nkPX6wwhDzSa_k0AIFM-_x0bNWsG_-zLhFmsZOaivDJ8m8bRmUK5FK_docUkhp7xHWq778wZuRbnaiNKsdzb7De1zC5j8yjPqoncs)

- Gradient checking is a technique used to save time and find bugs in back propagation implementations - Parameters (W1, B1, etc.) are reshaped into a giant vector, theta
- Cost function J is now a function of theta
- To implement gradient checking, a loop is used to computed theta approx i to b - J of theta is nudged by epsilon and then divided by 2 theta
- This should be approximately equal to d theta i - The two vectors (d theta approx and d theta) are compared by computing the Euclidean distance between them
- If the value is 10 ^ -7 or smaller, the derivative approximation is likely correct
- If the value is 10 ^-5 or larger, a bug should be suspected
- If the value is 10 ^-3 or larger, a bug should be seriously suspected
- After debugging, if the value is small, the implementation is likely correct
![](https://lh6.googleusercontent.com/K-sF5BGCPuxDnSF8htlRaZjQ7DaVmcwUqbXOmpjxZP9vAvpX3GqP1DBO1dNtFOqS50WPssFOhmWTcIuKSggnmO7t8WcAtuz6r9DAGW7yrw44oxfPQoUVHoav7XfFG_j7U819aIwiIxfUr2SZVfsmlh0)

computing d(theta)approx_i, for all the values of i is a very slow computation. So to implement gradient descent, you'd use backprop to compute d(theta)
some debugging tricks
- And it's only when you're debugging that you would compute this to make sure it's close to d(theta). But once you've done that, then you would turn off the grad check, and don't run this during every iteration of gradient descent, because that's much too slow.
- but sometimes it helps you give you some guesses about where to track down the bug. Next, when doing grad check, remember your regularization term if you're using regularization.
- turn off dropout, use grad check to double check that your algorithm is at least correct without dropout, and then turn on dropout. Finally, this is a subtlety.
![](https://lh5.googleusercontent.com/9us8HZfCsySYCp8KTSjNWG_MMfqqeAc0YAp6Ph0lZMjMx4YkPmRmpxFAI8debZq5oyDLP7wfoDIbLwHRacuuKVQyELglM7C5guC39pXGSWY1xSe_Dy-o2SF8PWWv15goE42HpA9GZwoiSBY33G5r7kI)

**Mini-Batch Gradient descent**
Z[L] comes from the Z value, for the L layer of the neural network and here we are introducing {T} to index into different mini batches. 
X{T}, Y{T}. 

the dimension of X{T} is an (X , M).
For the X values for a thousand examples, the dimension should be (Nx, 1,000) . To run mini-batch gradient descent on your training sets you run for T = (1, 5,000) because we had 5,000 mini batches as high as 1,000 each. 

use vectorization to process all 1,000 examples 
![](https://lh5.googleusercontent.com/rVqs-EVjOHgS6TBogH5xGO9r0O43oJquW9umCyazO29PY2HDPC7sJvA8KS2I8J2APAuOmIiKdlmVzWIpftThqbInULNOCgIZpnrP1E3xmpybFGDa3mMGpJoLTGOINAtgTr-PjVU82c_tWXcV7czJpVM)

- Mini-batch Gradient Descent is an optimization algorithm that enables faster training of neural networks. 
- It works best in the regime of big data, where training on a large data set is slow.
- Mini-batch Gradient Descent splits the training set into smaller mini-batches, each with 1,000 examples. .The notation X{T} and Y{T} is used to index into different mini-batches, where X{T} is an (Nx,1,000) matrix and YT is a (1,1,000) matrix.
- Mini-batch Gradient Descent is different from batch gradient descent, which processes the entire training set at once.
- To run mini-batch gradient descent, a for loop is used to process each mini-batch X{T} and Y{T}. 
Inside the for loop 
- forward propagation is implemented on X{T}, followed by computing the cost function J{T}.
- Back propagation is then used to compute gradients with respect to J{T}, and the weights W[L] are updated.
- One pass through the training set using mini-batch gradient descent is called one epoch, and multiple passes are usually taken until convergence is achieved. 
- Mini-batch gradient descent runs much faster than batch gradient descent when training on a large data set.

![](https://lh6.googleusercontent.com/t0jwghZF6dG0fT2KWa3OBE8c7WBX0cZ2rG3DtRj9-UQKX_d1huqbnGQOcS1RsaFxr9Y7HDyG8b2nzWk7LlC6zxv5CNYHaO9T5xHEAzRvEjRcQw3qnSlHXm8yX6oGo1DL0_nZc1TAaG3DLIzZQxJADL0)
So if you plot J{t}, as you're training mini batch in descent it may be over multiple epochs. But it should trend downwards, and the reason it'll be a little bit noisy is that, maybe X{1}, Y{1} is just the rows of easy mini batch so your cost might be a bit lower, but then maybe just by chance, X{2}, Y{2} is just a harder mini batch ,maybe you needed some mislabeled examples in it, in
which case the cost will be a bit higher and so on. So that's why you get these oscillations as you plot the cost when you're running mini batch gradient descent.

![](https://lh4.googleusercontent.com/BOhKvJwLDZdrjhGRMCqRuJhujZ5arpPVQ1jZ2UmmvJHssSTmlf61gMMTRmq4FWhUdY5GC-dHEDn-RyvDg_L-YJAvTrnNh0mnUIme0qts69_ecYlffvr1Mr6X9O0BGEe7TCVfTZ4-dkrNrkpkbJgGYnk)

- Mini-batch gradient descent is an optimization technique used to make progress and take gradient descent steps even when only partway through processing a training set. 
- With batch gradient descent, the cost should decrease on every iteration.
- With mini-batch gradient descent, the cost maynot decrease on every iteration, but should trend downwards.
- The size of the mini-batch should be somewhere between 1 and the size of the training set. 
- Batch gradient descent takes too long per iteration, while stochastic gradient descent can be noisy and never converge.
- Mini-batch gradient descent is a method of training a model that is more efficient than batch gradient descent and stochastic gradient descent.
- It is best used when the training set is very long.
- Stochastic gradient descent is noisy and can be reduced by using a smaller learning rate.
- Mini-batch gradient descent has two advantages: vectorization and making progress without needing to wait until the entire training set is processed.
- The mini-batch size should not be too big or too small, and should be something in between.
- If the training set is small (less than 2000 examples), batch gradient descent should be used.
- Typical mini-batch sizes are between 64 and 512, and it is often faster if the mini-batch size is a power of 2.
- Make sure the mini-batch fits in CPU/GPU memory. 
- The mini-batch size is a hyperparameter that can be searched to find the most efficient value.
![](https://lh4.googleusercontent.com/B8Mpzu9rngpU_ZnWVV_AdNdzK3D1GglNNw0nSXIwsTtD0KjQxabS7vpVt63vPEbN6d8XiLHNyy6AuBXkOO1HJxH4XsoVZk4e0X0A75YIKzpoqVwwcZX2CmU3XXlubtrO0bvVn652QnyJlUVKeGjAqd8)

So, very high value of beta the plot you get is much smoother because you're now averaging over more days of temperatures but on the flip side the curve has now shifted further to the right because you're now averaging over a much larger window of temperatures. And by averaging over a larger window, It adapts more slowly, when the temperature changes. So, there's just a bit more latency. And the reason for that is when Beta 0.98 then it's giving a lot of weight to the previous value and a much smaller weight just 0.02, to whatever you're seeing right now.
![](https://lh4.googleusercontent.com/rm31zyKDPkqC2Q8e7o8JtDZNkuoXwgflJNGXRL_pFNPvVu1d5hSUQHRSW8ND0Y-T-DnESzhXrA_vlbPEiGzgRUY3GOY7QFOHkafD0-bNzD4gcK9G88Je1fvSCaeq80ML_yaFCxpAkx65-msULD2aT3Y)

- Exponentially Weighted Averages (EWA) are used to optimize algorithms faster than gradient descent 
- An example of EWA is given using daily temperature data from London over the course of a year
- EWA is initialized with VO = 0 and then on every day, a weighted average is taken with a weight of 0.9 times the previous value plus 0.1 times the current temperature 
- The general formula for EWA is Vt = 0.9Vt-1 +0.10t 
- When beta is set to 0.9, it can be thought of as averaging over the last 10 days temperature 
- When beta is set to 0.98, it can be thought of as averaging over the last 50 days temperature 
- When beta is set to 0.5, it can be thought of as averaging over the last 2 days temperature - By varying the parameter, different effects can be achieved and usually there is a value in between that works best
![](https://lh6.googleusercontent.com/ILO1Kg2EyBR_0IeSDmkMGWMdc-VsoWBXKzqBcRx3TzTKZMvpU3S5IkUPL6AfZUkW95gaqW3WzkeOs1EmN0FaQdvPp4ayPHcgNVn5F8d755cgrmDeD8Ew_E1FriC4tShKNBGveMM4ilAexGZLcv3TiVs)

- Exponentially Weighted Averages (EWA) can be improved with a technical detail called bias correction
- Without bias correction, the purple curve is obtained when Beta = 0.98, which starts off low
- To fix this, V_t is divided by 1 - Beta^t, where t is the current day
- This removes the bias and produces the green curve
- Bias correction is not often implemented in machine learning as it is more convenient to have a slightly more biased assessment

Bias correction can help obtain a better estimate early on while the EWA is warming up and this up and down oscillations slows down gradient descent and prevents you from using a much larger learning rate. In particular, if you were to use a much larger learning rate you might end up over shooting and end up diverging and so the need to prevent the oscillations from getting too big forces you to use a learning rate that's not itself too large. 
- on the vertical axis you want your learning to be a bit slower, because you don't want those oscillations.
- on the horizontal axis, you want faster learning. Right, because you want it to aggressively move from left to right, toward that minimum, toward that red dot. So here's what you can do if you implement gradient descent with momentum.

![](https://lh3.googleusercontent.com/DrN_1TxSl7PrwtiWp29Kifh_TbNUXSSfE9D0IIZIQ4hMZb3zeiBY2MmbEo5oTTLCXbCKwJN-TjCGh4FArcCXBL4GPM10sJ4zpGCYHO7VSYHg5F0eDsbZ3qnu8lFIZ9mEr-XDnTEf3zDaMKVDxlXYRSU)

- Gradient Descent with Momentum is an algorithm that works faster than standard Gradient Descent
- The basic idea is to compute an exponentially weighted average of gradients and use that to update weights
- This algorithm helps to dampen oscillations in the vertical direction and move quickly in the horizontal direction
- The algorithm computes the derivatives dw and db on the current mini-batch 
- It then computes vdw and vdb as Beta vdw + 1- Beta dW and Beta vdb+ 1- Beta db Weights are then updated as W - learning rate
- This helps to smooth out the steps of gradient descent 
- The hyperparameter Beta is usually set to 0.9 and bias correction is usually not used
- The formula with the 1 minus Beta term is preferred as it is more intuitive and does not require retuning of the learning rate alpha when Beta is changed

RMSprop stands for root mean square prop and is an algorithm that can speed up gradient descent
- It works by keeping an exponentially weighted average of the squares of the derivatives
- The parameters are updated by dividing the derivatives by the square root of the exponentially weighted average
- This helps dampen the oscillations in the vertical direction and speed up learning in the horizontal direction
- RMSprop was first proposed in a Coursera course by Jeff Hinton
- Combining RMSprop with momentum can result in an even better optimization algorithm
![](https://lh4.googleusercontent.com/ndEHngg3fbGNLlosQJsiLh8DU4Fmxlwy-24BkSvGOydtTNRjTtFpJjtRkfn40EIG5wGaSvE3l4kg7PVEgSmZ4ShBamsXY5HUvjL6pjHkUS12F_wK83UvRjwCrtG8csS612Vf_YTWTdMBHb9RDvH5vyA)![](https://lh3.googleusercontent.com/EPCJJwgPcw8ltwugRDYtea6hgbN2wk7AplJ2P_A-JZBJ8rGi1icGKq0cWdk2NMeTsvdqeFFTHl19omcIXF_k-uGA0dI0alTU0SpVf9KIz895tLMFafKgzWGxGlAeRxWoP3yoMt0FCTbGQDARezlB7YM)

- Adam Optimization Algorithm is a rare algorithm that has been shown to work well across a wide range of deep learning architectures
- To implement Adam, you initialize V_dw, S_dw, V_db, and S_db to 0 - On iteration t, compute derivatives, compute dw and db using current mini-batch - Momentum exponentially weighted average is computed using hyperparameter Beta_1 RMSprop-like update is computed using
![](https://lh3.googleusercontent.com/H6FhsHn8DRH5EnC490kDL6mSsmEnkboTxxqLBR91FLLeQH59Ka0RYX2uOVKSSwtIzhevJGywWb_PmM9NSS7tmIAVZV8U3QiF1EAOTS1IGgJcw0h7Ugm1PH9Yu7anN0VI1AwQmzDOqOuMb0Rko5M4xTg)
hyperparameter Beta_2 Bias correction is implemented on V and S
- Update W and b using V_dw/V_db corrected and S_dw/S_db corrected respectively
- Hyperparameters Alpha, Beta_1, Beta_2, and Epsilon need to be tuned
- Adam stands for Adaptive Moment Estimation
- Learning rate decay is a technique used to speed up a learning algorithm by slowly reducing the learning rate over time
- This helps to reduce noise in the mini-batches and allows the algorithm to converge to the minimum more quickly
- Learning rate decay can be implemented by setting the learning rate Alpha to be equal to 1 over 1 plus a parameter (the decay rate) times the epoch number
- Other formulas for learning rate decay include exponential decay, Alpha equal to some constant over epoch number square root times Alpha 0, and Alpha equal to some constant k and another hyperparameter over the mini-batch number t square rooted times Alpha 0
- Manual decay is also an option, where the learning rate is adjusted hour-by-hour or day-by-day - Hyperparameter tuning will be discussed in the next lesson, and learning rate decay is usually lower down on the list of things to try

__![](https://lh5.googleusercontent.com/_xk2JJxA4xd1OD_wGc50q8_Ah0DdCtvWPP8CCb818iWuOJcXboF69_SeF0hxOMxycVZx6ANvHkUMngGpBj3dHyjDZC1D6xYt09INKf1uBjguYo0fQOMmtX8RLRKD6AG29H5ET7LHw3hKLZi-OOy_y28)

- In the early days of deep learning, people used to worry a lot about the optimization algorithm getting stuck in bad local optima
- People used to think of local optima as points in a two-dimensional surface where the cost function is at its highest
- However, it turns out that most points of zero gradients in a cost function are actually saddle points
- Saddle points are points where the gradient is zero, but in each direction it can either be a convex or concave light function
- In very high-dimensional spaces, it is much more likely to run into a saddle point than a local optimum
- Plateaus can slow down learning, as the gradient is close to zero for a long time - Algorithms like momentum, RmsProp, and Adam can help speed up learning by helping to move down plateaus
- Our understanding of high-dimensional spaces is still evolving
- Tuning process involves setting a lot of different hyperparameters Most important hyperparameter to tune is the learning
![](https://lh3.googleusercontent.com/VR9Gp3L-JORZB_9ko5MrxzPl_UwanMCkRUUckXi8Urb3o64cgTpK7UXUH128IDqOeOrNVREGibPlzDRgRQq_mM9MgtvhwfDB0OlJRKxwFhbi6T6vL83AwDvbl0jKMNqAyn4x7ytRetGQJkmpXORW5kY)

![](https://lh5.googleusercontent.com/VdP7oDjjhmr9Xlde7gEj7VjloPk8wr6E3iZWJ4mi6TF2jeG9kNU5-wLk3czsFIEfm4NQ76jJsI5kqfQrVsaDfk5sH28eQmzTWslWpEE8ky-fjA0xgRJUIWxE-FUXp0htnL32Ty1gRu_nUtgxmGM94Tw)

**rate alpha**
- Other important hyperparameters to tune include momentum term, mini-batch size, number of hidden units, number of layers, and learning rate decay
- Earlier generations of machine learning algorithms used a grid to sample points, but deep learning uses random sampling
- Coarse to fine search process can be used to focus resources on a smaller region of the hyperparameters
![](https://lh3.googleusercontent.com/iOazhDN5st5B5HcX7sA-FPZgeDrkJ5faSg8Y9wkfSBjuzgvQeUv5DURZQfDReblY6GU_qYvYsH80KsmJBh6POzgJ7796TipLOOL9QZkJM6b0WHEWSW1y9y_HISdsJpzJHM7VxLE9LfJMqkEjcBYUixo)

![](https://lh6.googleusercontent.com/l7ktHNTvGXMv-KZKrmrY-imGyVSPWFkkpWe06MhzfKxC1T88aLBVze7-3W14JQFwkZ_oMYc1mbMBZYGvtb4sIISWm6zpChHOgVf3O29PVSIV-E3TWpTzqjXs2MnnxSv5Ls8j2tRV1SEBp89_VMaEe0M)

- Deep learning is applied to many different application areas and intuitions about hyperparameter settings from one application area may or may not transfer to a different one.
- It is recommended to reevaluate hyperparameters at least once every several months to make sure they are still the best settings.
- There are two major schools of thought for searching for hyperparameters: the Panda approach and the Caviar approach.
- The Panda approach is used when there is a huge dataset but not a lot of computational resources, and involves gradually babysitting the model as it trains. 
- The Caviar approach is used when there are enough computational resources to train many models in parallel, and involves trying a lot of different hyperparameters and picking the one that works best. 
- There is one other technique that can make neural networks more robust to the choice of hyperparameters, which will be discussed in the next video.
- Batch normalization is an algorithm created by Sergey loffe and Christian Szegedy that makes hyperparameter search easier and makes neural networks more robust. 
- Normalizing the input features of a logistic regression model can speed up learning by computing the means, subtracting off the means from the training set, computing the variances, and normalizing the dataset according to the variances.
- Batch normalization applies the same normalization process to the values of hidden layers in a neural network, normalizing the mean and variance of the values to make training of weights and biases more efficient.
- The values are normalized by subtracting off the mean and dividing by the standard deviation, and then gamma and beta parameters are used to set the mean and variance of the normalized values to whatever is desired.
- The gamma and beta parameters are learnable and are updated using gradient descent or other algorithms.
- Batch normalization allows the hidden unit values to have a different distribution than mean 0 and variance 1, and it helps to take advantage of the nonlinearity of activation functions.
![](https://lh3.googleusercontent.com/MQ5zogTXTzmmmWwvcauYTjvmDbJPBaFhskdSv0Wu17xXkpIdLVlhLXMZPOOQHnb8-Ne7PnVmmBI-Wm1isBF9CllFG69L3KcpAckk1eyh52eKxSmncf747xllVwhyPRyrf2gkwzgh78zbcO9O6_QDP24)![](https://lh3.googleusercontent.com/eZQdaw4qVEZ0LALXvN7HDXpr5eKrwiHzQvKRcHuKyw6F1991k6kOuuNLsIwN_Lye1ZIIWO3bDpZKtFDvODgjlEf4RTtzeRIwbZALLlHB8vYIfnCvjuNvo0HJDHg50JtddnVcKvpg5HFBUGmDzoQrghg)

- Batch Norm is a technique used to normalize the inputs of each layer in a neural network
- Batch Norm is applied between the computation of Z and A in each layer - Parameters of the network include W1, B1, WL, BL Beta 1, Gamma 1, Beta 2, Gamma 2, etc. for each layer in which Batch Norm is applied -Parameters are updated using optimization techniques such as gradient descent, Adam, RMSprop, or momentum Batch Norm is usually applied with mini-batches of the training set
- Mean and variance of Z1 are computed on the mini-batch and Batch Norm subtracts by the mean and divides by the standard deviation and then re-scales by Beta 1, Gamma 1
- Batch Norm is used to normalize the data in a mini-batch before applying the activation function - Batch Norm is used to compute Z tilde, which is the normalized version of Z, using the data in the mini-batch - The parameters WL, BL, Beta L, and Gamma L are used to compute ZL 
- However, since Batch Norm zeroes out the mean of the ZL values, the parameter BL is no longer necessary and can be eliminated 
- Instead, Beta L and Gamma L are used to control the mean and variance of each of the hidden units -To implement gradient descent using Batch Norm,one must use forward prop on the mini-batch, use Batch Norm to replace ZL with Z tilde L, use back prop to compute DW, DB, D Beta, and D Gamma, and then update the parameters
- Other optimization algorithms such as gradient descent with momentum, RMSprop, or Adam can also be used to update the parameters Beta and Gamma that Batch Norm added to the algorithm
![](https://lh5.googleusercontent.com/4rVSXYVPv2Dj1r6InBdZrbFMNzJRle8tWm3NmkqyLIurxG2FqRH0u4_ar9Q1hfZklHgqa0dzq0BEs1d_S5ZK9QIr8kIWcJqhywPJo80D7znC7-b8BVYpyg2k3pTkKqTja9JCWu4RbaGHGK8RIPqGdBA)![](https://lh4.googleusercontent.com/evmYZgAYKo1XasRwSUD425I9KeUq1iQxe8IvC9JUWjdHf2D3BhmOnEV3pt_vPoz7BFyiO1E0IE7_vE62UycQ8x2-P3kh4p8Y5SaF1KHZ6f5JfGEuw0iOJXvUpOSpR7y8q-5k4syGLTiSGmlA7hF5uFY)

- Batch norm works by normalizing the input features (X's) to have a mean of zero and variance of one, which can speed up learning. -It also makes weights in deeper layers of the network,more robust to changes in earlier layers.
- This is known as the problem of covariate shift, where the distribution of X changes and the learning algorithmn needs to be retrained.
- Batch norm reduces the amount that the distribution of hidden unit values shifts around by ensuring that the mean and variance of these values remain the same.
- Batch norm reduces the problem of input values changing, making them more stable so that later layers of the neural network have more firm ground to stand on.
- It weakens the coupling between what the early layers parameters have to do and what the later layers parameters have to do, allowing each layer to learn. more independently.
- Batch norm has a slight regularization effect, adding noise to each hidden layer's activations. -It adds noise to the hidden layers, forcing the downstream hidden units not to rely too much on any one hidden unit.
- Using a bigger mini-batch size reduces the regularization effect.
- Batch norm handles data one mini-batch at a time, computing mean and variances on mini-batches
- At test time, something slightly different needs to be done in order to make predictions.
- Batch norm processes data one mini batch at a time, but at test time, you might need to process examples one at a time
- During training, the equations used to implement batch norm involve summing over the examples in one mini batch, computing the mean and variance, and then scaling by the mean and standard deviation with Epsilon added for numerical stability
- At test time, you don't have a mini batch of examples to process at the same time, so you need a different way of coming up with mu and sigma squared - This is typically done by estimating mu and sigma squared using an exponentially weighted average across mini batches
- To compute the exponentially weighted average, you keep track of the mu and sigma squared values you're seeing during training -At test time, you use the exponentially weighted average of the mu and sigma squared to do the scaling and compute 2 using the beta and gamma parameters learned during training
- Any reasonable way to estimate the mean and variance of your head and unit values Z should work fine at test
![](https://lh6.googleusercontent.com/W5W1rftK_zwiyp3PgDG9Y1JPBneJ84rkWJmRsYFdOQOfK3HjVZI2-lmcYsT_Nt99uOtcTnxqj_UuDfT9diUYdi3MHQBFD10Y7mH-lEk9ZbwhWUsmqfLorg7Dcm9bnLMq54_6R8X2WbvUeK2z28d0kJA)![](https://lh4.googleusercontent.com/Gp6O8rHXqXxsutElrkcBv6GKDEN8qvILcjeoT9SYINyeCCqfjKmzq7hkO7QYAdqjBKfRsawBT78Zz6Y1uLHRuRH7GzEy-CsahQxPaBwPQwewcCn_HTu2M9g3fOijInY7tGcyPcD_VglfxD7bTWL6OJg)![](https://lh6.googleusercontent.com/BU5tKzD77Ai9Qou2ui7pYErFK-tSebwDcpOVkf_rJlwiqgdoiyIsGVEBbUHxkmL1f0qLKIDzWiH9d2oAi1nfKAbTULHUMl6NlnBsywNk0tkZm7Rt54aYtxrpwKR-Sibie8HhYOXkYK1nHXtxPZ9XrUU)

- Softmax regression is a generalization of logistic regression used for multi-class classification - In this example, the classes are cats (class 1), dogs (class 2), baby chicks (class 3), and other/none of the above (class 0)
- The notation used is capital C to denote the number of classes, and the numbers indexing the classes are 0 through capital C minus one - The output labels y hat is a four by one dimensional vector, containing the probabilities of each of the four classes 
- The standard model for getting the network to do this uses a Softmax layer 
- The output layer computes the linear part of the layers (z, capital L) and then applies the Softmax activation function
- The Softmax activation function computes a temporary variable (t) which is e to the zL, and then normalizes it to sum to 1
- An example is given to illustrate the math behind the Softmax activation function
- Softmax Regression is a type of activation function that
- It is used to calculate the probability of a given image belonging to a certain class
- Softmax layers can represent linear decision boundaries between multiple classes
- Examples of Softmax layers with 3, 4, 5, and 6 classes are given
- Softmax layers can be used in neural networks with no hidden layers or with multiple hidden layers to learn more complex non-linear decision boundaries
![](https://lh4.googleusercontent.com/c5rfw-OQnvu_sLgm8uzEIT16vhOGDPHGhvf6gCqX9E_v0f6NeWAaf2Cbr8jhKAte5SxNNna33cQf3qpBZxYhfAO7Fvb5aV_cPYl7I7Nx6kLXlgB58s1Rashh_KI_nWeW58fQHOB0wdxHH1P8_uQB6Q0)![](https://lh4.googleusercontent.com/74ja9AT_drRsCxvuzhDjUg9quUa3QD6X2JXRLe4IaJPHM-121_dNbiyNGp53k9btfGW9BQvJLy7bMTohl7Y-auYHtHzMJnEIObB15z6uOCfdVM1VnmH5uFFgZOOLyHTl_yjYs6s4abLoqLfmQBTsRFA)

- Softmax activation function is a generalization of logistic regression to more than two classes - Softmax is a more gentle mapping from Z to probabilities, in contrast to the hard max which puts a 1 in the position of the biggest element of Z and Os everywhere else
- Training a neural network with a softmax output layer involves using a loss function that is the negative sum of yj log y hat of j - In the example given, the neural network outputs y hat=(0.1, 0.4, 0, 0) and the ground truth label is (0, 1, 0, 0) -The only term in the summation that is not 0 is -y2 log y hat 2, which is -log y hat 2, and the goal of the learning algorithm is to make this small
- Softmax classifier is used to classify inputs into one of C different classes
- Loss function looks at the ground true class in the training set and tries to make the corresponding probability of that class as high as possible -Cost J on the entire training set is the sum of the learning algorithm's predictions over the training samples
- Use gradient descent to try to minimize the cost - Y and Y hat are 4 by 1 vectors, and Y and Y hat matrices are 4 by m dimensional - Derivative with respect to z at the loss layer is computed as Y hat-Y - Primary frameworks will take care of the derivative computation for you
![](https://lh4.googleusercontent.com/C7QFLNR0IFEhO5MUj8IQbj8FNPRSJTGfjQCkxuL1PbyksIXwhhoF0hcw2-g1P7vxmgJUkfbgXJa7WaPHjlCqrVio-BV_idAF2QIb0M9V1e48r6Bf-UcCE7Clf5CH_g-WBl0OHz_pYblE-46PeEcBHgM)

- Learned to implement deep learning algorithms from scratch using Python and NumPy
- Implementing more complex models (e.g.convolutional neural networks, recurrent neural networks) is not practical for most people - Many good deep learning software frameworks available to help implement models - Frameworks have dedicated user and developer
communities
- Criteria for choosing frameworks: ease of programming, running speeds, open source
- Depending on preferences of language and application, multiple frameworks could be a good choice

![](https://lh6.googleusercontent.com/6HyuWnb0tNM7Ee3rJS3wvM7U0gfiAJJPOp0psfsnD0RXXiK8tQzWeA380JlNUbXYxoYCExI1u97ibTSay-6YCxRWHHBmwOu6JGGNcU_HgvRzO4piqGf44DY4Eul2WUOeHD1cmXcIqm5HxiKU-gznUHs)
• TensorFlow is a deep learning program framework that can help developers be more efficient in developing and using deep learning algorithms • Motivating problem: Cost function J to minimize, J of w
equals w squared minus 10w plus 25 
• Define parameter W using tf.variable and initialize it to zero
• Define optimization algorithm (Adam) and set learning rate to 0.1
- Define cost function (w squared minus 10w plus 25)
• Use tf.GradientTape to record sequence of operations to compute cost function
• Define training step function to loop over trainable variables is a list with only w
- Compute gradients with tape and cost and trainable variables
• Use optimizer to apply gradients • Run 1000 iterations of train_step and print W
- W is nearly five which is the minimum of the cost function
- TensorFlow can automatically figure out how to take derivatives with respect to the cost function 
- In TensorFlow, you only need to implement the forward prop step
- Cost function can be a function of not just the parameter w, but also the training data x and y
- x is defined as an array of numbers (1, -10, 25) which control the coefficients of the cost function - Cost function is the same as before, except the coefficients are controlled by the data in the array x - Optimizer is defined and w is set to an initial value of 0
- After one step of Adam Optimization, w is roughly 0.1
- For loop is used to run 1000 iterations and w is nearly at the minimum set, roughly the value of 5 - TensorFlow automatically figures out the derivatives and how to minimize the cost
- Different optimization algorithms can be used by changing one line of code
