**Sequence data**
Supervised learning with label data. 

![](https://lh5.googleusercontent.com/GnHXUuOKH_XNDmadaGe2w3_VcWLQ7jLRPnQeO5b4W7BVpsuvvuCeXxnH_uXYMAgF7Qpy-SmQ-9SPRCa7GTI0FPk8dzJDpCNrjo5ZOPPng6GocFrFM4v1BtY5slf0qiqBNavn7gTpIikqJxXjVtBs7PQ)

Both input X and output Y are sequences. X & Y can be of different length. Only input X or output Y are sequences.
 ![](https://lh4.googleusercontent.com/iM-RW70hLkRBucWGREWwWr9aEbI1Eyj614wW3nYdAby_cOUVM-l3lCc3YVSo2d5w9CWPYKXqNSomOEnCCjS-wJPRFHOFjV9xu9kT6BEu1BEfsY6OJ_hgU62otDoMnm_RCKJlH04ywhUqhrQ37WejDmo)

for output – Y (i) (t) ; Ty (i)
**Vocabulary / Dictionary** is a list of words that will use in your representation
x(t) are of the dimension of a Vocabulary and are a one hot  representation for the word. 

**one hot**- only one 1 is present rest all zeros
when a word that does not exist in vocabulary, we create new token, new fake word called Unknown word (UNK)
![](https://lh3.googleusercontent.com/WhxagoPZkuD0TV1qYuUnAVOKhcTffwu19o9wWGNaqBHKBZ1AINBhdLHTkbZmETw2vpO9ZGV8rZ0K22YvjCWyg7W8Gswvx9kj_yiPxDyuKKVSQXRRSzGh13sClH9rzgAJNek-vWCROO3_ZeX9e6sKooo)
Converting the word to one-hot vectors and supplying it in network then getting the output.
![](https://lh6.googleusercontent.com/0nPa3OybIIU4GmX5nB0wuuEv1OjifNv5QLbZiL8Di_XCoipx0nhTgBHGx57HJVhzKMM9XMEBVKA0RtjO3tmsdknpnDamt4GOkPw-TbD9pfsoB_Shu6pVKyq1vW-OE16KkMVB1uQaPWR18yNkkTD4k8E)
**Problems:**
- length at input and output can be different this causes the problem. We can pad the surrounding of input & output. But its not as efficient.
- If name repeats twice, it is difficult to recognize
- many parameters.
![](https://lh6.googleusercontent.com/aVqVpoAqWFG6kETw269hGbsvJEwzoQ6Pw0V1bBLriHOCtSAmJUsb9n0mn42oML900M5_Rd8JQCFmaxrB_JkHEtz6GcocipTqDXsvyhHeU4rvSg8T4oD5BMvK005OPURxHImXvfu-ygtkFjgptqs98Uw)

**Recurrent Neural Networks** 
When output length = Input length.

Wax is the parameter inputted. 
Waa is parameter supplied to 2nd layer.
At timestep 1, activation is a<0>
Reading & inputing the first word x<1>, output predicted y<1>. 
In second timestep, 
we input x<2>, and a<1>out put predicted 
![](https://lh5.googleusercontent.com/CHdtiV8ZKC9LzW4CPVekr6IBTLJATrRjdmNXFWdhBKy28yZj_V_Ojrpr8lb8Qu7VjMA0EGxMLhOHmGrNVU2AuK8coEHTAHR2iDy6hNbLScBV-KpkfLqMWu7th5DgWj950iDr6Oik0ihMG4NFms_4FRo)

![](https://lh5.googleusercontent.com/LofQRDJPsBhgY58xmQKBoqBzwrwmz4-ZqiAXPJijlYcUoHE4bTl64d7A_p3QVD3xXvQcW9_poJ24rfZ0Svn-ZsvvuLh6yODfDs2YosvO3kEeQ2tFf9L9Ciucvpme2PXLKOVhddDJB0QtC5xD1613XZQ)Weakness
only inputs of previous layers is considered for output of the certain lager.
To find if Teddy is a name 
- Teddy Roosevelt was a great president 
- Teddy bears are on sale. 
words that come later matter as well

**Bidirectional RNN (BRNN).**

Forward Propogation. Activation functions tanh, Relv
for a<1>
for y<1>= g(Wya a<1> + by)-- Sigmoid 
![](https://lh6.googleusercontent.com/vcrjH0u-amznT1xgE1s_BZ6JSYTGYYbo0eE6_12pjiM7hy8rnEDC1o2oBis-u_57JEK5vvAP2nA5fxKkCT_KDY24nMeczdwMGtmD_nnIs5c7M5xrOLFuQ9Je9EyfUwTcfDrjwMs_gCkbo9e85tqSi-g)

**Back propagation in time.**
moves from right to left and calculates Loss function (cross entropy luss).
Then parameten are updated through. gradient descent.

**Examples of RNN architecture.**
Language Modelling. 
fundamental for speech recognition and machine translation system calculates probability of what the word might be
Estimates probability of that particular, sequence of words. 

**To build a model.**
- you need a corpus - NLP technology- body or very large tons of english sentences.
- Tokenize each word to one - hot vectors. 
- (EOS) end of sentence - when end of a sentence is reached, eos tag is passed
- It taken of training set is not in your vocabulary-You take the word and replace it with. (UNK)
Input and activation of first neuron is zero.
Output is a softmax function with probability of first word y> y <1>
for second step- inputs correct first word y<1> = x <2>. 
And predicts the 2nd word y<2> given what had come previously
![](https://lh6.googleusercontent.com/qiZQmE6M_xJybhWs7mnH743r3U7GKNrXarHtycxwW-ftob0ZEITpFs-_fWxVY-7KK33JWNegeM8VsqlZsnCSScM6Wg7amE1NtsXvhROO9bfANVNcllmwWLm4HCx_blEfv3k28xgX_ik_R1Y-MYa9qVo)
- RNN architectures can have different numbers of inputs and outputs, Tx and Ty
- Andrej Karpathy's blog post, The Unreasonable - Effectiveness of Recurrent Neural Networks, inspired the presentation
- Many-to-many architecture: input and output sequences have the same length, example is name entity recognition
- Many-to-one architecture: input is a sequence, output is a single number, example is sentiment classification 
- One-to-many architecture: input is a single number, output is a sequence, example is music generation 
- Many-to-many architecture with different input and
output lengths: example is machine translation

![](https://lh4.googleusercontent.com/RqJwlAeimrBjI7tmBpI61d0CK-56rn4fSGSHjSVwin_wdo8c0QP0STghnnciMBre66HuB5oQ7KaX-aeI4Q15Gi6xb7nasTCp0IRZuAhpxuC5Fiyuv0IAipyRJcdWeBN1StWUTZ-UXbce43RHBOLs1t0)

- Language modeling is a task in natural language processing (NLP) that RNNs do well 
- Language model estimates the probability of a particular sequence of words 
- To build a language model using an RNN, a training set of a large corpus of text is needed
- Tokenization step is used to form a vocabulary and map words to one-hot vectors or indices 
- An extra token (EOS) can be appended to the end of every sentence to model when sentences end
- If words in the training set are not in the vocabulary, they can be replaced with a unique token (UNK) - RNN model is built to model the chance of different sequences
- Inputs x^t are set to be equal to y of t minus 1
- RNN architecture is used to compute activation a_1 as a function of input x_1, which is set to a zero vector
- a 1 makes a Softmax prediction to figure out the probability of the first word y_1 -y_hat_1 is output according to a Softmax, predicting the probability of any word in a dictionary
- RNN steps forward to the next step and has some activation a_2, with the correct first word y_1 being given
- y_hat_2 is output according to a Softmax, predicting the probability of any word in a dictionary given the first word
- This process is repeated until the end, with the RNN predicting the probability of the next word given the preceding words
- To train the RNN, a cost function is defined, with the overall loss being the sum of the losses associated with the individual predictions
- The RNN can be used to predict the probability of a sentence given an initial set of words
- Sampling sequences from the model is one of the most fun things to do with a language model. Sequence models model the chance of any particular sequence of words
- To sample, first input x1=0 and a0=0 and sample according to the soft max distribution 
- Pass the y1 hat that was just sampled into the next time step as the input. Keep sampling until you generate an EOS token or reach a certain number of time steps - Reject any sample that comes out as an unknown word token
- Character level RNNs have a vocabulary of alphabets, space, punctuation, and digits - Character level RNNs can assign a non-zero probability to words not in the vocabulary Character level RNNs are more computationally expensive to train 
- Word level RNNs are better at capturing long dependencies between parts of the sentence range
- RNNs can be used for name entity recognition and language modeling
- Basic RNNs have a problem with vanishing gradients - In language modeling, singular/plural nouns can have long-term dependencies

Vanishing gradients can make it difficult for the output to be strongly influenced by an input that was very early in the sequence
- Exploding gradients can be addressed by gradient - clipping
- NaN result of numerical overflow. Solution: gradient clipping.
if gradient vectors is bigger than some threshold then rescale some of the
gradient vectors so it is clipped to some maximum or minimum value.

- GRUS are an effective solution for addressing the vanishing gradient problem and can capture longer range dependencies

![](https://lh4.googleusercontent.com/rzgr5X9YadkX8dGy_FtSuLtnXoROcW9CuttyAKFK344PYWKZmXso4T7CjRZLXenNdz6kqIrsvoNampfew2zSu666pLAndjhwes5umv-GMk1jSNhmuh6LB__pBjs_W5fWyjHOYZr5CT49VMwN_tUYydE)

![](https://lh4.googleusercontent.com/iKEX12T-X3j519Jf9FBTY5Ni93-41hqnoSeQHEi89_MTjTkfLnifjy1yc16t8rDHvBr9n9nsineMEKgig9mpCPD014SKzXRZkpNi0-GnmT72FGek7OOZddq0wKPh4MQG7h-ngnw1GqwlqIZGYbBL_9A)

![](https://lh4.googleusercontent.com/tuqDCqDjDFFFa0VrRkO6aTwcHhLhn-I7DE-wQFLbKsWQKuSdW0SNvVa762IuVtTvgeXa5u71wADqexyn9PjKQbIWSvB2usiCQDabplzViddR-QtRCjxiEHZTdNJSVYeWmGL_0bXgcphGV-zO-C_xm6g)

- Gated Recurrent Unit (GRU) is a modification to the RNN hidden layer that helps capture long-range connections and helps with the vanishing gradient problem
- GRU unit has a new variable called C (memory cell) which provides a bit of memory to remember whether the subject of the sentence is singular or plural 
- GRU unit outputs an activation value a of t that is equal to c of t
- Equations that govern the computations of a GRU unit include overwriting the memory cell with a value c tilde oft and computing a gate value Gamma_u (update gate) between 0 and 1
- Gate value Gamma_u decides when to update the memory cell value C^t
- GRU unit is often represented by a picture of a gated fence
- Gated Recurrent Unit (GRU) is a type of Recurrent Neural Network (RNN) 
- GRU takes two inputs: C^t-1 (previous time step) and X^t (current time step)
- These two inputs are combined and processed to generate c tilde t (candidate for replacing C^t) 
- Through a sigmoid activation function, this gives gamma u (update gate) 
- All of these inputs are combined through another operation to generate C^t (new value for memory cell) 
- GRU is good at deciding when to update the memory cell and when to maintain the value
- Gamma can be close to zero, which helps with the vanishing gradient problem.
- GRU can help with learning long-range dependencies.
- In practice, C^t, Gamma, and other inputs can be vectors 
- Element-wise multiplication is used to decide which bits of the memory cell vector to update 
- Full GRU unit includes an additional gate (Gamma r) to determine relevance of C^t-1
- GRU is one of the most commonly used versions RNNs
- LSTM- Long Short Term Memory- powerful version of GRU

![](https://lh4.googleusercontent.com/x3t9ql7oR5SaM6UtNc6s8V6piuD7ChrB_7tazYEQvzZeYXE6NkvQXkGO4pwfjGTGWZTaU7afyb_dKOLAfbCr9ZKBPjB8cqvVL70ZCJM7GdH0vYb6DPV2gdwWAvuQc4u3Bp3XzKkEZg-jG_H2uR0MyJs)

![](https://lh4.googleusercontent.com/lvKBD0Dcornn1PYiLFmny7epzZIQIqJkVxBo8i6JOgcxgS21LdhPzGGw8YVr2wdgjLUJqjudNLeM9uXFw0QV4BGOlFsFGI4EhnYDZL_rptVPHsi6v9Rcyexod8XU71O_BUMDyR-biTpG5hbJiud4Qfg)

- Long Short Term Memory (LSTM) is a more powerful and general version of the Gated Recurrent Unit (GRU)
- LSTM was developed by set hook writer and Jurgen Schmidt Huber and has had a huge impact on sequence modeling
- LSTM equations govern the memory cell c, candidate value for updating it c(tilde) t, update gate, forget gate, and output gate
- Picture of LSTM is often used to explain the equations and shows how a t-1 and xt are used to compute all the gate values
- LSTM is good at memorizing certain values for a long time due to the forget and update gates - Variations of LSTM include peephole connection, which means the gate values may depend on a t-1, xt, and the previous memory cell value
There is no universally superior algorithm between GRU and LSTM, but LSTM is the historically more proven choice and GRU is simpler and easier to scale to bigger models

![](https://lh6.googleusercontent.com/cu23AdCePjN-Fst9kDNGPYtxOX3bT9F-xTcEjHlsdMwjvMlUxzGoJk-Tivo0NaLHZYEBCip4DqQ-WDRRSj7WSyoiVuB32Zx5AoLfpdKtbqP1K4FhXkPq38rIJ_3baW-KjbF3a-Lp32-12n599NmX4Es)
**Bidirectional RNNs (BRNNs)**
- Bidirectional RNNs (BRNNs) are a modification to the basic RNN architecture, GRU, or LSTM, that allows the model to make predictions anywhere in the sequence, taking into account information from the entire sequence.
- BRNNs have a forward recurrent component (a1, a2, a3, a4) and a backward recurrent component (a1, a2, a3, a4) connected to each other, going backwards in time.
- To make a prediction at time T, the network uses both the forward activation at time T and the backward activation at time T.
- BRNNs are commonly used for natural language processing problems, such as labeling things in a sentence.
- The disadvantage of BRNNS is that the entire sequence of data must be available before predictions can be made.

![](https://lh3.googleusercontent.com/QdUHPSWfzlldZBquqDhxMLU4JbeWON3qZziJL-YAhTIX6VjhtMLIRCwocMQK5eHIQK1-dXnBpFMuzVKohkS12f89KYhyuYZ-6flfn_qFiMbC7XMH3W602V0F5vdHVIBR48KCvljUm7YiykQMxysae8Y)
**Deep RNNs**
- Deep RNNs are a type of neural network that stacks. multiple layers of RNNS together to learn complex functions
- Standard RNNs have an input X that is stacked to some hidden layer with activations a1, a2, a3, etc. Deep RNNs are like this, but unrolled in time
- Notation for deep RNNs is a[l](t) to denote an activation associated with layer I over time t 
- Deep RNNs have three hidden layers and the activations are computed by applying an activation function g to a weight matrix Wa and a bias ba 
- Deep RNNs are not usually stacked up to be 100 layers due to the temporal dimension Deep RNNs can have recurrent layers stacked on top of each other, followed by a deep network
- Blocks in deep RNNs can be standard RNNS, GRUS, LSTMs, or bidirectional RNNs
- Deep RNNs are computationally expensive to train, so they don't usually have many layers


blogpost Chris Kohler -- on LSTM