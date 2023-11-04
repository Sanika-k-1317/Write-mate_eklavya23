
# Write-mate

## Aim

The project aims to create a system consisting of trained neural network that generates realistic handwritten text from input text. 

## Description
The project is an implementation of Alex Graves 2014 paper [Generating Sequences With
Recurrent Neural Networks](https://arxiv.org/abs/1308.0850). It processes input text and outputs realistic-looking handwriting using stacked LSTM layers along with attention mechanism. The output is x, y co-ordinates of the handwriting sequences which can be plotted to get the required longhand sentences.

# Theory

The model uses Long-short term memory cell layers and attention mechanism

### LSTM: 
LSTM can use its memory to generate complex,
realistic sequences containing long-range structure. It uses purpose-built memory cells to store information, is better at finding and exploiting long range dependencies in the data.
![](https://pluralsight2.imgix.net/guides/8a8ac7c1-8bac-4e89-ace8-9e28813ab635_3.JPG)

### Attention-mechanism
The number of co-ordinates used to write
each character varies greatly compared to text according to style, size, pen speed etc. The attention-mechanism assists to create a soft-window so that it dynamically determines an alignment between the text and the pen locations. That is, it helps model to learn to decide which character to write next.
![](https://miro.medium.com/v2/resize:fit:700/1*wa4zt-LcMWRIYLfiHfBKvA.png)

### Mixture Density Network (MDN)
Mixture Density Networks are neural networks which can measure their own uncertainty and help to capture randomness of handwriting. They output parameters μ(mean), σ(standard deviation), and ρ(correlation) for several multivariate Gaussian components. They also estimate a parameter π(weights) for each of these distributions which helps to gauge contribution of each distriution in the output.
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/1f53d4344df7e9670e7701be9594ce3f42ad2234/15-Figure1-1.png)

# Model Architecture
## Handwriting prediction
It consists of 3 LSTM stacked layers. The output is feeded to a mixture density layer which provides mean, variance, correlation and weights of 20 mixture components.
![](https://miro.medium.com/v2/resize:fit:1051/1*Hc2IazDoQm94gWIVNICVyA.png)
## Handwriting synthesis
The model consists of 3 LSTM layers stacked upon each other along with the added input from the character sequence mediated by the window layer. 

The input (x, y, eos) are taken from [IAM-OnDB online database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) where x and y are the pen co-ordinates and eos is the points in the sequence when the pen is lifted off the whiteboard.

Onehots is the onehot representation of text inputted

![](https://greydanus.github.io/assets/scribe/model_unrolled.png)
The output is (x, y, eos) of the hand-writing form of the input text which can be plotted to get the final output.

# Future works
Some of the ways in which this model can advance are listed below

- Controlling the width of the output handwriting. Introducting variance to it to give more realistic look.
- Generate handwriting in the style of a particular writer. This can be achieved with primed sampling

# Contributors
- Mohammed Bhadsorwala
- Sanika Kumbhare

# Acknowledgements
- [SRA VJTI](https://sravjti.in/) - Eklavya 2023
- Heartfelt gratitude to our mentors [Lakshaya Singhal](https://github.com/LakshayaSinghal) and [Advait Dhamorikar](https://github.com/advait-0) for guiding us at every point of our journey. The project wouldn't have been possible without them

# Resources
Deep Learning courses
- [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/week/1)
- [Improving Deep Neural Networks: Hyperparameter Tuning](https://www.coursera.org/learn/deep-neural-network/home/week/1)
- [Convolution Neural Network](https://www.coursera.org/learn/convolutional-neural-networks/home/week/1)
- [Sequence model](https://www.coursera.org/learn/nlp-sequence-models/home/week/1)
[Referenced Github repo](https://github.com/laihaotao/handwriting-synthesis)

[Referred for understanding subclassing](https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e)





