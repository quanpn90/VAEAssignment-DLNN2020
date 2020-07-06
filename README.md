# Variational Auto Encoder (VAE) task

This is the template for the 2nd assignment of the Deep Learning and Neural Network course.

The implementation of the feed-forward Auto-Encoder is shown, together with the template for the Variational AE.

# Requirement.

Python 3.7 and Numpy are the only requirement. If you have any problem running on Windows, please tell me. 

I have tested the code for both Windows and Linux under Anaconda: https://www.anaconda.com/products/individual. I recommend this because the Intel Math Kernel Librariy (MKL) will be automatically installed via Anaconda.

# Running.

First, ensure that you are in the same directory with the python files and the "data" directory with the "frey_rawface.mat" inside. Also provided are the model weights in model/ (when loading, they will override the hyperparameters in the train file) so you can directly run eval and sampling (below).

For the default Auto-Encoder you can run four things:

## Training
- Training the model on the GreyFace dataset https://cs.nyu.edu/~roweis/data/frey_rawface.jpg with 1965. You can manually change the hyperparameters (the neural network size, learning rate, etc) to play around with the code a little bit.

```
python train_ae train
```

## Gradient Checking
- Checking the gradient correctness. This step is normally important when implementing back-propagation. The idea of grad-check is actually very simple:

+ We need to know how to verify the correctness of the back-prop implementation.
+ In order to do that we rely on comparison with the gradients computed using numerical differentiation
+ For each weight in the network we will have to do the forward pass twice (one by increasing the weight by \delta, and one by decreasing the weight by \delta)
+ The difference between two forward passes gives us the gradient for that weight
+ (maybe the code will be self-explanationable)

```
python train_ae.py gradcheck
```


## Evaluation / Reconstruction
- Using the model for reconstruction. There are around 1965 images in the dataset, so you can enter the image ID from 0 to 1900 to see the image (left) and its reconstruction (right).

```
python train_ae.py eval
```

## Sampling 
- Using the model to sample from a random code. The code will then be randomly generated from a normal distribution N(0, I). This will be then passed to the decoder to generate the image. However, with this model I expect very much to see a **darkspawn** instead of human faces. The VAE model if successfully implemented, will be able to help us generate human faces from samples of a known distribution. I also provided a model with 256 hidden units and 16 latent units trained after 120000 steps and batch size 64. 
 
 ```
python train_ae.py sample
```

# Your Variational AE
I have already prepared the same template so that you can proceed with your implementation. The key is that the "middle" layer consists of two different components: the means and variances of the latent variable [1].  

The gradcheck should be very similar and the implementation should be able to pass it (If too many warnings happen and the relative difference is > 0.001 then its probably incorrect). One thing that needs to consider is the workaround of the sampling process which is stochastic and not differentiable. In [1] you can find the trick to "reparameterize" so that the additional noise becomes irrelevant to the back-propagation process, making it deterministic again. 

Then you can implement sampling and see how it generates the faces from known distribution. Reconstruction might be worse, so increase the hidden layer size might help (because this algorithm isn't exactly theoretically tight). 

The main goal of this assignment is to understand the back-propagation algorithm via implementing a stochastic process in which back-propagation is feasible only via reparameterization trick. I also hope that it can provide you with experience of training a neural network model and maybe enjoy (or not) the modeling samples.

# Reference
1. Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

# FAQs

1. RuntimeWarning: invalid value encountered in double_scalars rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)“ ... might happen. This can happen randomly when the denominator is 0 (or so small) and is not really a problem during gradcheck. 
2. The gradcheck might take too long. We should reduce the network size (hidden size and latent size) before doing gradcheck. There are 2 * (network size) + 1 forward passes to be ran in the check, so it is better to do on a small scale network.
3. My explanation is bad. I agree with this, if you find the paper and my writing not clear enough, I think this article can also help from another perspective: https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
