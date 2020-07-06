import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import sys
import pickle

import scipy.io

debug = False

img_rows = 28
img_cols = 20
ff = scipy.io.loadmat('data/frey_rawface.mat')
ff = ff["ff"].T.reshape((-1, 1, img_rows, img_cols))
ff = ff.astype('float32') / 255.
print(ff.shape)

n_samples = ff.shape[0]

input_size = 560
hidden_size = 256
latent_size = 16
std = 0.02
learning_rate = 0.01
loss_function = 'bce'  # mse or bce


def get_minibatch(batch_size, idx=0, indices=None):
    start_idx = batch_size * idx
    end_idx = min(start_idx + batch_size, n_samples)

    if indices is None:
        sample_b = ff[start_idx:end_idx]
    else:
        idx = indices[start_idx:end_idx]
        sample_b = ff[idx]

    sample_b = np.resize(sample_b, (batch_size, 560))

    sample_b = np.transpose(sample_b, (1, 0))

    return sample_b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid function
def dsigmoid(y):
    return y * (1 - y)


# The derivative of the tanh function
def dtanh(y):
    return 1 - y * y


def sample_unit_gaussian(latent_size):
    return np.random.standard_normal(size=(latent_size))


def relu(x):
    x[x < 0] = 0

    return x


def drelu(y):
    return 1. * (y > 0)


# Initialization was done according to Kingma et al. 2014.
# input to hidden weight
# Wi = np.random.randn(hidden_size, input_size) * std
Wi = np.random.uniform(-std, std, size=(hidden_size, input_size))
Bi = np.random.uniform(-std, std, size=(hidden_size, 1))
# Bi = np.random.randn(hidden_size, 1) * std
# encoding weight (hidden to code mean)
# Wm = np.random.randn(latent_size, hidden_size) * std
Wm = np.random.uniform(-std, std, size=(latent_size, hidden_size))
Bm = np.random.uniform(-std, std, size=(latent_size, 1))
# Bm = np.random.randn(latent_size, 1) * std

# weight mapping code to hidden
# Wd = np.random.randn(hidden_size, latent_size) * std
Wd = np.random.uniform(-std, std, size=(hidden_size, latent_size))
Bd = np.random.uniform(-std, std, size=(hidden_size, 1))
# Bd = np.random.randn(hidden_size, 1) * std
# decoded hidden to output
# Wo = np.random.randn(input_size, hidden_size) * std
Wo = np.random.uniform(-std, std, size=(input_size, hidden_size))
Bo = np.random.uniform(-std, std, size=(input_size, 1))
# Bo = np.random.randn(input_size, 1) * std


def forward(input):
    if debug:
        print("input shape:", input.shape)

    if input.ndim == 1:
        input = np.expand_dims(input, axis=1)

    batch_size = input.shape[-1]

    # (1) linear
    h = np.dot(Wi, input) + Bi

    # (2) ReLU
    h = relu(h)

    # # (3) h > z
    z = np.dot(Wm, h) + Bm

    # (4) relu and residual
    z = relu(z)

    # (5) z > dec
    dec = np.dot(Wd, z) + Bd

    # (6) ReLU
    dec = relu(dec)

    # (7) dec to output
    output = np.dot(Wo, dec) + Bo

    # # (8) dec to p(x)
    # and (9) loss function
    if loss_function == 'bce':
        p = sigmoid(output)
        loss = -np.sum(np.multiply(input, np.log(p)) + np.multiply(1 - input, np.log(1 - p)))

    elif loss_function == 'mse':
        p = output
        loss = np.sum(0.5 * (p - input) ** 2)

    if debug:
        print("output shape: ", p.shape)

    activations = (h, z, dec, output, p)

    return loss, activations


def backward(input, activations, scale=True):
    # allocating the gradients for the weight matrice
    dWi = np.zeros_like(Wi)
    dWm = np.zeros_like(Wm)
    dWd = np.zeros_like(Wd)
    dWo = np.zeros_like(Wo)
    dBi = np.zeros_like(Bi)
    dBm = np.zeros_like(Bm)
    dBd = np.zeros_like(Bd)
    dBo = np.zeros_like(Bo)

    if input.ndim == 2:
        batch_size = input.shape[-1]
    else:
        batch_size = 1

    h, z, dec, output, p = activations

    # backprop from (8) and (9) (if there is an additional activation function)
    if loss_function == 'mse':
        dl_dp = p - input

        # I found that normalizing the loss and gradient by batch size makes learning more stable
        if scale:
            dl_dp = dl_dp / batch_size
        dl_doutput = dl_dp

    elif loss_function == 'bce':
        dl_dp = -1 * (input / p - (1 - input) / (1 - p))
        dl_dp = dl_dp
        if scale:
            dl_dp = dl_dp / batch_size
        dl_doutput = np.multiply(dl_dp, dsigmoid(p))

    # backprop from (7) through fully-connected
    dl_ddec = np.dot(Wo.T, dl_doutput)
    dWo += np.dot(dl_doutput, dec.T)
    if batch_size == 1:
        dBo += dl_doutput
    else:
        dBo += np.sum(dl_doutput, axis=-1, keepdims=True)

    # backprop from (6) through ReLU
    dl_ddec = np.multiply(drelu(dec), dl_ddec)

    # backprop from (5) through fully-connected
    dl_dz = np.dot(Wd.T, dl_ddec)
    dWd += np.dot(dl_ddec, z.T)
    if batch_size == 1:
        dBd += dl_ddec
    else:
        dBd += np.sum(dl_ddec, axis=-1, keepdims=True)

    # backprop from (4) through ReLU
    dl_dz = np.multiply(drelu(z), dl_dz)

    # # backprop from (3) through fully connected
    dl_dh = np.dot(Wm.T, dl_dz)
    dWm += np.dot(dl_dz, h.T)
    if batch_size == 1:
        dBm += dl_dz
    else:
        dBm += np.sum(dl_dz, axis=-1, keepdims=True)

    # # # backprop from (2) through ReLU
    dl_dh = np.multiply(drelu(h), dl_dh)

    # backprop from (1) through fully connected
    dl_dinput = np.dot(Wi.T, dl_dh)
    dWi += np.dot(dl_dh, input.T)
    if batch_size == 1:
        dBi += dl_dh
    else:
        dBi += np.sum(dl_dh, axis=-1, keepdims=True)

    gradients = (dWi, dWm, dWd, dWo, dBi, dBm, dBd, dBo)

    return gradients


def train():
    # Momentums for adagrad
    mWi, mWm, mWd, mWo = np.zeros_like(Wi), np.zeros_like(Wm), np.zeros_like(Wd), np.zeros_like(Wo)
    mBi, mBm, mBd, mBo = np.zeros_like(Bi), np.zeros_like(Bm), np.zeros_like(Bd), np.zeros_like(Bo)

    def save_weights():

        print("Saving weights to %s and moments to %s" % ('weights.pkl', 'momentums.pkl'))

        weights = (Wi, Wm, Wd, Wo, Bi, Bm, Bd, Bo)
        with open('models/weights.pkl', 'wb') as output:
            pickle.dump(weights, output, pickle.HIGHEST_PROTOCOL)

        momentums = (mWi, mWm, mWd, mWo, mBi, mBm, mBd, mBo)
        with open('models/momentums.pkl', 'wb') as output:
            pickle.dump(momentums, output, pickle.HIGHEST_PROTOCOL)

        return

    batch_size = 64
    n_epoch = 100000

    save_every = 2000

    # first we have to shuffle the data
    n_samples = ff.shape[0]
    indices = np.arange(n_samples)
    total_loss = 0
    total_pixels = 0
    count = 0

    n_minibatch = math.ceil(n_samples / batch_size)
    for epoch in range(n_epoch):

        rand_indices = np.random.permutation(indices)

        for i in range(n_minibatch):

            x_i = get_minibatch(batch_size, i, rand_indices)
            bsz = x_i.shape[-1]

            loss, acts = forward(x_i)

            total_loss += loss
            total_pixels += bsz * 560

            gradients = backward(x_i, acts)

            dWi, dWm, dWd, dWo, dBi, dBm, dBd, dBo = gradients

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([Wi, Wm, Wd, Wo,
                                           ],
                                          [dWi, dWm, dWd, dWo, dBi, dBm, dBd, dBo],
                                          [mWi, mWm, mWd, mWo, mBi, mBm, mBd, mBo]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
                # param += -learning_rate * dparam

            count += 1

            if count % 50 == 0:
                avg_loss = total_loss / total_pixels
                print("Epoch %d Iteration %d Updates %d Loss per pixel %0.6f " % (epoch, i, count, avg_loss))

            # save weights to file every 500 updates so we can load to visualize later
            if count % 500 == 0:
                save_weights()

    return


def grad_check():
    batch_size = 8
    delta = 0.0001

    x = get_minibatch(batch_size)

    loss, acts = forward(x)

    gradients = backward(x, acts, scale=False)

    dWi, dWm, dWd, dWo, dBi, dBm, dBd, dBo = gradients

    for weight, grad, name in zip([Wi, Wm, Wd, Wo, Bi, Bm, Bd, Bo], [dWi, dWm, dWd, dWo, dBi, dBm, dBd, dBo],
                                  ['Wi', 'Wm', 'Wd', 'Wo', 'Bi', 'Bm', 'Bd', 'Bo']):

        str_ = ("Dimensions dont match between weight %s and gradient %s and %s." % (name, weight.shape, grad.shape))
        assert (weight.shape == grad.shape), str_

        print(name)
        n_warnings = 0
        # print(weight, grad)
        for i in range(weight.size):

            w = weight.flat[i]

            weight.flat[i] = w + delta
            loss_positive, _ = forward(x)

            weight.flat[i] = w - delta
            loss_negative, _ = forward(x)

            weight.flat[i] = w  # reset old value for this parameter

            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)

            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)

            if rel_error > 0.001:
                n_warnings += 1
                # print('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
        print("%d gradient mismatch warnings found in these weights. " % n_warnings)

    return


def decode(z):
    dec = np.dot(Wd, z) + Bd
    dec = relu(dec)
    output = np.dot(Wo, dec) + Bo
    if loss_function == 'bce':
        p = sigmoid(output)

    elif loss_function == 'mse':
        p = output

    return p


def eval():
    while True:

        # read weights from file
        cmd = input("Enter an image number:  ")

        img_idx = int(cmd)

        if img_idx < 0:
            exit()

        fig = plt.figure(figsize=(4, 4))

        sample_ = ff[img_idx]
        org_img = sample_ * 255
        sample_ = np.resize(sample_, (1, 560)).T
        sample_ = sample_.flatten()

        loss, act = forward(sample_)

        h, z, dec, output, p = act
        img = p * 255

        print(loss)

        fig.add_subplot(1, 2, 1)
        plt.imshow(org_img.reshape(28, 20), cmap='gray')

        fig.add_subplot(1, 2, 2)
        plt.imshow(img.reshape(28, 20), cmap='gray')
        # plt.title('reconstructed face %d' % 0)
        plt.show(block=True)

        print("Done")


def sample():
    while True:
        cmd = input("Press anything to continue:  ")

        z = np.random.randn(latent_size)
        z = np.expand_dims(z, axis=1)

        p = decode(z)

        print(p.shape)
        img = p

        fig = plt.figure(figsize=(2, 2))

        plt.imshow(img.reshape(28, 20), cmap='gray')
        plt.show(block=True)


if len(sys.argv) != 2:
    print("Need an argument train or gradcheck or eval or sample")
    exit()

option = sys.argv[1]

if option == 'train':
    train()
elif option in ['grad_check', 'gradcheck']:
    grad_check()
elif option in ['eval', 'sample']:

    # read trained weights from file
    with open('models/weights.pkl', "rb") as f:
        weights = pickle.load(f)

    Wi, Wm, Wd, Wo, Bi, Bm, Bd, Bo = weights

    if option == 'eval':
        eval()
    else:
        sample()
else:
    raise NotImplementedError
