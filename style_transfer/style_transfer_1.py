from __future__ import print_function, division

from builtins import range
from os.path import expanduser

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.preprocessing import image

tf.compat.v1.disable_eager_execution()


def vgg16_avg_pool(shape):
    # we want to account for features accros the entire image
    # so we get rid of maxpool which throws information
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    i = vgg.input
    x = i
    for layer in vgg.layers:
        if (layer.__class__ == MaxPooling2D):
            # replace with avg pooling
            x = AveragePooling2D()(x)
        else:
            x = layer(x)
    return Model(i, x)


def vgg16_avg_pool_cutoff(shape, num_convs):
    # there are 13 convolutions in total
    # we can pick any of them as the "output"
    # of out content model
    if (num_convs < 1 or num_convs > 13):
        raise Exception("num_convs must be in range [1, 13]")
    model = vgg16_avg_pool(shape)
    n = 0
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        if n >= num_convs:
            output = layer.output
            break
    return Model(model.input, output)


# VGG preprocesses the image, we have to do a reverse to that
def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


# just for plotting
def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x


if __name__ == '__main__':

    h, w = (384, 512)
    img = image.load_img(expanduser('~/datasets/style/pizza.jpg'), target_size=(h, w))

    # convert image to array and
    x = image.img_to_array(img)
    # add 1 batch dim
    x = np.expand_dims(x, axis=0)
    # process for vgg
    x = preprocess_input(x)

    plt.imshow(x[0])
    plt.show()

    batch_shape = x.shape
    shape = x.shape[1:]

    # make a content model up to 11th layer
    content_model = vgg16_avg_pool_cutoff(shape, 11)

    # the idea, we try to learn the input of a new image
    # so the output of new and original images would be as close a spossible
    target = K.variable(content_model.predict(x))
    loss = K.mean(K.square(target - content_model.output))
    grads = K.gradients(loss, content_model.input)

    # tensor function
    get_loss_and_grads = K.function(
        inputs=[content_model.input],
        outputs=[loss] + grads
    )


    def get_loss_and_grads_wrapper(x_vec):
        # scipy's minimizer allows us to pass back
        # function value f(x) and its gradient f'(x)
        # simultaneously, rather than using the fprime arg
        #
        # we cannot use get_loss_and_grads() directly
        # input to minimizer func must be a 1-D array
        # input to get_loss_and_grads must be [batch_of_images]
        #
        # gradient must also be a 1-D array
        # and both loss and gradient must be np.float64
        # will get an error otherwise
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)


    from datetime import datetime

    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(10):
        x, l, _ = fmin_l_bfgs_b(
            func=get_loss_and_grads_wrapper,
            x0=x,
            maxfun=20
        )
        x = np.clip(x, -127, 127)
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)
    print("duration:", datetime.now() - t0)

    plt.plot(losses)
    plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)

    plt.imshow(scale_img(final_img[0]))
    plt.show()
