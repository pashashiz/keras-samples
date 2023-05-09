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


def gram_matrix(img):
    # input (H, W, C) -> (C, W*H)
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
    # gram matrix = X dot XT / N
    G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
    return G


def style_loss(y, t):
    return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


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


def load_img_and_preprocess(path, shape=None):
    img = image.load_img(path, target_size=shape)
    # convert image to array and
    x = image.img_to_array(img)
    # add 1 batch dim
    x = np.expand_dims(x, axis=0)
    # process for vgg
    x = preprocess_input(x)
    return x


def minimize(fn, epochs, batch_shape):
    from datetime import datetime
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = fmin_l_bfgs_b(
            func=fn,
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
    return final_img[0]


if __name__ == '__main__':

    h, w = (384, 512)

    content_img = load_img_and_preprocess(expanduser('~/datasets/style/pizza.jpg'), (h, w))

    plt.imshow(content_img[0])
    plt.show()

    style_img = load_img_and_preprocess(expanduser('~/datasets/style/starry_night.jpg'), (h, w))

    plt.imshow(style_img[0])
    plt.show()

    batch_shape = content_img.shape
    shape = content_img.shape[1:]

    vgg = vgg16_avg_pool(shape)

    content_model = Model(vgg.input, vgg.layers[13].get_output_at(1))

    content_target = K.variable(content_model.predict(content_img))

    # Note: need to select output at index 1, since outputs at
    # index 0 correspond to the original vgg with maxpool
    symbolic_conv_outputs = [
        layer.get_output_at(1) for layer in vgg.layers \
        if layer.name.endswith('conv1')
    ]

    # pick the earlier layers for
    # a more "localized" representation
    # this is opposed to the content model
    # where the later layers represent a more "global" structure
    # symbolic_conv_outputs = symbolic_conv_outputs[:2]

    # make a big model that outputs multiple layers' outputs
    style_model = Model(vgg.input, symbolic_conv_outputs)

    # calculate the targets that are output at each layer
    style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]

    style_weights = [1, 2, 3, 4, 5]

    # calculate the total content loss
    loss = K.mean(K.square(content_model.output - content_target))

    # add style loss
    for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
        # gram_matrix() expects a (H, W, C) as input
        loss += w * style_loss(symbolic[0], actual[0])

    grads = K.gradients(loss, vgg.input)

    # just like theano.function
    get_loss_and_grads = K.function(
        inputs=[vgg.input],
        outputs=[loss] + grads
    )


    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)


    final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)

    plt.imshow(scale_img(final_img))
    plt.show()
