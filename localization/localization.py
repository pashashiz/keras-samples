from glob import glob
from os.path import expanduser

import matplotlib.pyplot as plt
import numpy as np
from imageio.v2 import imread
from skimage.transform import resize
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Concatenate
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from matplotlib.patches import Rectangle

ds_dir = expanduser('~/datasets/localization/')
backgounds = []
backgound_files = glob(ds_dir + 'backgrounds/*.jpg')
for f in backgound_files:
    # might have different sizes
    bg = np.array(image.load_img(f))
    backgounds.append(bg)

plt.imshow(backgounds[0])

ch = imread(ds_dir + 'charmander-tight.png')
bb = imread(ds_dir + 'bulbasaur-tight.png')
sq = imread(ds_dir + 'squirtle-tight.png')

plt.imshow(sq)
plt.show()

POKE_DIM = 200
ch = np.array(ch)
bb = np.array(bb)
sq = np.array(sq)
CH_H, CH_W, CH_C = ch.shape
BB_H, BB_W, BB_C = bb.shape
SQ_H, SQ_W, SQ_C = sq.shape

poke_data = [
    [ch, CH_H, CH_W, CH_C],
    [bb, BB_H, BB_W, BB_C],
    [sq, SQ_H, SQ_W, SQ_C]
]
class_names = ['charmander', 'bulbasaur', 'squirtle']


def image_generator(batch_size=64):
    while True:
        # each epoch will have 50 batches, why? no reason
        for _ in range(50):
            X = np.zeros((batch_size, POKE_DIM, POKE_DIM, 3))
            Y = np.zeros((batch_size, 8))
            for i in range(batch_size):
                # select a random background
                bg_index = np.random.choice(len(backgounds))
                bg = backgounds[bg_index]
                bg_h, bg_w, _ = bg.shape
                # and choose random area from it with POKE_DIM size
                rnd_h = np.random.randint(bg_h - POKE_DIM)
                rnd_w = np.random.randint(bg_w - POKE_DIM)
                X[i] = bg[rnd_h:rnd_h + POKE_DIM, rnd_w:rnd_w + POKE_DIM].copy()
                # appearance
                appear = (np.random.random() < 0.75)
                if appear:
                    # choose an object
                    pk_idx = np.random.choice(3)
                    pk, h, w, _ = poke_data[pk_idx]
                    # resize obj image
                    scale = 0.5 + np.random.random()
                    new_height = int(h * scale)
                    new_width = int(h * scale)
                    obj = resize(pk, (new_height, new_width), preserve_range=True).astype(np.uint8)
                    # flip
                    if np.random.random() < 0.5:
                        obj = np.fliplr(obj)
                    # choose slice
                    row0 = np.random.randint(POKE_DIM - new_height)
                    col0 = np.random.randint(POKE_DIM - new_width)
                    row1 = row0 + new_height
                    col1 = col0 + new_width
                    # merge obj with background
                    mask = (obj[:, :, 3] == 0)  # find where obj is transparent
                    bg_slice = X[i, row0:row1, col0:col1, :]  # object area
                    bg_slice = np.expand_dims(mask, -1) * bg_slice  # remove object
                    bg_slice += obj[:, :, :3]  # add obj to the slice
                    X[i, row0:row1, col0:col1, :] = bg_slice  # merge img with slice
                    # tragets
                    Y[i, 0] = row0 / POKE_DIM
                    Y[i, 1] = col0 / POKE_DIM
                    Y[i, 2] = (row1 - row0) / POKE_DIM
                    Y[i, 3] = (col1 - col0) / POKE_DIM
                    Y[i, 4 + pk_idx] = 1
                Y[i, 7] = float(appear)
            yield X / 255., Y


gen = image_generator()

plt.imshow(next(gen)[0][0])


def custom_loss(y_true, y_pred):
    # target is: (row, col, depth, width, class1, class2, class3, object_appeared)
    bce = binary_crossentropy(y_true[:, :4], y_pred[:, :4])  # location
    cce = categorical_crossentropy(y_true[:, 4:7], y_pred[:, 4:7])  # object class
    bce2 = binary_crossentropy(y_true[:, -1], y_pred[:, -1])  # object appeared
    return bce * y_true[:, -1] + cce * y_true[:, -1] + 0.5 * bce2


def make_model():
    vgg = VGG16(input_shape=[POKE_DIM, POKE_DIM, 3], weights='imagenet', include_top=False)
    x = Flatten()(vgg.output)
    x1 = Dense(4, activation='sigmoid')(x)  # location
    x2 = Dense(3, activation='softmax')(x)  # object class
    x3 = Dense(1, activation='sigmoid')(x)  # object appeared
    x = Concatenate()([x1, x2, x3])
    model = Model(vgg.input, x)
    model.compile(loss=custom_loss, optimizer=Adam(learning_rate=0.0001))
    return model


model = make_model()
model.fit(image_generator(), steps_per_epoch=50, epochs=3)


def make_predictions():
    # select a random background
    bg_index = np.random.choice(len(backgounds))
    bg = backgounds[bg_index]
    bg_h, bg_w, _ = bg.shape
    # and choose random area from it with POKE_DIM size
    rnd_h = np.random.randint(bg_h - POKE_DIM)
    rnd_w = np.random.randint(bg_w - POKE_DIM)
    x = bg[rnd_h:rnd_h + POKE_DIM, rnd_w:rnd_w + POKE_DIM].copy()
    # appearance
    appear = (np.random.random() < 0.75)
    if appear:
        # choose an object
        pk_idx = np.random.choice(3)
        pk, h, w, _ = poke_data[pk_idx]
        # resize obj image
        scale = 0.5 + np.random.random()
        new_height = int(h * scale)
        new_width = int(w * scale)
        obj = resize(pk, (new_height, new_width), preserve_range=True).astype(np.uint8)
        # flip
        if np.random.random() < 0.5:
            obj = np.fliplr(obj)
        # choose slice
        row0 = np.random.randint(POKE_DIM - new_height)
        col0 = np.random.randint(POKE_DIM - new_width)
        row1 = row0 + new_height
        col1 = col0 + new_width
        # merge obj with background
        mask = (obj[:, :, 3] == 0)  # find where obj is transparent
        bg_slice = x[row0:row1, col0:col1, :]  # object area
        bg_slice = np.expand_dims(mask, -1) * bg_slice  # remove object
        bg_slice += obj[:, :, :3]  # add obj to the slice
        x[row0:row1, col0:col1, :] = bg_slice  # merge img with slice
        print("true:", row0, col0, row1, col1)
    # predict
    X = np.expand_dims(x, 0) / 255.
    p = model.predict(X)[0]
    # plot
    fig, ax = plt.subplots(1)
    ax.imshow(x / 255.)
    # draw the box
    if p[7] > 0.5:
        row0 = int(p[0] * POKE_DIM)
        col0 = int(p[1] * POKE_DIM)
        row1 = int(row0 + p[2] * POKE_DIM)
        col1 = int(col0 + p[3] * POKE_DIM)
        class_pred_idx = np.argmax(p[4:7])
        class_pred = class_names[class_pred_idx]
        print("pred:", p[7], class_pred, row0, col0, row1, col1)
        rect = Rectangle((col0, row0), col1 - col0, row1 - row0, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    else:
        print("not pred:", p[7])
    plt.show()

make_predictions()
make_predictions()
make_predictions()
make_predictions()
make_predictions()
