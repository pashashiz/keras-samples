from glob import glob
from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Dense, Flatten, \
    Conv2D, BatchNormalization, MaxPooling2D, Activation, add, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = [224, 224]
epochs = 16
batch_size = 128

ds_dir = expanduser('~/datasets/blood_cell_images')
train_path = ds_dir + '/TRAIN'
valid_path = ds_dir + '/TEST'

train_image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')
folders = glob(ds_dir + '/TRAIN/*')


def identity_block(input_, kernel_size, filters):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1))(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = add([x, input_])
    x = Activation('relu')(x)
    return x


def conv_block(input_, kernel_size, filters, strides):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1), strides=strides)(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(f3, (1, 1), strides=strides)(input_)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


# custom mini resnet
i = Input(shape=IMAGE_SIZE + [3])
x = ZeroPadding2D(padding=(3, 3))(i)
x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = ZeroPadding2D(padding=(1, 1))(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256])
x = identity_block(x, 3, [64, 64, 256])

x = conv_block(x, 3, [128, 128, 512], strides=(2, 2))
x = identity_block(x, 3, [128, 128, 512])
x = identity_block(x, 3, [128, 128, 512])
x = identity_block(x, 3, [128, 128, 512])

# our layers
x = Flatten()(x)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=i, outputs=prediction)

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


def preprocess_input2(x):
    x /= 127.5
    x -= 1.0
    return x


train_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input2
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input2
)

test_gen = val_gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=1,
    class_mode='sparse')

print("Let's look at the data...")
x, y = test_gen.next()

labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
    labels[v] = k

print('min:', x[0].min(), 'max:', x[0].max())
plt.title(labels[int(y[0])])
plt.imshow(x[0])
plt.show()

train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size,
    class_mode='sparse')

valid_generator = val_gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size,
    class_mode='sparse')

checkpoint_filepath = '/tmp/checkpoint'

r = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(train_image_files) // batch_size,
    validation_steps=len(valid_image_files) // batch_size,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
    ]
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(valid_generator)
print("Loss: {}, accuracy: {}".format(loss, accuracy))

print("Let's make a prediction...")
x, y = valid_generator.next()

labels = [None] * len(valid_generator.class_indices)
for k, v in valid_generator.class_indices.items():
    labels[v] = k

plt.title(labels[np.argmax(y[0])])
plt.imshow(x[0])
plt.show()

p_batch = model.predict(x)
print("Predicted: {}, Expected {}".format(
    labels[np.argmax(p_batch[0] > 0.5)], labels[np.argmax(y[0])]))

batches = len(valid_image_files) // batch_size

y_predicted = np.zeros((batches, batch_size), dtype=float)
y_expected = np.zeros((batches, batch_size), dtype=float)
valid_generator.reset()
for i in range(0, batches):
    x, y = valid_generator.next()
    y_predicted[i] = model.predict(x).argmax(axis=1)
    y_expected[i] = y

cm = confusion_matrix(y_expected.flatten(), y_predicted.flatten())
print("Confusion matrix:\n{}".format(cm))
