from glob import glob
from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = [100, 100]
epochs = 3
batch_size = 128

ds_dir = expanduser('~/datasets/fruits-360')
train_path = ds_dir + '/Training'
valid_path = ds_dir + '/Validation'

train_image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')
folders = glob(ds_dir + '/Training/*')

plt.imshow(image.load_img(np.random.choice(train_image_files)))

res = ResNet50(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False)

for layer in res.layers:
    layer.trainable = False

# our layers
x = Flatten()(res.output)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=res.inputs, outputs=prediction)

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# train
train_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size)

valid_generator = val_gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size)

r = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(train_image_files) // batch_size,
    validation_steps=len(valid_image_files) // batch_size)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# accuracy
loss, accuracy = model.evaluate(valid_generator)
print("Loss: {}, accuracy: {}".format(loss, accuracy))  # ~97%

# let's make a prediction
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

# confusion matrix
batches = len(valid_image_files) // batch_size
y_predicted = np.zeros((batches, batch_size), dtype=float)
y_expected = np.zeros((batches, batch_size), dtype=float)
valid_generator.reset()
for i in range(0, batches):
    x, y = valid_generator.next()
    y_predicted[i] = model.predict(x).argmax(axis=1)
    y_expected[i] = y.argmax(axis=1)

cm = confusion_matrix(y_expected.flatten(), y_predicted.flatten())
print("Confusion matrix:\n{}".format(cm))
