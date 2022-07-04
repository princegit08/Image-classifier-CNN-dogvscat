
from google.colab import drive
drive.mount('/content/drive')

!unzip '/content/drive/MyDrive/catvsdog/data.zip'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

train_path = '/content/data/train'
valid_path = '/content/data/valid'
test_path = '/content/data/test'

train_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=5)
valid_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=5)
test_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=5, shuffle=False)

imgs, labels = next(train_set)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),

    Flatten(),
    Dense(units=2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    x=train_set, steps_per_epoch=len(train_set),
    validation_data=valid_set, validation_steps=len(valid_set),
    epochs=10,
)
model.fit(
    x=train_set, steps_per_epoch=len(train_set),
    validation_data=valid_set, validation_steps=len(valid_set),
    epochs=5,
)

model.save('/content/model_byprince.h5')