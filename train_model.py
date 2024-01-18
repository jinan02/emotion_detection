import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from PIL import Image
import scipy

gen_train = ImageDataGenerator(rescale=1. / 255)
gen_valid = ImageDataGenerator(rescale=1. / 255)

train_data = gen_train.flow_from_directory(
    'archive/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_data = gen_valid.flow_from_directory(
    'archive/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# model structure

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

optimizer = tf.keras.optimizers.legacy.Adam(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# training phase
model_info = model.fit(
    train_data,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_data,
    validation_steps=7178 // 64
)

model_json = model.to_json()


with open("model/emotion_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model/model.h5')


