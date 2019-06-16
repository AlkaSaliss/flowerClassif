# import keras
from tensorflow import keras
import os


model = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=7, input_shape=(224, 224, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(3),
    keras.layers.Dropout(0.0),
    keras.layers.Conv2D(128, kernel_size=5, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(3),
    keras.layers.Dropout(0.0),
    keras.layers.Conv2D(256, kernel_size=3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(3),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(5, activation='softmax')
])

opt = keras.optimizers.Adam(lr=1e-3)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

log_dir = "C:\\Users\\A739748\\Documents\\COMPUTER_VISION\\TF\\flowerClassif\\logs"
os.makedirs(log_dir, exist_ok=True)
tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

t_path = "C:\\Users\\A739748\\Documents\\COMPUTER_VISION\\flowers-recognition\\split_data\\train"
v_path = "C:\\Users\\A739748\\Documents\\COMPUTER_VISION\\flowers-recognition\\split_data\\valid"

train_images = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_images = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_images.flow_from_directory(t_path, target_size=(224, 224), batch_size=64, class_mode='categorical')
val_gen = train_images.flow_from_directory(v_path, target_size=(224, 224), batch_size=64, class_mode='categorical')

model.fit_generator(train_gen, validation_data=val_gen, epochs=2, callbacks=[tb])