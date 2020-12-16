import os
import csv
import sys
import cv2
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

TRAIN_DIR = "./"
DRIVING_LOG = os.path.join(TRAIN_DIR, "driving_log.csv")
CORRECTION = 0.3


def extract_dl(driving_log_path):
    """Extract and load all training data in images directors and driving log csv file."""
    entries = []
    with open(driving_log_path) as csv_file:
        reader = csv.reader(csv_file)
        for entry in reader:
            entries.append(entry)
    empty_lists = [[] for i in range(7)]
    center_images, left_images, right_images, steerings, throttles, brakes, speeds = empty_lists
    for entry in entries:
        center_image_path, left_image_path, right_image_path = (entry[0], entry[1], entry[2])
        steering = float(entry[3])
        throttle = float(entry[4])
        brake = float(entry[5])
        speed = float(entry[6])
        center_image = cv2.imread(center_image_path)
        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)
        center_images.append(center_image)
        left_images.append(left_image)
        right_images.append(right_image)
        steerings.append(steering)
        throttles.append(throttle)
        brakes.append(brake)
        speeds.append(speed)
    return center_images, left_images, right_images, steerings, throttles, brakes, speeds


def augment_images(images, measurements, correction=0.0):
    """Augment out training image repository by adding flipped versions with inverted steering."""
    aug_imgs, aug_msrs = [], []
    for image, measurement, in zip(images, measurements):
        corr_msr = measurement + correction
        aug_imgs.append(image)
        aug_msrs.append(corr_msr)
        aug_imgs.append(cv2.flip(image, 1))
        aug_msrs.append(corr_msr * -1)
    return aug_imgs, aug_msrs


def plot_fit_history(fit_history_obj):
    """Plot loss and validation loss of the trained model on the same graph."""
    plt.plot(fit_history_obj.history['loss'])
    plt.plot(fit_history_obj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("learning_curve")
    plt.show()


if __name__ == '__main__':
    cr_imgs, lt_imgs, rt_imgs, steerings, throttles, brakes, speeds = extract_dl(DRIVING_LOG)
    aug_cr_imgs, aug_cr_msrs = augment_images(cr_imgs, steerings)
    aug_lt_imgs, aug_lf_msrs = augment_images(lt_imgs, steerings, correction=CORRECTION)
    aug_rt_imgs, aug_rt_msrs = augment_images(rt_imgs, steerings, correction=CORRECTION * -1)
    aug_imgs = aug_cr_imgs + aug_lt_imgs + aug_rt_imgs
    aug_msrs = aug_cr_msrs + aug_lf_msrs + aug_rt_msrs
    X_train = np.array(aug_imgs)
    y_train = np.array(aug_msrs)
    X_train,X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2)

    # Convolutional deep neural network based on the NVIDIA network.
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    modelCheckpoint = ModelCheckpoint("model-best.h5", monitor="val_loss", save_best_only=True, mode="min")
    # Fit the model. No need for generator as I am using a computer with 32GB RAM
    fit_hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), shuffle=True, epochs=20, callbacks=[modelCheckpoint])
    # fit_hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, callbacks=[modelCheckpoint])
    # model.save('model.h5')
    model.summary()
    plot_fit_history(fit_hist)

    model.evaluate(X_val,y_val)
    preds = model.predict(X_val)
# from pympler.asizeof import asizeof
#
# def memory():
#     for var, obj in list(globals().items()):
#         size = asizeof(obj)//1024**2
#         # if size > 0.0:
#         print(f"{var} is {size} MB")
