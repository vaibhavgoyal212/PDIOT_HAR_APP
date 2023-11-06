import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D


def load_data():
    data_path = os.path.join(os.getcwd(), 'data')
    data_folders = os.listdir(data_path)
    sensor = "Respeck"
    cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'activity']
    all_data = os.path.join(os.getcwd(), 'train_data.csv')
    pd.DataFrame(columns=cols).to_csv(all_data, mode='w', header=True, index=False)
    for folder in data_folders:
        folder_dir = os.path.join(data_path, folder)
        files = os.listdir(folder_dir)
        for file in files:
            if file.endswith(".csv") and sensor in file and "clean" in file:
                df = pd.read_csv(os.path.join(folder_dir, file))
                act, subact = file.split('_')[2:4]
                df['activity'] = pd.Series(f"{act} {subact}", index=df.index)
                df = df[cols]
                df.to_csv(all_data, mode='a', header=False, index=False)


random_seed = 42
n_time_steps = 50
n_features = 6
step = 10
n_epochs = 40
batch_size = 64
learning_rate = 0.0025
l2_loss = 0.0015


def set_data():
    segments = []
    labels = []
    df = pd.read_csv(os.path.join(os.getcwd(), 'train_data.csv'))

    for i in range(0, len(df) - n_time_steps, step):
        xs = df['accel_x'].values[i: i + n_time_steps]
        ys = df['accel_y'].values[i: i + n_time_steps]
        zs = df['accel_z'].values[i: i + n_time_steps]
        gx = df['gyro_x'].values[i: i + n_time_steps]
        gy = df['gyro_y'].values[i: i + n_time_steps]
        gz = df['gyro_z'].values[i: i + n_time_steps]
        label = stats.mode(df['activity'][i: i + n_time_steps])[0][0]
        segments.append([xs, ys, zs, gx, gy, gz])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, n_time_steps, n_features)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    x_train, x_test, y_train, y_test = train_test_split(reshaped_segments, labels,
                                                        test_size=0.2, random_state=random_seed)

    return x_train, x_test, y_train, y_test


def set_model():

    model_1 = Sequential()
    model_1.add(LSTM(100, input_shape=(n_time_steps, n_features)))
    model_1.add(Dropout(0.5))
    model_1.add(Dense(100, activation='relu'))
    model_1.add(Dense(44, activation='softmax'))
    model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_1


def set_model_2(y_train):
    n_output_shape = y_train.shape[1]
    model_2 = Sequential()
    model_2.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_time_steps, n_features)))
    model_2.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model_2.add(Dropout(0.5))
    model_2.add(MaxPooling1D(pool_size=2))
    model_2.add(Flatten())
    model_2.add(Dense(100, activation='relu'))
    model_2.add(Dense(n_output_shape, activation='softmax'))
    print(model_2.summary())
    #plot the model
    keras.utils.plot_model(model_2, to_file='model.png')
    model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_2


def evaluate_model(model, x_test, y_test):
    model.evaluate(x_test, y_test)


def save_transform_model(model):
    model.save('model.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # save tflite model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(40, 40))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, cbar=False)
    plt.title("Confusion matrix", fontsize=30)
    plt.xticks(np.arange(len(classes)), classes, fontsize=20)
    plt.yticks(np.arange(len(classes)), classes, fontsize=20)
    plt.xlabel("Predicted Label", fontsize=30)
    plt.ylabel("True Label", fontsize=30)
    plt.show()


def main():
    x_train, x_test, y_train, y_test = set_data()
    model = set_model_2(y_train)
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))
    evaluate_model(model, x_test, y_test)
    save_transform_model(model)


main()
