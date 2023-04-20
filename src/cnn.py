import keras as keras
from keras import layers


def build_model(x_in):
    n_features1,n_features2,n_features3 = x_in.shape

    model = keras.Sequential(name="cnn")

    model.add(layers.Conv2D(filters=80, kernel_size=2, activation='relu', input_shape=(n_features1, n_features2, n_features3),data_format='channels_first'))
    # # model.add(layers.AveragePooling2D(pool_size=2,padding="same",data_format='channels_first'))
    model.add(layers.AveragePooling2D(pool_size=2,data_format='channels_first'))
    # model.add(layers.MaxPooling2D(pool_size=2,data_format='channels_first'))

    model.add(layers.Conv2D(filters=40, kernel_size=2, activation='relu',data_format='channels_first'))
    # # model.add(layers.AveragePooling2D(pool_size=2,padding="same",data_format='channels_first'))
    model.add(layers.MaxPooling2D(pool_size=2,data_format='channels_first'))

    model.add(layers.Conv2D(filters=20, kernel_size=1, activation='relu',data_format='channels_first'))
    # model.add(layers.AveragePooling2D(pool_size=2,data_format='channels_first'))


    # # model.add(layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',data_format='channels_first'))
    # # model.add(layers.MaxPooling2D(pool_size=2,data_format='channels_first'))
    model.add(layers.Flatten())

    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae'])
    return model
