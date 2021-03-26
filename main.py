import keras
from keras import layers



def train_model():

    model = keras.Sequential()
    model.add(layers.Dense(4, input_dim=8, activation='relu'))
    model.add(layers.Dense(2,activation='relu'))

    return model


