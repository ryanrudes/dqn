from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def make():
    inputs = Input(shape = (84, 84, 4))
    x = Conv2D(32, 8, 4, activation = 'relu')(inputs)
    x = Conv2D(64, 4, 2, activation = 'relu')(x)
    x = Conv2D(64, 3, 1, activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation = 'relu')(x)
    q = Dense(num_actions)(x)
    model = Model(inputs, q)
    return model
