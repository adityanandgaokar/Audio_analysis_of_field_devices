from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from tensorflow.keras.models import Model
from keras import regularizers
from tensorflow.keras.models import load_model


def Alexnet(no_classes, sample_rate, sample_time):

    if K.image_data_format() == "channels_first":
        
        input_shape = (1, 40, 969)
        channeldim = 1
        #input_shape = (257,63,1)
    else:
        input_shape = (40,969,1)
        channeldim = -1


    model = Sequential()
    model.add(Conv2D(96, 11, strides= 4, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(3,2, padding='same'))
    model.add(Conv2D(256,5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(3,2, padding='same'))
    model.add(Conv2D(384, 3, strides= 1, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, strides= 1, padding='same', activation='relu'))
    model.add(MaxPooling2D(3,2, padding='same'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_classes, activation='softmax'))

    model.summary()

    model.compile(optimizer= 'sgd', loss='categorical_crossentropy',
                  metrics = ['accuracy'])

    return model

