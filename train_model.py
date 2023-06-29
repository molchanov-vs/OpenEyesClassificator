import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU

def train_base_model(train_X, train_label, valid_X, valid_label, batch_size, epochs):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(24,24,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))             
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.legacy.Adam(),
        metrics=['accuracy'])
    
    history = model.fit(train_X, train_label, 
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=0,
                             validation_data=(valid_X, valid_label))

    return history, model


def train_upd_model(train_X, train_label, valid_X, valid_label, batch_size, epochs):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(24,24,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))                  
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.legacy.Adam(),
        metrics=['accuracy'])
    
    history = model.fit(train_X, train_label, 
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=0,
                             validation_data=(valid_X, valid_label))

    return history, model