import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels
    
def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


path_train_images='./stl10_binary/train_X.bin'
path_train_labels='./stl10_binary/train_y.bin'
train_images=read_all_images(path_train_images)
train_labels=read_labels(path_train_labels)
N=train_images.shape[0]
print(train_images.shape, train_labels.shape)
# normalize inputs from 0-255 to 0.0-1.0
train_images = (train_images.astype('float32'))/255.0

path_to_test_data='./stl10_binary/test_X.bin'
path_to_test_label='./stl10_binary/test_y.bin'
test_images=read_all_images(path_to_test_data)
test_labels=read_labels(path_to_test_label)
test_images = (test_images.astype('float32'))/255.0 # normalize inputs from 0-255 to 0.0-1.0



#define our model
def getModel():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
 
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
 
  

    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    model.summary()
    return model


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]
file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=10)


ready_final_train=1
if ready_final_train==1:
    #one hot encoding
    test_labels = np_utils.to_categorical(test_labels-1)  
    train_labels = np_utils.to_categorical(train_labels-1)  
    gmodel=getModel()
    mod2=gmodel.fit(train_images, train_labels,
          batch_size=40,
          epochs=5,
          verbose=1,
          validation_data=(test_images, test_labels),
          callbacks=callbacks)
    print(mod2.history['val_acc'])
    print(mod2.history['val_acc'][-1])



numpy_loss_history = np.array(mod2.history['loss'])
numpy_val_loss_history = np.array(mod2.history['val_loss'])
numpy_acc_history = np.array(mod2.history['acc'])
numpy_val_acc_history = np.array(mod2.history['val_acc'])
numpy.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
numpy.savetxt("val_loss_history.txt", numpy_val_loss_history, delimiter=",")
numpy.savetxt("acc_history.txt", numpy_acc_history, delimiter=",")
numpy.savetxt("val_acc_history.txt", numpy_val_acc_history, delimiter=",")

