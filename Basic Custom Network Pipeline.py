# Perform all installations
!pip install tensorflow-gpu==2.0.0
!pip install tensorflow-datasets
!pip install tensorwatch

# Get TensorBoard to run 
%load_ext tensorboard

# Import necessary packages
import tensorflow as tf
import tensorflow_datasets as tfds

# tfds makes a lot of progress bars and they take up a lot of screen space, so lets diable them
tfds.disable_progress_bar()

import math
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import CSVLogger

tf.random.set_seed(1234)
np.random.seed(1234)

# Variables
BATCH_SIZE = 32
NUM_EPOCHS= 20
IMG_H = IMG_W = 224
IMG_SIZE = 224
LOG_DIR = './log'
SHUFFLE_BUFFER_SIZE = 1024
IMG_CHANNELS = 3

# View all available datasets
print(tfds.list_builders())

dataset_name = "oxford_flowers102"

def preprocess(ds):
  x = tf.image.resize_with_pad(ds['image'], IMG_SIZE, IMG_SIZE)
  x = tf.cast(x, tf.float32)
  x = (x/127.5) - 1
  return x, ds['label']

def augmentation(image, label):
    #image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)

    # Random Crop: randomly crops an image and fits to given size
    #image = tf.image.random_crop(image,[IMG_SIZE, IMG_SIZE, IMG_CHANNELS])

    # Brightness: Adjust brightness by a given max_delta
    image = tf.image.random_brightness(image, .1)

    # Random Contrast: Add a random contrast to the image
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)

    # Flip: Left and right
    image = tf.image.random_flip_left_right(image)

    # Rotation: Only 90 degrees is currently supported.
    # Not all images still look the same after a 90 degree rotation
    # Most images are augmented by a 10-30 degree tilt
    #image = tf.keras.preprocessing.image.random_rotation(image,10)

    # Finally return the augmented image and label
    return image, label

def get_dataset(dataset_name, *split_value):
    # see all possible splits in dataset
    _, info = tfds.load(dataset_name, with_info=True)
    #print(info)
    if "validation" in info.splits:
        # then load train and validation
        if len(split_value) == 1:
            print(
                "INFO: Splitting train dataset according to splits provided by user"
            )
            if "test" in info.splits:
                print('INFO: Test dataset is available')
                all_data = tfds.Split.TEST + tfds.Split.TRAIN
                print("INFO: Added test data to train data")
                train, info_train = tfds.load(dataset_name,
                                              split=all_data,
                                              with_info=True)
                NUM_CLASSES = info_train.features['label'].num_classes
                NUM_EXAMPLES = info_train.splits['train'].num_examples
            else:
                all_data = tfds.Split.TRAIN
            split_train, _ = all_data.subsplit(
                weighted=[split_value[0], 100 - split_value[0]])
            # Load train with the new split
            train, info_train = tfds.load(dataset_name,
                                          split=split_train,
                                          with_info=True)
        else:
            #load training dataset the without any user intervention way
            print("INFO: Loading standard splits for training dataset")
            train, info_train = tfds.load(dataset_name,
                                          split=tfds.Split.TRAIN,
                                          with_info=True)
        # Load validation dataset as is standard
        val, info_val = tfds.load(dataset_name,
                                  split=tfds.Split.VALIDATION,
                                  with_info=True)
        NUM_CLASSES = info_train.features['label'].num_classes
        NUM_EXAMPLES = info_train.splits['train'].num_examples
    else:
        # Validation not in default datasets
        print(
            "INFO: Defining a 90-10 split between training and validation as no default split exists."
        )
        # Here we have defined how to split the original train dataset into train and val
        # Use 90% as train dataset and 10% as validation dataset
        train, info_train = tfds.load(dataset_name,
                                      split='train[:90%]',
                                      with_info=True)
        val, info_val = tfds.load(dataset_name,
                                  split='train[90%:]',
                                  with_info=True)
        NUM_CLASSES = info_train.features['label'].num_classes
        # The total number of classes in training dataset should either be equal to or more than the total number of classes in validation dataset
        assert NUM_CLASSES >= info_val.features['label'].num_classes
        NUM_EXAMPLES = info_train.splits['train'].num_examples * 0.9

    # Standard processing for training and validation set
    IMG_H, IMG_W, IMG_CHANNELS = info_train.features['image'].shape
    if IMG_H == None or IMG_H != IMG_SIZE:
        IMG_H = IMG_SIZE
    if IMG_W == None or IMG_W != IMG_SIZE:
        IMG_W = IMG_SIZE
    if IMG_CHANNELS == None:
        IMG_CHANNELS = 3

    # Training specific processing
    train = train.map(preprocess).repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(
        BATCH_SIZE)
    train = train.map(augmentation)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    # Validation specific processing
    val = val.map(preprocess).repeat().batch(BATCH_SIZE)
    val = val.prefetch(tf.data.experimental.AUTOTUNE)

    return train, info_train, val, info_val, IMG_H, IMG_W, IMG_CHANNELS, NUM_CLASSES, NUM_EXAMPLES

train, info_train, val, info_val, IMG_H, IMG_W, IMG_CHANNELS, NUM_CLASSES, NUM_EXAMPLES = get_dataset(
    dataset_name, 100)

print("\n\nIMG_H, IMG_W", IMG_H, IMG_W)
print("IMG_CHANNELS", IMG_CHANNELS)
print("NUM_CLASSES", NUM_CLASSES)
print("BATCH_SIZE", BATCH_SIZE)
print("NUM_EXAMPLES", NUM_EXAMPLES)
print("NUM_EPOCHS", NUM_EPOCHS)

# If you want to print even more information on both the splits, uncomment the lines below:
#print(info_train)
#print(info_val)

# Allow TensorBoard callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR,
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_grads=True,
                                                      batch_size=BATCH_SIZE,
                                                      write_images=True)

def create_model():
  model = tf.keras.Sequential([
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
          input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
     tf.keras.layers.Dropout(rate=0.3),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dropout(rate=0.3),
     tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
  ])
  return model 

def scratch(train, val, learning_rate):
  model = create_model()
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

  earlystop_callback = tf.keras.callbacks.EarlyStopping(
                 monitor='val_accuracy', 
                 min_delta=0.0001, 
                 patience=5)

  model.fit(train,
           epochs=NUM_EPOCHS,
           steps_per_epoch=int(NUM_EXAMPLES/BATCH_SIZE),
           validation_data=val, 
           validation_steps=1,
           validation_freq=1,
           callbacks=[tensorboard_callback, earlystop_callback])
  return model
  
def transfer_learn(train, val, unfreeze_percentage, learning_rate):
    mobile_net = tf.keras.applications.MobileNet(input_shape=(IMG_SIZE,
                                                              IMG_SIZE,
                                                              IMG_CHANNELS),
                                                 include_top=False)
    # Use mobile_net.summary() to view the model
    mobile_net.trainable = False
    # Unfreeze some of the layers according to the dataset being used
    num_layers = len(mobile_net.layers)
    for layer_index in range(
            int(num_layers - unfreeze_percentage * num_layers), num_layers):
        mobile_net.layers[layer_index].trainable = True
    model_with_transfer_learning = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')], )
    model_with_transfer_learning.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])
    model_with_transfer_learning.summary()
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.0001, patience=10)
    csv_logger = CSVLogger('colorectal-transferlearn-' + 'log.csv',
                           append=True,
                           separator=';')
    model_with_transfer_learning.fit(
        train,
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(NUM_EXAMPLES / BATCH_SIZE),
        validation_data=val,
        validation_steps=1,
        validation_freq=1,
        callbacks=[tensorboard_callback, earlystop_callback, csv_logger])
    return model_with_transfer_learning

# Start TensorBoard
%tensorboard --logdir ./log

# select the percentage of layers to be trained while using the transfer learning
# technique. The selected layers will be close to the output/final layers.
unfreeze_percentage = 0

learning_rate = 0.001

if training_format == "scratch":
    print("Training a model from scratch")
    model = scratch(train, val, learning_rate)
elif training_format == "transfer_learning":
    print("Fine Tuning the MobileNet model")
    model = transfer_learn(train, val, unfreeze_percentage, learning_rate)

# Save the model to load it in the What-If tool
tf.saved_model.save(model, "tmp/model/1/")

# Load the saved model
loaded = tf.saved_model.load("tmp/model/1")
print(list(loaded.signatures.keys()))  # ["serving_default"]

# Zip the directory so that we can download it
!zip model.zip  tmp/model/* 

# If you are running this in Google Colab,
# Go to the content directory and download the trained model
!pwd
