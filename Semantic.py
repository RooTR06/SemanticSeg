import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.vgg16 import VGG16


conv_base = VGG16(weights='imagenet',include, include_top=False)


#get layer names and make loadeded-model non trainable
for layer in conv_base.layers:
    print(conv_base.layers)
    layers.trainable = False



#extract VGG16 layers????
#layer_output = keras.Model.get_layer('vgg16').get_layer('block3_conv1').output
"""

layer3 =
layer4 =
layer7 =
"""
######FCN Layers#####

#1x1 conv
fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1)

# upsample to match VGG16 layer4
fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
kernel_size=4, strides=(2, 2), padding='SAME')

# skip connection between VGG layer4 and fcn9 
fcn9_skip = tf.add(fcn9, layer4)

# upsample to match VGG16 layer3
fcn10 = tf.layers.conv2d_transpose(fcn9_skip, filters=layer3.get_shape().as_list()[-1],
kernel_size=8, strides=(2, 2), padding='SAME')

# skip connection between VGG layer3 and fcn10
fcn10_skip = tf.add(fcn10, layer3)

# upsample to match input
fcn11 = tf.layers.conv2d_transpose(fcn10_skip, filters=num_classes,
kernel_size=16, strides=(8, 8), padding='SAME')


###################################################
#calculate distance from ground-truth pixels for loss and write a custom optimizer

#training

"""
num_classes = 3 (movable, drivable path and not movable objects)
IM_SHAPE = 
EPOCHS =
BATCH_SIZE = 
DROPOUT = 
"""