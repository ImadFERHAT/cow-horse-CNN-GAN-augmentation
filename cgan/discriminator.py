import tensorflow as tf
from tensorflow.keras import layers

def make_discriminator(num_classes, img_size=64):
    image_input = layers.Input(shape=(img_size,img_size,3))
    label_input = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, img_size*img_size)(label_input)
    label_embedding = layers.Reshape((img_size,img_size,1))(label_embedding)

    x = layers.Concatenate()([image_input, label_embedding])

    for filters in [64,128,256]:
        x = layers.Conv2D(filters,4,strides=2,padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    return tf.keras.Model([image_input,label_input], x)
