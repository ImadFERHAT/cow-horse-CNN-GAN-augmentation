import tensorflow as tf
from tensorflow.keras import layers

def make_generator(latent_dim, num_classes):
    label_input = layers.Input(shape=(1,), dtype='int32')
    noise_input = layers.Input(shape=(latent_dim,))

    label_embedding = layers.Embedding(num_classes, latent_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    model_input = layers.Multiply()([noise_input, label_embedding])

    x = layers.Dense(4*4*256, use_bias=False)(model_input)
    x = layers.Reshape((4,4,256))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for filters in [128,64,32]:
        x = layers.Conv2DTranspose(filters,4,strides=2,padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(3,4,strides=2,padding='same',activation='tanh')(x)
    return tf.keras.Model([noise_input,label_input], x)
