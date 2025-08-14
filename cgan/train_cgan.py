import tensorflow as tf
from generator import make_generator
from discriminator import make_discriminator

latent_dim = 100
num_classes = 2
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = make_generator(latent_dim, num_classes)
discriminator = make_discriminator(num_classes)

g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

# Use your load_dataset and train_step functions from the code you provided
