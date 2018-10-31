import tensorflow as tf
import numpy as np

initializer = tf.variance_scaling_initializer()
activation_fn = tf.nn.elu
lr = 1e-3
gamma = 0.95

class BEGAN:

    def __init__(self, num_filters, num_units, dim_latent):
        self.num_filters = num_filters
        self.num_units = num_units

        self.input      = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 4))
        self.z          = tf.placeholder(dtype=tf.float32, shape=(None, dim_latent))
        self.k          = tf.placeholder(dtype=tf.float32)
        self.multiplier = tf.placeholder(dtype=tf.float32)

    def encoder(self):
        # variable scope for discriminator
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
            # first conv layer : [-1, 64, 64, n]
            hidden = tf.layers.conv2d(inputs=self.input, filters=self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            # second layer -> output : [-1, 32, 32, 2 * n]
            hidden = tf.layers.conv2d(inputs=hidden, filters=self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            hidden = tf.layers.conv2d(inputs=hidden, filters=2 * self.num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=activation_fn, kernel_initializer=initializer)
            # third layer -> output : [-1, 16, 16, 3 * n]
            hidden = tf.layers.conv2d(inputs=hidden, filters=2 * self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            hidden = tf.layers.conv2d(inputs=hidden, filters=3 * self.num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=activation_fn, kernel_initializer=initializer)
            # fourth layer -> output : [-1, 16, 16, 3 * n]
            hidden = tf.layers.conv2d(inputs=hidden, filters=3 * self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            hidden = tf.layers.conv2d(inputs=hidden, filters=3 * self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            # fully connected layer
            embedding = tf.reshape(tensor=hidden, shape=(-1, 16 * 16 * 3 * self.num_filters))
            embedding = tf.layers.dense(inputs=embedding, units=self.num_units, activation=None, kernel_initializer=initializer)
        return embedding

    def generator_or_decoder(self, embedding, scope_name="decoder"):
        # choose what to make
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            # embedding to feature
            h_0 = tf.layers.dense(inputs=embedding, units=16 * 16 * self.num_filters, activation=None, kernel_initializer=initializer)
            h_0 = tf.reshape(tensor=h_0, shape=(-1, 16, 16, self.num_filters))
            # first conv layer -> output : [-1, 32, 32, 2 * n]
            hidden = tf.layers.conv2d(inputs=h_0, filters=self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            hidden = tf.layers.conv2d(inputs=hidden, filters=self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            upsampled_hidden = tf.image.resize_nearest_neighbor(hidden, size=(32, 32))
            upsampled_skip_connection = tf.image.resize_nearest_neighbor(h_0, size=(32, 32))
            hidden = tf.concat([upsampled_hidden, upsampled_skip_connection], axis=3)
            # second conv layer -> output : [-1, 64, 64, 2 * n]
            hidden = tf.layers.conv2d(inputs=hidden, filters=self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            hidden = tf.layers.conv2d(inputs=hidden, filters=self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            upsampled_hidden = tf.image.resize_nearest_neighbor(hidden, size=(64, 64))
            upsampled_skip_connection = tf.image.resize_nearest_neighbor(h_0, size=(64, 64))
            hidden = tf.concat([upsampled_hidden, upsampled_skip_connection], axis=3)
            # third conv layer -> output : [-1, 64, 64, n]
            hidden = tf.layers.conv2d(inputs=hidden, filters=self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            hidden = tf.layers.conv2d(inputs=hidden, filters=self.num_filters, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
            # final conv layer -> output : [-1, 64, 64, 4]
            image = tf.layers.conv2d(inputs=hidden, filters=4, kernel_size=(3, 3), padding='same', activation=activation_fn, kernel_initializer=initializer)
        return image

    def _loss(self, image):
        # encode the image
        embedding = self.encoder()
        reconstructed = self.generator_or_decoder(embedding, "decoder")
        # return the loss
        return tf.reduce_mean(tf.abs(image - reconstructed))

    def loss(self):
        # the real image
        d_real_loss = self._loss(image=self.input)
        # the fake image
        gen_fake = self.generator_or_decoder(self.z, scope_name="generater")
        d_fake_loss = self._loss(gen_fake)
        # for the discriminator
        discriminator_loss  = d_real_loss - self.k * d_fake_loss
        generater_loss      = d_fake_loss
        m_global = d_real_loss + tf.abs(gamma * d_real_loss - d_fake_loss)

        return discriminator_loss, generater_loss, m_global

    def train(self):
        # get both losses
        discriminator_loss, generater_loss, m_global = self.loss()
        # get the variables
        encoder_vars = tf.trainable_variables(scope="encoder")
        decoder_vars = tf.trainable_variables(scope="decoder")
        auto_encoder_vars = encoder_vars + decoder_vars
        generater_vars = tf.trainable_variables(scope="generater")
        # use Adam optimizer
        discriminator_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(discriminator_loss, var_list=auto_encoder_vars)
        generater_train     = tf.train.AdamOptimizer(learning_rate=lr).minimize(generater_loss, var_list=generater_vars)

        return discriminator_train, generater_train, discriminator_loss, generater_loss, m_global


