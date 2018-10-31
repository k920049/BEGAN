import tensorflow as tf
import numpy as np
import os

from model.BEGAN import BEGAN, gamma
from src.data_loader import DataLoader

def main(train=True):

    eps = 1e-3
    save_path = "./model.ckpt"
    validation_per_batch = 10
    save_interval = 16
    batch_size = 64
    dim_z = 1024
    k = 0.0
    multiplier = 1e-3

    # load the data
    data = DataLoader(path="./data/img_align_celeba_png")

    with tf.name_scope("model"):
        model = BEGAN(num_filters=128, num_units=1024, dim_latent=dim_z)

    with tf.name_scope("train"):
        discriminator_train, generater_train, discriminator_loss, generater_loss, m_global = model.train()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    # if we're now training the model
    if train:
        iteration = 0

        # get default session
        with tf.Session() as sess:
            # initialize all the variables
            sess.run(init)
            iteration = iteration + 1
            # restore the checkpoint, if exists
            file_names= os.listdir("./")
            files = [f for f in file_names if f.endswith(".ckpt")]
            if len(files) != 0:
                saver.restore(sess, save_path=save_path)
            # get batch
            batch_image = data.batch(batch_size=batch_size)
            batch_z = 2.0 * np.random.rand(batch_size, dim_z) - 1.0
            feeding = {model.input : batch_image, model.z : batch_z, model.k : k, model.multiplier : multiplier}
            # validate every 10 iteration
            if iteration % validation_per_batch == 0:
                loss = sess.run(m_global, feed_dict=feeding)
                print("At iteration {}, Validation loss: {}".format(iteration, loss))
            else:
                # run one iteration
                _, d_loss, loss = sess.run([discriminator_train, discriminator_loss, m_global], feed_dict=feeding)
                _, g_loss       = sess.run([generater_train, generater_loss], feed_dict=feeding)
                # update k
                k = k + multiplier * (gamma * d_loss - g_loss)

            if iteration % save_interval == 0:
                saver.save(sess=sess, save_path=save_path)


if __name__ == '__main__':
    main(train=True)