import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, time
import cv2

from model.BEGAN import BEGAN
from src.data_loader import DataLoader

def main(train=True):

    niter = 50
    niter_snapshot=2048

    batch_size = 16
    data_size = 64
    filters = 64
    dim_z = 128
    kt = 0.0
    gamma = 0.4
    multiplier = 1e-3
    lr = 1e-4

    ckpt_dir = "./"
    max_to_keep = 5

    # load the data
    data = DataLoader(path="./data/img_align_celeba_png")
    batch_total = int(data.size() / batch_size)

    with tf.name_scope("model"):
        model = BEGAN(batch_size, data_size, filters, dim_z, dim_z, gamma)

    with tf.name_scope("train"):
        opt_g, g_loss, d_real_loss, d_fake_loss = model.generator_ops()
        opt_d, d_loss = model.discriminator_ops()
        g_opt = [opt_g, g_loss, d_real_loss, d_fake_loss]
        d_opt = [opt_d, d_loss]

    with tf.name_scope("miscellaneous"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=(max_to_keep))

    # if we're now training the model
    if train:
        with tf.Session() as sess:
            count = 0
            k_list = []
            loss_list = []
            # initializer
            sess.run(init)
            try:
                ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(ckpt_dir, ckpt_name))
            except:
                saver.save(sess, "./model.ckpt", write_meta_graph=True)

            for epoch in range(niter):

                for idx in range(0, batch_total):
                    count += 1

                    batch_x = np.random.uniform(-1., 1., size=[batch_size, dim_z])
                    batch_data = data.batch(batch_size=batch_size)

                    # opt & feed list (different with paper)

                    feed_dict = {model.x: batch_x, model.y: batch_data, model.kt: kt, model.lr: lr}

                    # run tensorflow
                    _, loss_g, d_real_loss, d_fake_loss = sess.run(g_opt, feed_dict=feed_dict)
                    _, loss_d = sess.run(d_opt, feed_dict=feed_dict)

                    # update kt, m_global
                    kt = kt + multiplier * (gamma * d_real_loss - d_fake_loss)
                    kt = np.clip(kt, 0.0, 1.0)
                    m_global = d_real_loss + np.abs(gamma * d_real_loss - d_fake_loss)
                    loss = loss_g + loss_d
                    k_list.append(kt)
                    loss_list.append(m_global)

                    print("Epoch: [%2d] [%4d/%4d], loss: %.4f, loss_g: %.4f, loss_d: %.4f, d_real: %.4f, d_fake: %.4f, kt: %.8f, M: %.8f" % (epoch, idx, batch_total, loss, loss_g, loss_d, d_real_loss, d_fake_loss, kt, m_global))

                    # Test during Training
                    if count % niter_snapshot == (niter_snapshot - 1):
                        # update learning rate
                        lr *= 0.95
                        # save & test
                        saver.save(sess, "./model.ckpt", global_step=count, write_meta_graph=False)
                        test_data = np.random.uniform(-1., 1., size=[batch_size, dim_z])
                        output_gen = sess.run(model.recon_gen, feed_dict={model.x: test_data})
                        output_dec = sess.run(model.recon_dec, feed_dict={model.x: test_data})

                        fig = plt.figure()
                        ax1 = fig.add_subplot(2, 2, 1)
                        ax2 = fig.add_subplot(2, 2, 2)
                        ax3 = fig.add_subplot(2, 2, 3)
                        ax4 = fig.add_subplot(2, 2, 4)

                        ax1.plot(np.array(k_list), 'k--', label="k")
                        ax2.plot(np.array(loss_list), 'k-', label="m_global")
                        ax3.imshow(output_dec[0])
                        ax4.imshow(batch_data[0])

                        fig.show()

if __name__ == '__main__':
    main(train=True)