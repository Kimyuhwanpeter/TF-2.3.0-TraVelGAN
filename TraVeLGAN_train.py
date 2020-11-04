# -*- coding:utf-8 -*-
from TraVeLGAN_model import *
from absl import flags
from random import shuffle, random
from itertools import combinations

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os

flags.DEFINE_integer("load_size", 286, "Load input size")

flags.DEFINE_integer("input_size", 256, "Model input size")

flags.DEFINE_integer("batch_size", 4, "Batch size")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_float("lr", 0.0002, "Leanring rate")

flags.DEFINE_integer("num_classes", 24, "Number of age classes")

flags.DEFINE_string("A_txt_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/male_16_39_train.txt", "A text path")

flags.DEFINE_string("A_img_path", "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/", "A image path")

flags.DEFINE_string("B_txt_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/female_40_63_train.txt", "B text path")

flags.DEFINE_string("B_img_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/female_40_63/", "B image path")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Restored the checkpoint files")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_string("save_checkpoint", "", "Saving checkpoint files")

flags.DEFINE_string("graphs", "", "Saving train graphs")

flags.DEFINE_string("save_images", "C:/Users/Yuhwan/Pictures/sample", "Saving images")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.9)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.9)
s_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.9)

def tr_func(A, B):

    A_img = tf.io.read_file(A)
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.input_size, FLAGS.input_size, 3])

    B_img = tf.io.read_file(B)
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.load_size, FLAGS.load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.input_size, FLAGS.input_size, 3])


    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img) / 127.5 - 1.
        B_img = tf.image.flip_left_right(B_img) / 127.5 - 1.
    else:
        A_img = A_img / 127.5 - 1.
        B_img = B_img / 127.5 - 1.

    return A_img, B_img, A, B

def abs_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))

def mae_criterion(input, target):
    return tf.reduce_mean((input - target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))

def siamese_lossfn(logits, labels=None, diff=False, diffmargin=10., samemargin=0.):
    if diff:
        return tf.maximum(0., diffmargin - tf.reduce_sum(logits, axis=-1))

    return tf.reduce_sum(logits, axis=-1)

def siamese_loss(SA_real, SB_real, SA_fake, SB_fake):

    loss_S = tf.constant(0.)
    orders = [np.array(list(range(i, FLAGS.batch_size)) + list(range(i))) for i in range(1, FLAGS.batch_size)]
    losses_S1 = []
    losses_S2 = []
    #losses_S3 = []
    for i, order in enumerate(orders):
        other = tf.constant(order)
        dists_withinx1 = SA_real - tf.gather(SA_real, other)
        dists_withinx2 = SB_real - tf.gather(SB_real, other)
        dists_withinG1 = SA_fake - tf.gather(SA_fake, other)
        dists_withinG2 = SB_fake - tf.gather(SB_fake, other)

        losses_S1.append(tf.reduce_mean(siamese_lossfn((dists_withinx1)**2, diff=True)))
        losses_S1.append(tf.reduce_mean(siamese_lossfn((dists_withinx2)**2, diff=True)))

        losses_S2.append(tf.reduce_mean((dists_withinx1 - dists_withinG1)**2))
        losses_S2.append(tf.reduce_mean((dists_withinx2 - dists_withinG2)**2))

        #losses_S3.append(tf.reduce_mean(tf.reduce_sum(-(tf.nn.l2_normalize(dists_withinx1, axis=[-1]) * tf.nn.l2_normalize(dists_withinG1, axis=[-1])), axis=-1)))
        #losses_S3.append(tf.reduce_mean(tf.reduce_sum(-(tf.nn.l2_normalize(dists_withinx2, axis=[-1]) * tf.nn.l2_normalize(dists_withinG2, axis=[-1])), axis=-1)))

    loss_S1 = tf.reduce_mean(losses_S1)
    loss_travel = tf.reduce_mean(losses_S2)
    #marginloss = tf.reduce_mean(losses_S3)
    #loss_S1 + loss_travel

    return loss_S1, loss_travel

@tf.function
def cal_loss(A, B, A2B_gener, B2A_gener, A_dis, B_dis, Si):

    with tf.GradientTape(persistent=True) as tape:
        fake_B = A2B_gener(A, True)
        fake_A = B2A_gener(B, True)

        DA_real = A_dis(A, True)
        DA_fake = A_dis(fake_A, True)

        DB_real = B_dis(B, True)
        DB_fake = B_dis(fake_B, True)

        SA_real = Si(A, True)
        SA_fake = Si(fake_B, True)

        SB_real = Si(B, True)
        SB_fake = Si(fake_A, True)

        loss_S, loss_travel = siamese_loss(SA_real, SB_real, SA_fake, SB_fake)
        g_adv =  1 * (mae_criterion(DB_fake, tf.ones_like(DB_fake)) + mae_criterion(DA_fake, tf.ones_like(DA_fake)))
        #g = g_loss
        g_loss = 10*(loss_travel + loss_S) + g_adv

        disc_A_loss = (mae_criterion(DA_real, tf.ones_like(DA_real)) + mae_criterion(DA_fake, tf.zeros_like(DA_fake))) / 2
        disc_B_loss = (mae_criterion(DB_real, tf.ones_like(DB_real)) + mae_criterion(DB_fake, tf.zeros_like(DB_fake))) / 2

        d_loss = disc_A_loss + disc_B_loss

    g_grads = tape.gradient(g_loss, A2B_gener.trainable_variables + B2A_gener.trainable_variables)
    s_grads = tape.gradient(g_loss, Si.trainable_variables)
    d_grads = tape.gradient(d_loss, A_dis.trainable_variables + B_dis.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, A2B_gener.trainable_variables + B2A_gener.trainable_variables))
    g_optim.apply_gradients(zip(s_grads, Si.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, A_dis.trainable_variables + B_dis.trainable_variables))
    
    return g_loss, d_loss

def main():
    A2B_generator = ResnetGenerator(input_shape=(FLAGS.input_size, FLAGS.input_size, 3))
    B2A_generator = ResnetGenerator(input_shape=(FLAGS.input_size, FLAGS.input_size, 3))
    A_dis = ConvDiscriminator(input_shape=(FLAGS.input_size, FLAGS.input_size, 3))
    B_dis = ConvDiscriminator(input_shape=(FLAGS.input_size, FLAGS.input_size, 3))
    Siemese = siamese(input_shape=(FLAGS.input_size, FLAGS.input_size, 3), num_classes=FLAGS.num_classes)

    A2B_generator.summary()
    A_dis.summary()
    Siemese.summary()
    
    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_generator=A2B_generator,
                                   B2A_generator=B2A_generator,
                                   A_dis=A_dis,
                                   B_dis=B_dis,
                                   Siemese=Siemese)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("* Restored the latest checkpoint files!!")

    if FLAGS.train:
        count = 0

        A_tr_img = np.loadtxt(FLAGS.A_txt_path, dtype="<U200", skiprows=0, usecols=0)
        A_tr_img = [FLAGS.A_img_path + img for img in A_tr_img]
        B_tr_img = np.loadtxt(FLAGS.B_txt_path, dtype="<U200", skiprows=0, usecols=0)
        B_tr_img = [FLAGS.B_img_path + img for img in B_tr_img]

        for epoch in range(FLAGS.epochs):
            np.random.shuffle(A_tr_img)
            np.random.shuffle(B_tr_img)

            tr_gener = tf.data.Dataset.from_tensor_slices((A_tr_img, B_tr_img))
            tr_gener = tr_gener.shuffle(len(A_tr_img))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = len(A_tr_img) // FLAGS.batch_size
            tr_iter = iter(tr_gener)
            for step in range(tr_idx):
                A_imgs, B_imgs, A_file_name, B_file_name = next(tr_iter)

                g_loss, d_loss = cal_loss(A_imgs,
                                          B_imgs,
                                          A2B_generator,
                                          B2A_generator,
                                          A_dis,
                                          B_dis,
                                          Siemese)
                print("Epoch: {} [{}/{}] g loss = {}, d loss = {}".format(epoch, step + 1, tr_idx, g_loss, d_loss))
                print(g_loss, d_loss)

                if count % 1000 == 0:
                    fake_B = A2B_generator(A_imgs, training=False)
                    fake_A = B2A_generator(B_imgs, training=False)

                    A_name = str(tf.compat.as_str_any(A_file_name[0].numpy())).split('/')[-1].split('.')[0]
                    B_name = str(tf.compat.as_str_any(B_file_name[0].numpy())).split('/')[-1].split('.')[0]
                    
                    plt.imsave(FLAGS.save_images + "/"+ A_name + "_fake_B_{}.jpg".format(count), fake_B[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.save_images + "/"+ B_name +"_fake_A_{}.jpg".format(count), fake_A[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.save_images + "/"+ A_name + "_real_A_{}.jpg".format(count), A_imgs[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.save_images + "/"+ B_name + "_real_B_{}.jpg".format(count), B_imgs[0].numpy() * 0.5 + 0.5)

                    ckpt = tf.train.Checkpoint(A2B_generator=A2B_generator,
                                               B2A_generator=B2A_generator,
                                               A_dis=A_dis,
                                               B_dis=B_dis,
                                               Siemese=Siemese)
                    num_ = int(count // 1000)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                    if not os.path.isdir(model_dir):
                        print("Make {} folder to save checkpoint files..".format(num_))
                        os.makedirs(model_dir)
                    ckpt_dir = model_dir + "/" + "TravelGAN_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__":
    main()