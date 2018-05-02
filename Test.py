""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import os
import time

import GBMSeg as network
import SODTester as SDT
import SODLoader as SDL
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Define an instance of the loader and testing file
sdl = SDL.SODLoader(os.getcwd())
sdt = SDT.SODTester(False, False)

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('testing_dir', 'testing/', """Path to the testing directory.""")
tf.app.flags.DEFINE_string('test_files', '1', """Testing files""")
tf.app.flags.DEFINE_integer('box_dims', 256, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 128, """dimensions for the network input""")
tf.app.flags.DEFINE_integer('num_classes', 2, """Number of classes""")

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('epoch_size', 10000, """How many examples""")
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 1.0, """ Keep probability""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")
tf.app.flags.DEFINE_float('dice_threshold', 0.5, """ The threshold value to declare positive""")
tf.app.flags.DEFINE_float('size_threshold', 1.0, """ The size threshold value to declare detected PE""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-5, """ The gamma value for regularization loss""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'First_Run/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")

# Define a custom training class
def eval():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('cpu:0'):

        # Load the images and labels.
        _, validation = network.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Perform the forward pass:
        logits, l2loss = network.forward_pass_res(validation['image_data'], phase_train=phase_train)

        # To retreive labels
        labels = validation['label_data']

        # Retreive the softmax for testing purposes
        softmax = tf.nn.softmax(logits)

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        while True:

            # config Proto sets options for configuring the session like run on GPU, allocate GPU memory etc.
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as mon_sess:

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)

                # Initialize the variables
                mon_sess.run(var_init)

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the learned variables
                    restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                    # Restore the graph
                    restorer.restore(mon_sess, ckpt.model_checkpoint_path)
                    print("Sucessfully restored: ", ckpt.model_checkpoint_path)

                    # Extract the epoch
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

                else:
                    print ('No model checkopoint found... quitting')
                    break

                # Initialize the step counter
                tot, TP, TN, FP, FN, DICE, total, step = 0, 0, 0, 0, 0, 0, 1e-8, 0
                sdt = SDT.SODTester(False, False)
                display_lab, display_log, display_img = [], [], []
                avg_softmax, ground_truth = [], []

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Use slim to handle queues:
                with slim.queues.QueueRunners(mon_sess):

                    for i in range(max_steps):

                        # Retreive the predictions and labels
                        preds, labs, egs = mon_sess.run([softmax, labels, validation], feed_dict={phase_train: False})

                        # Get metrics
                        Dixe = sdt.calc_metrics_segmentation(preds, labs, egs['image_data'], dice_threshold=FLAGS.dice_threshold, batch_size=FLAGS.batch_size)

                        if Dixe:
                            DICE += Dixe
                            total += 1

                        # Convert inputs to numpy arrays
                        p11 = np.squeeze(preds.astype(np.float))
                        l11 = np.squeeze(labs.astype(np.float))
                        eg = np.squeeze(egs['image_data'].astype(np.float))
                        picd, display = [], []

                        for i in range(FLAGS.batch_size):

                            # Retreive one image, label and prediction from the batch to save
                            p1 = np.copy(p11[i, :, :, 1])
                            p2 = np.copy(l11[i])  # make an independent copy of labels map

                            # display.append(p1)
                            # display.append(p2)
                            # display.append(eg[i, 2, :, :])

                            # Now create boolean masks
                            p1[p1 > FLAGS.dice_threshold] = True  # Set predictions above threshold value to True
                            p1[p1 <= FLAGS.dice_threshold] = False  # Set those below to False
                            p2[p2 == 1] = False  # Mark lung and background as False
                            p2[p2 > 0] = True  # Mark embolisms as True

                            # display.append(p1)
                            # display.append(p2)

                            # Check error
                            if np.sum(p1) > FLAGS.size_threshold and np.sum(p2) > 0: TP += 1
                            elif np.sum(p2) > 0 and np.sum(p1) < FLAGS.size_threshold: FN += 1
                            elif np.sum(p2) == 0 and np.sum(p1) < FLAGS.size_threshold: TN += 1
                            elif np.sum(p2) == 0 and np.sum(p1) > FLAGS.size_threshold: FP += 1
                            tot += 1

                            # Generate an overlay display
                            if np.sum(p2) > 5: picd.append(sdl.display_overlay(eg[i, 2, :, :], p1))

                        # sdl.display_volume(np.asarray(display), True)

                        # Garbage collection
                        del preds, labs, egs, eg, picd, p1, p2

                        # Increment step
                        step += 1

                        if step % 10 == 0: print ('Step %s of %s done' %(step, max_steps))

                    # Print final errors here
                    DICE_Score = DICE/total
                    print ('DICE Score: %s', (DICE_Score))
                    print('TP: %s, TN: %s, FP: %s, FN: %s, Slices: %s' % (TP, TN, FP, FN, tot))
                    #print ('Sensitivity: %.2f %%, Specificity: %.2f %%' %((100*TP / (TP + FN)), (100* TN / (TN + FP))))
                    try:sdl.display_volume(np.asarray(picd), True)
                    except:pass

            # Print divider
            print('-' * 70)
            break


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(0)
    if tf.gfile.Exists('testing/' + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo)
    tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo)
    eval()


if __name__ == '__main__':
    tf.app.run()