"""
This folder is for the RSNA bone age competition
To keep it separate.

Train
"""

import os
import time

import Model as network
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Define flags
FLAGS = tf.app.flags.FLAGS

# Define some of the data variables
tf.app.flags.DEFINE_string('data_dir', 'data/', """Path to the data directory.""")
tf.app.flags.DEFINE_string('training_dir', 'training/', """Path to the training directory.""")
tf.app.flags.DEFINE_string('test_files', '1', """Testing files""")
tf.app.flags.DEFINE_integer('box_dims', 256, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 128, """Center: 80/208 for 128/256""")
tf.app.flags.DEFINE_integer('slice_gap', 2, """Slice spacing for pre processing in mm""")
tf.app.flags.DEFINE_integer('num_classes', 2, """Number of classes""")

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('num_epochs', 200, """Number of epochs to run""")
tf.app.flags.DEFINE_integer('epoch_size', 35000, """How many examples""")
tf.app.flags.DEFINE_integer('print_interval', 1, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 25, """How many Epochs to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.75, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-5, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")
tf.app.flags.DEFINE_float('loss_factor',1.1, """The loss weighting factor""")
tf.app.flags.DEFINE_float('dice_threshold', 1.0, """ The threshold value to declare GBM""")

# Hyperparameters to control the optimizer
tf.app.flags.DEFINE_float('learning_rate',1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'New1_OldDICE/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")

def train():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/gpu:' + str(FLAGS.GPU)):

        # Load the images and labels.
        data, _ = network.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Perform the forward pass:
        logits, l2loss = network.forward_pass_res(data['image_data'], phase_train=phase_train)

        # Calculate loss
        SCE_loss = network.total_loss(logits, data['label_data'], loss_type='DICE')

        # Add the L2 regularization loss
        loss = tf.add(SCE_loss, l2loss, name='TotalLoss')

        # Update the moving average batch norm ops
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Retreive the training operation with the applied gradients
        with tf.control_dependencies(extra_update_ops): train_op = network.backward_pass(loss)

        # -------------------  Housekeeping functions  ----------------------

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=4)

        # -------------------  Session Initializer  ----------------------

        # Set the intervals
        max_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.num_epochs)
        print_interval = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.print_interval)
        checkpoint_interval = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.checkpoint_interval)
        print('Max Steps: %s, Print Interval: %s, Checkpoint: %s' % (max_steps, print_interval, checkpoint_interval))

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as mon_sess:

            # Initialize the variables
            mon_sess.run(var_init)

            # Initialize the handle to the summary writer in our training directory
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir + FLAGS.RunInfo, mon_sess.graph)

            # Initialize the step counter
            timer = 0

            # Use slim to handle queues:
            with slim.queues.QueueRunners(mon_sess):
                for i in range(max_steps):

                    # Run and time an iteration
                    start = time.time()
                    mon_sess.run(train_op, feed_dict={phase_train: True})
                    timer += (time.time() - start)

                    # Calculate current epoch
                    Epoch = int((i * FLAGS.batch_size) / FLAGS.epoch_size)

                    # Console and Tensorboard print interval
                    if i % print_interval == 0:

                        # First retreive the loss values
                        l2, sce, tot = mon_sess.run([l2loss, SCE_loss, loss], feed_dict={phase_train: True})
                        tot *= 1e6
                        l2 *= 1e6
                        sce *= 1e6

                        # Get timing stats
                        elapsed = timer/print_interval
                        timer = 0

                        # Calc epoch
                        Epoch = int((i * FLAGS.batch_size) / FLAGS.epoch_size)

                        # Now print the loss values
                        print ('-'*70)
                        print('Epoch: %s, Time: %.1f sec, L2 Loss (ppm): %.4f, Prediction Loss (ppm): %.4f, Total Loss (ppm): %.4f, Eg/s: %.4f, Seconds Per: %.4f'
                              % (Epoch, elapsed, l2, sce, tot, FLAGS.batch_size/elapsed, elapsed/FLAGS.batch_size))

                        # Run a session to retrieve our summaries
                        summary = mon_sess.run(all_summaries, feed_dict={phase_train: True})

                        # Add the summaries to the protobuf for Tensorboard
                        summary_writer.add_summary(summary, i)

                        # Timer
                        start_time = time.time()

                    if i % checkpoint_interval == 0:

                        print('-' * 70, '\nSaving... GPU: %s, File:%s' % (FLAGS.GPU, FLAGS.RunInfo[:-1]))

                        # Define the filename
                        file = ('Epoch_%s' % Epoch)

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join(FLAGS.train_dir + FLAGS.RunInfo, file)

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(0)
    if tf.gfile.Exists(FLAGS.train_dir + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively(FLAGS.train_dir + FLAGS.RunInfo)
    tf.gfile.MakeDirs(FLAGS.train_dir + FLAGS.RunInfo)
    train()

if __name__ == '__main__':
    tf.app.run()