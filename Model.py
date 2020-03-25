# Defines and builds our network
#    Computes input images and labels using inputs() or distorted inputs ()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

_author_ = 'simi'

import tensorflow as tf
import Input
import SODNetwork as SDN
import SOD_DenseNet as SDDN

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Retreive helper function object
sdn = SDN.SODMatrix()
sdloss = SDN.SODLoss(2)

def forward_pass_dense(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    :param images: Images to analyze
    :param phase_train: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    # DenseNet class
    sddn = SDDN.DenseUnet(nb_blocks=5, filters=6, images=images, phase_train=phase_train)

    # Now run the network
    conv = sddn.define_network_25D(layers=[2, 4, 8, 16, 32], keep_prob=FLAGS.dropout_factor)

    # Output is a 1x1 box with 3 labels
    Logits = sdn.convolution('Logits', conv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)
    print('Logits: ', Logits)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def forward_pass_res(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    # First block: 3D conv -> Downsample (stried) -> Z Downsample
    print ('Input images: ', images)
    conv = sdn.convolution_3d('Conv1a', images, 3, 8, 1, phase_train=phase_train)
    conv = sdn.convolution_3d('Conv1b', conv, 3, 8, 1, phase_train=phase_train)
    skip1 = sdn.convolution_3d('Skip1', conv, [5, 1, 1], 8, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv1c', conv, [2, 3, 3], 16, [1, 2, 2], 'VALID', phase_train) # 4x63x63x16
    print('*' * 30, conv)

    conv = sdn.residual_layer_3d('Conv2a', conv, 3, 16, 1, phase_train=phase_train)
    conv = sdn.residual_layer_3d('Conv2b', conv, 3, 16, 1, phase_train=phase_train)
    skip2 = sdn.convolution_3d('Skip2', conv, [4, 1, 1], 16, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv2c', conv, [2, 3, 3], 32, [1, 2, 2], 'VALID', phase_train)  # 3x31x31x32
    print('*' * 22, conv)

    conv = sdn.residual_layer_3d('Conv3a', conv, 3, 32, 1, phase_train=phase_train)
    conv = sdn.residual_layer_3d('Conv3b', conv, 3, 32, 1, phase_train=phase_train,)
    skip3 = sdn.convolution_3d('Skip3', conv, [3, 1, 1], 32, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv3c', conv, [2, 3, 3], 64, [1, 2, 2], 'VALID', phase_train)  # 2x15x15x15
    print('*'*14,conv)

    conv = sdn.inception_layer_3d('Conv4a', conv, 64, 1, phase_train=phase_train)
    conv = sdn.inception_layer_3d('Conv4b', conv, 64, 1, phase_train=phase_train)
    skip4 = sdn.convolution_3d('Skip4', conv, [2, 1, 1], 64, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv4c', conv, [2, 3, 3], 128, [1, 2, 2], 'VALID', phase_train)  # 1x7x7x128
    print('*'*6,conv)

    # From now on, we're 2D
    conv = tf.squeeze(conv)

    # Bottom of the decoder: 7x7
    conv = sdn.inception_layer('conv5_Inception', conv, 128, 1, 'SAME', phase_train, BN=False, relu=False)
    conv = sdn.residual_layer('Conv5', conv, 3, 128, 1, padding='SAME', phase_train=phase_train)
    conv = sdn.inception_layer('Conv5_Inception2', conv, 128, phase_train=phase_train)
    print('End Encoder: ', conv)

    # Upsample 1
    conv = sdn.deconvolution('Dconv1', conv, 3, 64, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip4), out_shape=[FLAGS.batch_size, 15, 15, 64])
    conv = sdn.inception_layer('Dconv1b', conv, 64, phase_train=phase_train)
    print('-'*6, conv)

    # Upsample 2
    conv = sdn.deconvolution('Dconv2', conv, 3, 32, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip3), out_shape=[FLAGS.batch_size, 31, 31, 32])
    conv = sdn.inception_layer('Dconv2b', conv, 32, phase_train=phase_train)
    print ('-'*14, conv)

    # Upsample 3
    conv = sdn.deconvolution('Dconv3', conv, 3, 16, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip2), out_shape=[FLAGS.batch_size, 63, 63, 16])
    conv = sdn.residual_layer('Dconv3b', conv, 3, 16, 1, phase_train=phase_train)
    print ('-'*22,conv)

    # Upsample 4
    conv = sdn.deconvolution('Dconv4', conv, 3, 8, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip1), out_shape=[FLAGS.batch_size, 128, 128, 8])
    conv = sdn.residual_layer('Dconv4b', conv, 3, 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Dconv4c', conv, 3, 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Dconv4d', conv, 3, 8, 1, phase_train=phase_train, dropout=FLAGS.dropout_factor)
    print ('-'*30, conv)

    # Output
    Logits = sdn.convolution('Logits', conv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)
    print('Logits: ', Logits)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def total_loss(logits_tmp, labels_tmp, num_classes=2, class_weights=None, loss_type=None):

    """
    Cost function
    :param logitz: The raw log odds units output from the network
    :param labelz: The labels: not one hot encoded
    :param num_classes: number of classes predicted
    :param class_weights: class weight array
    :param loss_type: DICE or other to use dice or weighted
    :return:
    """

    # Reduce dimensionality
    labelz, logits = tf.squeeze(labels_tmp), tf.squeeze(logits_tmp)

    # Remove background label
    labels = tf.cast(labelz, tf.uint8)

    # Summary images
    imeg = int(FLAGS.batch_size/2)
    tf.summary.image('Labels', tf.reshape(tf.cast(labels[imeg], tf.float32), shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)
    tf.summary.image('Logits', tf.reshape(logits_tmp[imeg,:,:,1], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)

    if loss_type=='DICE':

        # # Get the generalized DICE loss
        # loss = sdloss.generalized_dice_loss(logits_tmp, labels_tmp, weight_map=class_weights, type_weight='Square')

        # Flatten
        logits = tf.reshape(logits_tmp, [-1, num_classes])
        labels = tf.reshape(labels_tmp, [-1, 1])

        # To prevent number errors:
        eps = 1e-5

        # Find the intersection. Multiplication gets broadcast
        intersection = 2 * tf.reduce_sum(logits * labels) + eps

        # find the union.
        union = eps + tf.reduce_sum(logits) + tf.reduce_sum(labels)

        """
        Some explanation of this hacked DICE score
        The intersection over union won't add up to a real dice score 
        but the drive of the network will still be to maximise the dice coefficient
        It does this by trying to make the numbers in the intersection as big as possible 
        while keeping the numbers in the union but not the intersection as small as possible
        """

        # Calculate the loss
        dice = intersection / union

        # Output the training DICE score
        tf.summary.scalar('DICE_Score', dice)

        # 1-DICE since we want better scores to have lower loss
        loss = 1 - dice


    elif loss_type=='DICE_X':

        # Get the sum of the cross entropy and DICE loss
        loss = sdloss.dice_plus_xent_loss(logits_tmp, labels_tmp, weight_map=class_weights)

    elif loss_type=='SN_SP':

        # Get the sum of r(specificity) and (1-r)(sensitivity) loss
        loss = sdloss.sensitivity_specificity_loss(logits_tmp, labels_tmp, weight_map=class_weights, r=0.05)

    elif loss_type=='WAS':

        # Get the the pixel-wise Wasserstein distance between the prediction and the labels (ground_truth) with respect
        #     to the distance matrix on the label space M.
        loss = sdloss.wasserstein_disagreement_map(logits_tmp, labels_tmp, weight_map=class_weights)

    else:

        # Classic cross entropy

        # Make labels one hot
        labels = tf.cast(tf.one_hot(labels, depth=FLAGS.num_classes, dtype=tf.uint8), tf.float32)

        # Generate class weights
        class_weights = tf.Variable([1, FLAGS.loss_factor], trainable=False)

        # Flatten
        logits = tf.reshape(logits, [-1, num_classes])
        labels = tf.cast(tf.reshape(labels, [-1, num_classes]), tf.float32)

        # Make our weight map
        weight_map = tf.multiply(labels, class_weights)
        weight_map = tf.reduce_sum(weight_map, axis=1)

        # Calculate the loss: Result is batch x 65k
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # Apply the class weights
        loss = tf.multiply(loss, weight_map)

        # Reduce the loss into a scalar
        loss = tf.reduce_mean(loss)

    # Output the Loss
    tf.summary.scalar('Loss_Raw', loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

    return loss


def backward_pass(total_loss):

    """
    Perform the backward pass and update the gradients
    :param total_loss:
    :return:
    """

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.train.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Compute the gradients. NAdam optimizer came in tensorflow 1.2
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables(): tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # Control graph execution
    with tf.control_dependencies([train_op, variable_averages_op]):  dummy_op = tf.no_op(name='train')

    return dummy_op


def inputs(skip=False):

    """
    Load the raw inputs
    :param skip:
    :return:
    """

    # Skip part 1 and 2 if the protobuff already exists
    if not skip:
        Input.pre_proc_25D(FLAGS.slice_gap, FLAGS.box_dims)
        Input.pre_proc_25D_BRATS(FLAGS.slice_gap, FLAGS.box_dims)

    else: print('-------------------------Previously saved records found! Loading...')

    # Part 2: Load the protobuff  -----------------------------
    print('----------------------------------------Loading Protobuff...')
    train = Input.load_protobuf()
    valid = Input.load_validation()


    return train, valid