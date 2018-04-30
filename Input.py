"""
Load and preprocess the files to a protobuff
"""

import os

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path

from random import shuffle

# Define flags
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
brats_dir = str(Path.home()) + '/PycharmProjects/Datasets/BRATS2015/'
cumc_dir = str(Path.home()) + '/PycharmProjects/Datasets/GBM/TxResponse/'

sdl = SDL.SODLoader(data_root=brats_dir)

# For loading the files for a 2.5 D network
def pre_proc_25D(slice_gap, dims):

    """
    Loads boxes from the Iran data set
    :param slice_gap: the gap between slices to save for 3D data
    :param dims: the dimensions of the images saved
    :return:
    """

    # First check all the labels and files
    check_labels()

    # Load the files
    filenames = sdl.retreive_filelist('gz', include_subfolders=True, path=cumc_dir)
    shuffle(filenames)
    print('Filenames: ', filenames)

    # Load the labels: {'1': {'LABEL': '1', 'TEXT': 'recurrent, with treatment effect', 'ACCNO': '4178333', 'MRN': '7647499'}, '3': ...
    label_file = sdl.retreive_filelist('csv', False, 'data/')[0]
    labels = sdl.load_CSV_Dict('ID', label_file)
    print('Labels: ', labels)

    # Counters: index = example, pts = patients
    index, pts, per = 0, 0, 0

    # Data array
    data = {}

    # Testing
    class_count, display = [0, 0], []

    # Now do the magic
    for lbl_file in filenames:

        # Only work on label files
        if 'label' not in lbl_file: continue

        # Load the patient info
        accession = int(lbl_file.split('-')[0].split('/')[-1])

        # load label
        for _, dic in labels.items():

            if int(dic['ACCNO']) == accession:
                label = int(dic['LABEL'])
                mrn = int(dic['MRN'])
                text = dic['TEXT']

        # Load the segments
        segments = np.squeeze(sdl.load_NIFTY(lbl_file)).astype(np.uint8)

        # Define image file name and load
        vol_file = lbl_file[:-13] + '.nii.gz'
        volume = np.squeeze(sdl.load_NIFTY(vol_file)).astype(np.int16)

        # Normalize the MRI
        volume = sdl.normalize_MRI_histogram(volume)

        # Resize volumes
        volume = sdl.resize_volume(volume, np.float32, dims, dims)
        segments = sdl.resize_volume(segments.astype(np.uint8), np.uint8, dims, dims)

        # Loop through the image volume
        for z in range (volume.shape[0]):

            # Calculate a scaled slice shift
            sz = int(slice_gap)

            # Skip very bottom and very top of image
            if ((z-3*sz) < 0) or ((z+3*sz) > volume.shape[0]): continue

            # Label is easy, just save the slice
            data_label = sdl.zoom_2D(segments[z].astype(np.uint8), (dims, dims))

            # Generate the empty data array
            data_image = np.zeros(shape=[5, dims, dims], dtype=np.float32)

            # Set starting point
            zs = z - (2*sz)

            # Save 5 slices with shift Sz
            for s in range(5): data_image[s, :, :] = sdl.zoom_2D(volume[zs+(s*sz)].astype(np.float32), [dims, dims])

            # If there is label here, save 5x the slices
            sum_check = np.sum(np.squeeze(data_label > 1).astype(np.uint8))
            if sum_check > 5: num_egs = int(sum_check/45) + 1
            else: num_egs = 1

            for _ in range(num_egs):

                # Save the dictionary: int16, uint8, int, int
                data[index] = {'image_data': data_image, 'label_data': data_label, 'accession':accession,
                               'progression': label, 'file':vol_file, 'mrn': mrn, 'path': text, 'slice': z}

                # Finished with this slice
                index += 1

            # Garbage collection
            del data_label, data_image

        # Finished with all of this patients GBMS
        pts += 1

    # Finished with all the patients
    print('%s Patients loaded, %s slices saved (%s this protobuf)' % (pts, index, (index - per)))
    if len(data)>0: sdl.save_tfrecords(data, 3, 'data/CUMC_GBM_')


# For loading the BRATS files
def pre_proc_25D_BRATS(slice_gap, dims):

    """
    Loads boxes from the Iran data set
    :param slice_gap: the gap between slices to save for 3D data
    :param dims: the dimensions of the images saved
    :return:
    """

    # Load the files
    filenames = sdl.retreive_filelist('mha', include_subfolders=True, path=brats_dir)
    shuffle(filenames)
    print('Filenames: ', filenames)

    # Counters: index = example, pts = patients
    index, pts, per = 0, 0, 0

    # Data array
    data = {}

    # Testing
    class_count, display = [0, 0], []

    # Now do the magic
    for lbl_file in filenames:

        # Only work on T1C+ sequences
        if 'T1c' not in lbl_file: continue

        # Load the patient info
        base = lbl_file.split('/')[-3]
        accession = base.split('_')[-2] + '_' + base.split('_')[-1]
        label, mrn, text = lbl_file.split('/')[-4], base.split('_')[-2], base.split('_')[1]

        # Load the image: BRATS is 1mm resampled, 155, 240 image dims for the T1c
        image, _, _ = sdl.load_MHA(lbl_file)

        # Generate segment filename and load segments: Edema = 2, Enhancment =4, Encephalomalacia = 1
        segment_file_raw = sdl.retreive_filelist('mha', include_subfolders=True, path=(brats_dir + label + '/' + base + '/'))
        segment_file = [x for x in segment_file_raw if 'OT' in x]
        segments, _, _ = sdl.load_MHA(segment_file[0])
        segments = np.squeeze(segments >3)

        # Normalize the MRI
        image = sdl.normalize_MRI_histogram(image)

        # Resize volumes
        image = sdl.resize_volume(image, np.float32, dims, dims)
        segments = sdl.resize_volume(segments.astype(np.uint8), np.uint8, dims, dims)

        # Loop through the image volume
        for z in range (image.shape[0]):

            # Calculate a scaled slice shift
            sz = int(slice_gap)

            # Skip very bottom and very top of image
            if ((z-3*sz) < 0) or ((z+3*sz) > image.shape[0]): continue

            # Label is easy, just save the slice
            data_label = sdl.zoom_2D(segments[z].astype(np.uint8), (dims, dims))

            # Generate the empty data array
            data_image = np.zeros(shape=[5, dims, dims], dtype=np.float32)

            # Set starting point
            zs = z - (2*sz)

            # Save 5 slices with shift Sz
            for s in range(5): data_image[s, :, :] = sdl.zoom_2D(image[zs+(s*sz)].astype(np.float32), [dims, dims])

            # Save the dictionary: float32, uint8
            data[index] = {'image_data': data_image, 'label_data': data_label, 'accession':accession,
                           'progression': label, 'file':lbl_file, 'mrn': mrn, 'path': text, 'slice': z}

            # Finished with this slice
            index += 1

            # Garbage collection
            del data_label, data_image

        # Finished with all of this patients GBMS
        pts += 1
        del segments, image

        # Save every 65 patients
        if pts % 65 == 0:

            print('%s Patients loaded, %s slices saved (%s this protobuf)' % (pts, index, (index - per)))
            sdl.save_tfrecords(data, 1, 0, file_root=('data/BRATS_%s' % int(pts / 65)))
            if pts <100: sdl.save_dict_filetypes(data[0])

            del data
            data = {}
            per = index

        # Finished with all the patients
    if len(data) > 0: sdl.save_tfrecords(data, 1, 'data/BRATS_Fin')

# Load the protobuf
def load_protobuf():

    """
    Loads the protocol buffer into a form to send to shuffle
    """

    # Load all the filenames in glob
    filenames1 = sdl.retreive_filelist('tfrecords', False, path='data/')
    filenames = []

    # Define the filenames to remove
    for i in range (0, len(filenames1)):
        if FLAGS.test_files not in filenames1[i]: filenames.append(filenames1[i])

    # Show the file names
    print('Training files: %s' % filenames)

    # now load the remaining files
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims, tf.float32, z_dim=5, segments='label_data', segments_dtype=tf.uint8)
    print (data['image_data'], data['label_data'])

    # Image augmentation. First calc rotation parameters
    angle = tf.random_uniform([1], -0.52, 0.52)

    # Random rotate
    data['image_data'] = tf.contrib.image.rotate(data['image_data'], angle)
    data['label_data'] = tf.contrib.image.rotate(data['label_data'], angle)

    # Random gaussian noise
    data['image_data'] = tf.image.random_brightness(data['image_data'], max_delta=1.5)
    data['image_data'] = tf.image.random_contrast(data['image_data'], lower=0.95, upper=1.05)

    # Reshape image
    data['image_data'] = tf.image.resize_images(data['image_data'], [FLAGS.network_dims, FLAGS.network_dims])
    data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims])

    # For noise, first randomly determine how 'noisy' this study will be
    T_noise = tf.random_uniform([1], 0, 0.2)

    # Create a poisson noise array
    noise = tf.random_uniform(shape=[5, FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)

    # Add the gaussian noise
    data['image_data'] = tf.add(data['image_data'], tf.cast(noise, tf.float32))

    # Display the images
    tf.summary.image('Train IMG', tf.reshape(data['image_data'][2], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
    tf.summary.image('Train Label IMG', tf.reshape(data['label_data'], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)

    # Return data as a dictionary
    return sdl.randomize_batches(data, FLAGS.batch_size)


# Load the validation set
def load_validation():
    """
    Loads the protocol buffer into a form to send to shuffle
    :param
    :return:
    """

    # Load all the filenames in glob
    filenames1 = sdl.retreive_filelist('tfrecords', False, path='data/')
    filenames = []

    # Define the filenames to remove
    for i in range(0, len(filenames1)):
        if FLAGS.test_files in filenames1[i]:
            filenames.append(filenames1[i])

    # Show the file names
    print('Testing files: %s' % filenames)

    # now load the remaining files
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims, tf.float32, z_dim=5, segments='label_data', segments_dtype=tf.uint8)

    # Reshape image
    data['image_data'] = tf.image.resize_images(data['image_data'], [FLAGS.network_dims, FLAGS.network_dims])
    data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims])

    # Display the images
    tf.summary.image('Test IMG', tf.reshape(data['image_data'][2], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
    tf.summary.image('Test Label IMG', tf.reshape(data['label_data'], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)

    # Return data as a dictionary
    return sdl.val_batches(data, FLAGS.batch_size)

def check_labels():

    """
    Check the files to make sure there is an annotation available
    :return:
    """

    # Load the files
    filenames = sdl.retreive_filelist('gz', include_subfolders=True, path=cumc_dir)
    shuffle(filenames)

    # Load the labels: {'1': {'LABEL': '1', 'TEXT': 'recurrent, with treatment effect', 'ACCNO': '4178333', 'MRN': '7647499'}, '3': ...
    label_file = sdl.retreive_filelist('csv', False, path='data/')[0]
    labels = sdl.load_CSV_Dict('ID', label_file)

    # Now check that all files have labels
    for file in filenames:

        # Only work on label files
        if 'label' not in file: continue
        accession2 = int(file.split('-')[0].split('/')[-1])
        exists2 = False

        # First check that all labels have nii.gz files
        for _, dic in labels.items():

            if int(dic['ACCNO'])==accession2:
                exists2 = True

        if not exists2: print('No label found: %s' % file)

    # First check that all labels have nii.gz files
    for _, dic in labels.items():

        acc = int(dic['ACCNO'])
        label = dic['LABEL']
        exists = False

        for file in filenames:

            # Only work on label files
            if 'label' not in file: continue
            accession = int(file.split('-')[0].split('/')[-1])

            if acc==accession:
                exists = True
                break

        if not exists: print ('Non Existant file found: %s' %dic)

# pre_proc_25D_BRATS(2, 256)