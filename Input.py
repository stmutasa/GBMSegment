"""
Load and preprocess the files to a protobuff
"""

import os

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import matplotlib.pyplot as plt

from random import shuffle

# Define flags
FLAGS = tf.app.flags.FLAGS

# Define an instance of the loader file
sdl = SDL.SODLoader(os.getcwd())

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
    filenames = sdl.retreive_filelist('gz', include_subfolders=True, path='data/raw/')
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

        # Reshape all to xxx
        new_dims = 512
        if segments.shape[-1] != new_dims:

            # Create empty array for broadcasting
            reshape_seg, reshape_vol = np.zeros((segments.shape[0], new_dims, new_dims), np.uint8), np.zeros((segments.shape[0], new_dims, new_dims), np.int16)

            # Slice by slice zoom
            for z in range(segments.shape[0]):
                reshape_vol[z] = sdl.zoom_2D(volume[z], [new_dims, new_dims])
                reshape_seg[z] = sdl.zoom_2D(segments[z], [new_dims, new_dims])

            # Rebroadacast
            del volume, segments
            volume, segments = reshape_vol, reshape_seg

        # Normalize the volume: TODO Create mask
        volume = sdl.normalize(volume).astype(np.float32)

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

        # Save every 5 patients
        if pts %15 == 0:

            print ('%s Patients loaded, %s slices saved (%s this protobuf)' %(pts, index, (index-per)))

            sdl.save_tfrecords(data, 1, 0, file_root=('data/Embo%s' %int(pts/15)))

            del data
            data = {}
            per=index

    # Finished with all the patients
    if len(data)>0: sdl.save_tfrecords(data, 1, 'data/EmboFin')

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
        if FLAGS.test_files not in filenames1[i]:
            filenames.append(filenames1[i])

    # Show the file names
    print('Training files: %s' % filenames)

    # now load the remaining files
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)

    reader = tf.TFRecordReader()  # Instantializes a TFRecordReader which outputs records from a TFRecords file
    _, serialized_example = reader.read(filename_queue)  # Returns the next record (key:value) produced by the reader

    # Restore the feature dictionary to store the variables we will retrieve using the parse
    feature_dict = {'id': tf.FixedLenFeature([], tf.int64), 'image_data': tf.FixedLenFeature([], tf.string),
                    'label_data': tf.FixedLenFeature([], tf.string), 'slice': tf.FixedLenFeature([], tf.string),
                    'accession': tf.FixedLenFeature([], tf.string), 'mrn': tf.FixedLenFeature([], tf.string),
                    'file': tf.FixedLenFeature([], tf.string), 'progression': tf.FixedLenFeature([], tf.string),
                    'path': tf.FixedLenFeature([], tf.string)}

    # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data
    features = tf.parse_single_example(serialized_example, features=feature_dict)

    box_dims = 256

    # Load the raw image data
    image = tf.decode_raw(features['image_data'], tf.float32)
    image = tf.reshape(image, shape=[5, box_dims, box_dims, 1])
    label = tf.decode_raw(features['label_data'], tf.uint8)
    label = tf.reshape(label, shape=[box_dims, box_dims, 1])

    # Cast all our data to 32 bit floating point units.
    id = tf.cast(features['id'], tf.float32)
    accession = tf.string_to_number(features['accession'], tf.float32)
    slice = tf.string_to_number(features['slice'], tf.float32)
    mrn = tf.string_to_number(features['mrn'], tf.float32)
    progression = tf.string_to_number(features['progression'], tf.float32)

    path = tf.cast(features['path'], tf.string)
    file = tf.cast(features['file'], tf.string)
    label = tf.cast(label, tf.float32)

    # Image augmentation. First calc rotation parameters
    angle = tf.random_uniform([1], -0.52, 0.52)
    image = tf.add(image, 200.0)

    # Random rotate
    image = tf.contrib.image.rotate(image, angle)
    label = tf.contrib.image.rotate(label, angle)

    # Return image to center
    image = tf.subtract(image, 200.0)

    # Random gaussian noise
    image = tf.image.random_brightness(image, max_delta=5)
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)

    # Reshape image
    image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims])
    label = tf.image.resize_images(label, [FLAGS.network_dims, FLAGS.network_dims])

    # For noise, first randomly determine how 'noisy' this study will be
    T_noise = tf.random_uniform([1], 0, FLAGS.noise_threshold)

    # Create a poisson noise array
    noise = tf.random_uniform(shape=[5, FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)

    # Add the gaussian noise
    image = tf.add(image, tf.cast(noise, tf.float32))

    # Display the images
    tf.summary.image('Train IMG', tf.reshape(image[2], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
    tf.summary.image('Train Label IMG', tf.reshape(label, shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)

    ## Return data as a dictionary
    final_data = {'image': image, 'label': label, 'accession': accession, 'path': path, 'file': file, 'mrn': mrn, 'progression': progression, 'slice': slice}

    returned_dict = {}
    returned_dict['id'] = id
    for key, feature in final_data.items():
        returned_dict[key] = feature
    return returned_dict

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
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)

    reader = tf.TFRecordReader()  # Instantializes a TFRecordReader which outputs records from a TFRecords file
    _, serialized_example = reader.read(filename_queue)  # Returns the next record (key:value) produced by the reader

    # Restore the feature dictionary to store the variables we will retrieve using the parse
    feature_dict = {'id': tf.FixedLenFeature([], tf.int64), 'image_data': tf.FixedLenFeature([], tf.string),
                    'label_data': tf.FixedLenFeature([], tf.string), 'slice': tf.FixedLenFeature([], tf.string),
                    'accession': tf.FixedLenFeature([], tf.string), 'mrn': tf.FixedLenFeature([], tf.string),
                    'file': tf.FixedLenFeature([], tf.string), 'progression': tf.FixedLenFeature([], tf.string),
                    'path': tf.FixedLenFeature([], tf.string)}

    # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data
    features = tf.parse_single_example(serialized_example, features=feature_dict)

    box_dims = 256

    # Load the raw image data
    image = tf.decode_raw(features['image_data'], tf.float32)
    image = tf.reshape(image, shape=[5, box_dims, box_dims, 1])
    label = tf.decode_raw(features['label_data'], tf.uint8)
    label = tf.reshape(label, shape=[box_dims, box_dims, 1])

    # Cast all our data to 32 bit floating point units.
    id = tf.cast(features['id'], tf.float32)
    accession = tf.string_to_number(features['accession'], tf.float32)
    slice = tf.string_to_number(features['slice'], tf.float32)
    mrn = tf.string_to_number(features['mrn'], tf.float32)
    progression = tf.string_to_number(features['progression'], tf.float32)

    path = tf.cast(features['path'], tf.string)
    file = tf.cast(features['file'], tf.string)
    label = tf.cast(label, tf.float32)

    # Reshape image
    image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims])
    label = tf.image.resize_images(label, [FLAGS.network_dims, FLAGS.network_dims])

    # Display the images
    tf.summary.image('Test IMG', tf.reshape(image[2], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
    tf.summary.image('Test Label IMG', tf.reshape(label, shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)

    # Return data as a dictionary
    final_data = {'image': image, 'label': label, 'accession': accession, 'path': path, 'file': file, 'mrn': mrn, 'progression': progression, 'slice': slice}

    returned_dict = {}
    returned_dict['id'] = id
    for key, feature in final_data.items():
        returned_dict[key] = feature
    return returned_dict


def check_labels():

    """
    Check the files to make sure there is an annotation available
    :return:
    """

    # Load the files
    filenames = sdl.retreive_filelist('gz', include_subfolders=True, path='data/raw/')
    shuffle(filenames)

    # Load the labels: {'1': {'LABEL': '1', 'TEXT': 'recurrent, with treatment effect', 'ACCNO': '4178333', 'MRN': '7647499'}, '3': ...
    label_file = sdl.retreive_filelist('csv', False, 'data/')[0]
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