"""
Helper and utility functions
"""

import SODLoader as SDL
import SOD_Display as SDD
from random import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import pydicom as dicom
import imageio

sdl = SDL.SODLoader('data')
sdd = SDD.SOD_Display()

home_dir = '/home/stmutasa/PycharmProjects/Datasets/GBM/Raw_Downloaded_GBMS2'

def sort_DICOMs():

    """
    Helper function to sort the raw downloaded dicoms into a folder structure that makes sense
    MRN/Accno/Series_Time.nii.gz
    MRN/Accno/Series_Time.gif
    MRN/Accno/Series_Time.json
    :return:
    """

    # First retreive lists of the the filenames
    interleaved = sdl.retreive_filelist('**', path=home_dir, include_subfolders=True)
    shuffle(interleaved)
    folders = list()
    for (dirpath, dirnames, filenames) in os.walk(home_dir):
        if filenames: folders.append(dirpath)
    shuffle(folders)

    # Variables to save
    patient, study = 0, 0

    # Load the images and filter them
    for folder in folders:

        try:
            # Read multiple volumes with Imageio (multiple DICOM series)
            vols = imageio.mvolread(folder, 'DICOM')
        except Exception as e:
            print('Image Error: %s,  --- Folder: %s' % (e, folder))
            continue

        try:
            # Get header the old way
            header = sdl.load_DICOM_Header(folder)
            Accno = header['tags'].AccessionNumber
            MRN = header['tags'].PatientID

        except Exception as e:
            # Print error then make a dummy header
            print('Header Error: %s,  --- Folder: %s' % (e, folder))
            Accno = folder.split('/')[6]
            MRN = ('UknownMRN_%s' %patient)
            header = {'MRN': MRN, 'Accno': Accno, 'path': folder}

        """
         TODO: Sort the folders and save as niftis
        """

        for volume in vols:

            # Convert to numpy
            volume = np.asarray(volume, np.int16)
            center = volume[volume.shape[0] // 2]

            # Skip obvious scout and other small series
            if volume.shape[0] <= 20: continue

            # Get savefile names
            save_root = home_dir + '/Sorted'
            fname_vol = ('%s/%s_%s_%s_%s.nii.gz' %(save_root, patient, MRN, Accno, study))
            fname_gif = fname_vol.replace('nii.gz', 'gif')
            fname_header = fname_vol.replace('nii.gz', 'p')
            img_path = fname_gif.replace('gif', 'jpg')

            # Create the root folder
            if not os.path.exists(os.path.dirname(fname_vol)): os.mkdir(os.path.dirname(fname_vol))

            # Save the gif and volume
            print('Saving: ', os.path.basename(fname_vol))

            try:
                sdl.save_gif_volume(volume, fname_gif)
                sdl.save_volume(volume, fname_vol, compress=True)
                sdl.save_image(center, img_path)
            except Exception as e:
                print('\nSaving Error %s: %s,  --- Folder: %s' % (volume.shape, e, folder))
                continue

            # Increment
            study += 1
            del volume

        # Save the header
        with open(fname_header, 'wb') as fp: pickle.dump(header, fp, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(fname_header, 'w') as fp: json.dump(header, fp)
        patient +=1


def sort_DICOMs2():

    """
    Helper function to sort the raw downloaded dicoms into a folder structure that makes sense
    MRN/Accno/Series_Time.nii.gz
    MRN/Accno/Series_Time.gif
    MRN/Accno/Series_Time.json
    :return:
    """

    # Path for this function
    path = '/home/stmutasa/PycharmProjects/Datasets/GBM/Wendy_GBMs'

    # First retreive lists of the the filenames
    folders = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        folders += [os.path.join(dirpath, dir) for dir in dirnames]
    folders = [x for x in folders if 'OBJ_0' in x]

    # Keep track of last MRN
    last_MRN = '10001'

    # Load the images and filter them
    for folder in folders:

        # Load the DICOMs
        try:
            volume, header = sdl.load_DICOM_3D(folder, return_header=True)
        except Exception as e:
            print('Image Error: %s,  --- Folder: %s' % (e, folder))
            continue

        # Retreive the headers
        try:
            Series = header['tags'].SeriesDescription
            Accno = header['tags'].AccessionNumber
            Time = header['tags'].AcquisitionTime
            MRN = header['tags'].PatientID
        except Exception as e:
            print('Header Error: %s,  --- Folder: %s' % (e, folder))
            continue

        # Remove illegal characters from series name
        try: Series.replace('/', '')
        except: pass

        # Convert to numpy
        volume = np.asarray(volume, np.int16)
        center = volume[volume.shape[0] // 2]

        # Skip obvious scout and other small series
        if volume.shape[0] <= 20: continue

        # Get savefile names
        save_root = path + '/Sorted'
        fname_vol = ('%s/%s/%s/%s_%s.nii.gz' %(save_root, MRN, Accno, Series, Time))
        fname_gif = fname_vol.replace('nii.gz', 'gif')
        fname_header = ('%s/%s/%s/Header.p' %(save_root, MRN, Accno))
        img_path = fname_gif.replace('gif', 'jpg')

        # Create the root folder
        if not os.path.exists(os.path.dirname(fname_vol)): os.makedirs(os.path.dirname(fname_vol))

        # Save the gif and volume
        print('Saving: ', os.path.basename(fname_vol))

        try:
            sdl.save_gif_volume(volume, fname_gif)
            sdl.save_volume(volume, fname_vol, compress=True)
            sdl.save_image(center, img_path)
        except Exception as e:
            print('\nSaving Error %s: %s,  --- Folder: %s' % (volume.shape, e, folder))
            continue

        # Save the header
        if MRN != last_MRN:
            with open(fname_header, 'wb') as fp: pickle.dump(header, fp, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(fname_header, 'w') as fp: json.dump(header, fp)

        # garbage
        del volume, center
        last_MRN = MRN


def load_DICOM_3D(path, dtype=np.int16, sort=False, overwrite_dims=513, display=False, return_header=False):

    # Some DICOMs end in .dcm, others do not
    fnames = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        fnames += [os.path.join(dirpath, file) for file in filenames]

    # Load the dicoms
    ndimage = [dicom.read_file(path, force=True) for path in fnames]

    # # Calculate how many volumes are interleaved here
    # sort_list = np.asarray([x.SliceLocation for x in ndimage], np.int16)
    # _, counts = np.unique(sort_list, return_counts=True)
    # repeats = np.max(counts)

    # Sort the slices
    if sort:
        if 'Lung' in sort: ndimage = sdl.sort_DICOMS_Lung(ndimage, display, path)
        elif 'PE' in sort: ndimage = sdl.sort_DICOMS_PE(ndimage, display, path)
    ndimage, fnames, orientation, st, shape, four_d = sdl.sort_dcm(ndimage, fnames)
    ndimage.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    # Retreive the dimensions of the scan
    try: dims = np.array([int(ndimage[0].Columns), int(ndimage[0].Rows)])
    except: dims = np.array([overwrite_dims, overwrite_dims])

    # Retreive the spacing of the pixels in the XY dimensions
    pixel_spacing = ndimage[0].PixelSpacing

    # Create spacing matrix
    numpySpacing = np.array([st, float(pixel_spacing[0]), float(pixel_spacing[1])])

    # Retreive the origin of the scan
    orig = ndimage[0].ImagePositionPatient

    # Make a numpy array of the origin
    numpyOrigin = np.array([float(orig[2]), float(orig[0]), float(orig[1])])

    # --- Save first slice for header information
    header = {'orientation': orientation, 'slices': shape[1], 'channels': shape[0], 'num_Interleaved': 1,
              'fnames': fnames, 'tags': ndimage[0], '4d': four_d, 'spacing': numpySpacing, 'origin': numpyOrigin}

    # Finally, make the image actually equal to the pixel data and not the header
    image = np.stack([sdl.read_dcm_uncompressed(s) for s in ndimage])

    image = sdl.compress_bits(image)

    # Set image data type to the type specified
    image = image.astype(dtype)

    # Convert to Houndsfield units
    if hasattr(ndimage[0], 'RescaleIntercept') and hasattr(ndimage[0], 'RescaleSlope'):
        for slice_number in range(len(ndimage)):
            intercept = ndimage[slice_number].RescaleIntercept
            slope = ndimage[slice_number].RescaleSlope

            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype('int16')
            image[slice_number] += np.int16(intercept)

    if return_header:
        return image, header
    else:
        return image, numpyOrigin, numpySpacing, dims

sort_DICOMs()