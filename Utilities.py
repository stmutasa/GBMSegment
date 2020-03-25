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

home_dir = '/home/stmutasa/Code/Datasets/GBM/GBM_Redownloads/'

def sort_DICOMs():

    """
    Helper function to sort the raw downloaded dicoms into a folder structure that makes sense
    MRN/Accno/Series_Time.nii.gz
    MRN/Accno/Series_Time.gif
    MRN/Accno/Series_Time.json
    :return:
    """

    # First retreive lists of the the filenames
    folders = list()
    for (dirpath, dirnames, filenames) in os.walk(home_dir):
        if filenames: folders.append(dirpath)
    shuffle(folders)

    # Variables to save
    patient, study = 0, 0

    # Load the images and filter them
    for folder in folders:

        try:
            vols = load_DICOM_VOLS(folder)
            key = next(iter(vols))
            header = vols[key]['header']
        except Exception as e:
            print('Image Error: %s,  --- Folder: %s' % (e, folder))
            continue

        try:
            # Get header the old way
            Accno = header['tags'].AccessionNumber
            MRN = header['tags'].PatientID
        except Exception as e:
            # Print error then make a dummy header
            print('Header Error: %s,  --- Folder: %s' % (e, folder))
            Accno = folder.split('/')[-2]
            MRN = ('UknownMRN_%s' %patient)
            header = {'MRN': MRN, 'Accno': Accno, 'path': folder}

        """
         TODO: Sort the folders and save as niftis
        """

        for series, dict in vols.items():

            # Convert to numpy
            volume = dict['volume']
            volume = np.asarray(volume, np.int16)
            center = volume[volume.shape[0] // 2]

            # Skip obvious scout and other small series
            if volume.shape[0] <= 20: continue

            # Get savefile names
            series = series.replace('(', '').replace(')', '')
            save_root = home_dir.replace('GBM_Redownloads/', 'Sorted')
            fname_vol = ('%s/%s_%s_%s_%s.nii.gz' %(save_root, series, MRN, Accno, study))
            fname_header = fname_vol.replace('nii.gz', 'p')
            # fname_gif = fname_vol.replace('nii.gz', 'gif')
            # img_path = fname_gif.replace('gif', 'jpg')

            # Create the root folder
            if not os.path.exists(os.path.dirname(fname_vol)): os.mkdir(os.path.dirname(fname_vol))

            # Save the gif and volume
            print('Saving: ', os.path.basename(fname_vol))

            try:
                # sdl.save_gif_volume(volume, fname_gif)
                # sdl.save_image(center, img_path)
                sdl.save_volume(volume, fname_vol, compress=True)
            except Exception as e:
                print('\nSaving Error %s: %s,  --- Folder: %s' % (volume.shape, e, folder))
                continue

            # Increment
            study += 1
            del volume

        # Save the header
        with open(fname_header, 'wb') as fp: pickle.dump(header, fp, protocol=pickle.HIGHEST_PROTOCOL)
        patient +=1
        del vols


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


def load_DICOM_VOLS(path):

    """
    This function loads a bunch of jumbled together DICOMs as separate volumes
    :param: path: The path of the DICOM folder
    :param: overwrite_dims = In case slice dimensions can't be retreived, define overwrite dimensions here
    :param: display = Whether to display debug text
    :param return_header = whether to return the header dictionary
    :return: image = A 3D numpy array of the image
    :return header: a dictionary of the file's header information
    """

    # Array of DICOM objects to make
    volumes, save_dict = {}, {}

    # Some DICOMs end in .dcm, others do not
    fnames = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        fnames += [os.path.join(dirpath, file) for file in filenames]

    # Load the dicoms
    dicoms = [dicom.read_file(path, force=True) for path in fnames]

    # Check for actual images
    def _SPP_Exists(dcm):
        try:
            _ = dcm.SamplesPerPixel, dcm.ImagePositionPatient
            return True
        except: return False

    # Get images only using SamplesPerPixel
    _dicoms = [x for x in dicoms if _SPP_Exists(x)]

    # Now go through and add to unique volumes
    for dcm in _dicoms:
        SIUID = dcm.SeriesInstanceUID
        if SIUID in volumes.keys():
            volumes[SIUID].append(dcm)
        else: volumes[SIUID] = []

    del _dicoms

    # Now work on each volume separately
    for SIUD, dicoms in volumes.items():

        # Sort the slices
        dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]))

        # --- Save first slice for header information
        header = {'tags': dicoms[0]}

        # Finally, load pixel data. You can use Imageio here
        try: image = np.stack([sdl.read_dcm_uncompressed(s) for s in dicoms])
        except: image = imageio.volread(path, 'DICOM')
        image = sdl.compress_bits(image)

        # Convert to Houndsfield units
        if hasattr(dicoms[0], 'RescaleIntercept') and hasattr(dicoms[0], 'RescaleSlope'):
            for slice_number in range(len(dicoms)):
                intercept = dicoms[slice_number].RescaleIntercept
                slope = dicoms[slice_number].RescaleSlope

                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype('int16')
                image[slice_number] += np.int16(intercept)

        # Get description and save dictionary
        SDesc = str(dicoms[0].SeriesDescription)
        if SDesc in save_dict.keys():
            SDesc = SDesc + ('_' + str(dicoms[0].SeriesNumber))
        save_dict[SDesc] = {'header': header, 'volume': image}

    # Return if not empty
    if not save_dict: return
    else: return save_dict

sort_DICOMs()