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

sdl = SDL.SODLoader('data')
sdd = SDD.SOD_Display()

home_dir = 'xxx'

def sort_DICOMs():

    """
    Helper function to sort the raw downloaded dicoms into a folder structure that makes sense
    MRN/Accno/Series_Time.nii.gz
    MRN/Accno/Series_Time.gif
    MRN/Accno/Series_Time.json
    :return:
    """

    # First retreive lists of the the filenames
    interleaved = sdl.retreive_filelist('dcm', path=home_dir, include_subfolders=True)
    shuffle(interleaved)
    redownloads = list()
    for (dirpath, dirnames, filenames) in os.walk(home_dir + 'DICOM/'):
        redownloads += [os.path.join(dirpath, dir) for dir in dirnames]
    redownloads = [x for x in redownloads if 'OBJ_0' in x]
    shuffle(redownloads)

    # Load the images and filter them
    for folder in redownloads:

        # First Load the study
        volume, header = sdl.load_DICOM_3D(folder, return_header=True)
        try:
            Series = header['tags'].SeriesDescription
            Accno = header['tags'].AccessionNumber
            Time = header['tags'].AcquisitionTime
            MRN = header['tags'].PatientID
        except:
            continue

        """
         TODO: Sort the folders and save as niftis
        """

        # Get savefile names
        save_root = home_dir + 'Sorted/'
        series_time = ('%s_%s' %(Series, Time))
        save_vol = (save_root, '%s/%s/%s.nii.gz' %(MRN, Accno, series_time))
        save_gif = save_vol.replace('nii.gz', 'gif')
        save_header = save_vol.replace('nii.gz', 'json')

        # Create the root folder
        if not os.path.exists(os.path.dirname(save_vol)): os.mkdir(os.path.dirname(save_vol))

        # Save the gif and volume
        print('Saving: ', os.path.basename(save_vol))
        sdl.save_volume(volume, save_vol, compress=True)
        sdl.save_gif_volume(volume, save_gif)
        # with open(save_header, 'wb') as fp: pickle.dump(header, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_header, 'w') as fp: json.dump(header, fp)