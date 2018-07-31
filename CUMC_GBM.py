"""
For the columbia university hepatocellular cancer data
"""

import numpy as np
import SODLoader as SDL
import SOD_Display as Display

from pathlib import Path
from random import shuffle

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/GBM/TxResponse/'

# Utility classes
sdl = SDL.SODLoader(data_root=home_dir)
sdd = Display.SOD_Display()
disp = Display.SOD_Display()

def pre_process(chunks=2):

    """
    Loads the files to a pickle dictionary
    :param chunks: How many chunks of data to save
    :return:
    """

    # Load the files
    filenames = sdl.retreive_filelist('gz', include_subfolders=True, path=home_dir)
    shuffle(filenames)

    # Load the labels: {'1': {'LABEL': '1', 'TEXT': 'recurrent, with treatment effect', 'ACCNO': '4178333', 'MRN': '7647499'}, '3': ...
    label_file = sdl.retreive_filelist('csv', False, 'data/')[0]
    labels = sdl.load_CSV_Dict('ID', label_file)

    total_volumes = len(filenames) // 2

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
        volume = sdl.normalize_MRI_histogram(volume, center_type='mean')

        # Easy, these are all T1 post contrast studies
        sequence = 'd1'

        """
        Prepare this patients data to save it to a custom pickle dictionary
        The volumes are numpy arrays. The rest of the indices are characters
        Accession numbers are more unique than MRNs. Save both. If both aren't available and every file is a unique patient then just duplicate.
        Sequence characters should be -'t1', 't2' etc with 'd1', 'd2', 'dn' etc for dynamic phases
        Please make sure your unique_pt_id is unique to all the datasets. Same for study ID. 
        MRN_Equivalent should be unique for each patient but some patients will have multiple studies.
        """

        data[index] = {'volume': volume.astype(np.float16), 'segments': segments.astype(np.uint8), 'label': 'gbm',
                       'mrn': str(mrn), 'accession': str(accession), 'sequence': sequence,'xy_spacing': 'na', 'z_spacing': 'na'}

        # Garbage
        del segments, volume
        index +=1
        if index % 10 == 0: print ('\n%s of %s volumes loaded... %s%% done\n' %(index, total_volumes, 100*index//total_volumes))

    # Save the dictionary in x chunks
    print ('Loaded %s volumes fully.' %len(data))
    split_dicts = sdl.split_dict_equally(data, chunks)
    for z in range (chunks): sdl.save_dict_pickle(split_dicts[z], ('data/CUMC_GBM%s' %(z+1)))

pre_process(1)