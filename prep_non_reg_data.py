#!/usr/bin/env python3

"""
The following code will download datasets and preprocess datasets.
"""
import numpy as np
import re, os
DATA_PATH = './datasets' # You may want to change this line. Set a path to a data directory.
DOWNLOAD_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
list_datasets=['skin_nonskin', 'diabetes_scale', 'australian_scale']
for dataset_name in list_datasets:
    np_list = []
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
        os.mkdir(DATA_PATH+'/reg_datasets')
        os.mkdir(DATA_PATH+'/clf_datasets')
        os.mkdir(DATA_PATH+'/image_datasets')
        print('Download regression datasets and extract features from image datasets \
            in order to reproduce point addition experiments. \
            Please see comments in data.py')

    # The following code will download datasets

    # NOTE: Make sure your system has wget installed.
    os.system(f'wget -t inf '+DOWNLOAD_URL+dataset_name)
    os.system(f'mv ./{dataset_name} {DATA_PATH}/clf_datasets/{dataset_name}')
    fh = open(DATA_PATH+f'/clf_datasets/{dataset_name}','r')
    error_count, chunk_number = 0, 1
    for i, line in enumerate(fh):
        if i == 0:
            print('-'*30)
            print(f'dataset: {dataset_name}')
            print('-'*30)
            sample = re.split(' |:| \n', line)
            sample_length = len(sample[0::2])
            print('Sample[:5]/length: ', sample[0:10:2], sample_length)
        line_ = re.split(' |:| \n', line)
        line_ = line_[0::2]
        if len(line_) == sample_length:
            np_list.append(line_)
        else:
            error_count += 1
    fh.close()
        
    # save
    np_list = np.vstack(np_list).astype('float32')
    print('Final dataset shape:', dataset_name, np_list.shape, flush=True)
    np.save(file=DATA_PATH+f'/clf_datasets/{dataset_name}.npy', arr=np_list)

    # show statistics
    print('Dataset/Error count : ', dataset_name, '/', error_count)
    del fh, np_list
    print('')


