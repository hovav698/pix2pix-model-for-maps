import params
from dataloader import DataLoaderInput
from torch.utils.data import DataLoader
import glob
from urllib import request
import shutil
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#download and extract the maps dataset
def get_data():
    if not os.path.exists('maps'):
        url = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz'
        local_file = 'maps.tar.gz'
        request.urlretrieve(url, local_file)

        shutil.unpack_archive('maps.tar.gz')


#load the data to the pytorch data loader
def load_data(files_path):
    files_name = glob.glob(files_path + "/*")
    dataset = DataLoaderInput(files_name)
    data_loader = DataLoader(dataset, params.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    return data_loader




