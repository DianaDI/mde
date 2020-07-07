from glob import glob
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.data.transforms import rebin, minmax_over_nonzero, minmax_custom, interpolate_on_missing, get_edges
from src.data import MIN_DEPTH, MAX_DEPTH


class DatadirParser():
    def __init__(self, data_dir="/mnt/data/davletshina/datasets/Bera_MDE"):
        self.data_dir = f'{data_dir}/splits2'
        self.img_name_prefixes = ["KirbyLeafOff2017RGBNEntireSite", "KirbyLeafOn2017RGBNEntireSite"]
        self.depth_dir = f'{data_dir}/depth_maps2/*'
        self.img_list = self.get_files(self.data_dir, self.img_name_prefixes)
        self.depth_list = sorted(glob(self.depth_dir))

    def get_files(self, data_dir, prefixes):
        file_paths = list()
        for prefix in prefixes:
            for dir in glob(f'{data_dir}/{prefix}*'):
                file_paths.extend(glob(f'{dir}/*'))
        return sorted(file_paths)

    def get_parsed(self):
        return self.img_list, self.depth_list


class TrainValTestSplitter:
    def __init__(self, images, depth, test_size=0.1, val=True, random_seed=42):
        """
        Train-validation-test splitter, stores all the filenames
        :param path_to_data: path to images
        :param val: boolean, true if validation set needed to be split up
        """
        self.val = val
        self.data = pd.DataFrame()
        self.data['image'] = images
        self.data['depth'] = depth
        self.test_size = test_size
        self.random_state = random_seed
        self.data_train = pd.DataFrame()
        self.data_test = pd.DataFrame()
        self.data_val = pd.DataFrame()
        self._split_data()

    def _split_stats(self, df):
        print(f'Size: {len(df)}')
        print(f'Percentage from original images: {round(len(df) / len(self.data), 3)}')

    def _split_data(self):
        """
        Creates data_train, data_val, data_test dataframes with filenames
        """

        data_train, data_test, labels_train, labels_test = train_test_split(self.data['image'], self.data['depth'], test_size=self.test_size,
                                                                            random_state=self.random_state)
        if self.val:
            data_train, data_val, labels_train, labels_val = train_test_split(data_train.reset_index(drop=True), labels_train.reset_index(drop=True),
                                                                              test_size=self.test_size,
                                                                              random_state=self.random_state)

        print('=============Train subset===============')
        self.data_train['image'] = data_train.reset_index(drop=True)
        self.data_train['depth'] = labels_train.reset_index(drop=True)
        self._split_stats(data_train)

        print('=============Test subset===============')
        self.data_test['image'] = data_test.reset_index(drop=True)
        self.data_test['depth'] = labels_test.reset_index(drop=True)
        self._split_stats(data_test)

        if self.val:
            print('===========Validation subset============')
            self.data_val['image'] = data_val.reset_index(drop=True)
            self.data_val['depth'] = labels_val.reset_index(drop=True)
            self._split_stats(data_val)


class BeraDataset(Dataset):
    def __init__(self, img_filenames, depth_filenames, normalise=True, normalise_type='local', interpolate=False):
        self.img_filenames = img_filenames
        self.depth_filenames = depth_filenames
        self.normalize = normalise
        self.normalize_type = normalise_type
        self.interpolate = interpolate
        self.dm_dim = (128, 128)

    def __len__(self):
        return len(self.depth_filenames)

    def __getitem__(self, index):
        """Reads sample"""
        image = cv2.imread(self.img_filenames[index])
        edges = get_edges(image, self.dm_dim) / 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.load(self.depth_filenames[index], allow_pickle=True)
        #label = label / 1000 # convert to kilometers
        label = rebin(label, self.dm_dim)
        # range = np.array([np.min(label[np.nonzero(label)]), np.max(label[np.nonzero(label)])])
        # range = range - (MIN_DEPTH / 1000)
        if self.normalize:
            if self.normalize_type == 'local':
                label = minmax_over_nonzero(label)
            else:
                label = minmax_custom(label, MIN_DEPTH, MAX_DEPTH)
            mask = (label >= 0).astype(int)  # 0 is smallest after minmax
        else:
            mask = (label > 0).astype(int)
        if self.interpolate:
            if np.min(mask) == 0:
                label = interpolate_on_missing(label * mask)
        return {'image': image, 'depth': label, 'mask': mask, 'edges': edges}
