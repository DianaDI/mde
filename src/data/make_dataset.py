from glob import glob
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DatadirParser():
    def __init__(self, data_dir="/mnt/data/davletshina/datasets/Bera_MDE"):
        self.data_dir = f'{data_dir}/splits2'
        self.pc_name_prefixes = ["KirbyLeafOff2017PointCloudEntireSite", "KirbyLeafOn2017PointCloudEntireSite"]
        self.img_name_prefixes = ["KirbyLeafOff2017RGBNEntireSite", "KirbyLeafOn2017RGBNEntireSite"]
        self.depth_dir = f'{data_dir}/depth_maps2/*'
        self.pc_list = self.get_files(self.data_dir, self.pc_name_prefixes)
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
    def __init__(self, images, depth, test_size=0.2, val=True):
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
        self.random_state = 42
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


def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging over nonzero elements."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    arr2 = arr.reshape(shape)
    cond = (arr2 > 0).sum(axis=(1, 3))
    out = np.zeros(new_shape)
    np.true_divide(arr2.sum(axis=(1, 3)), cond, where=(cond) > 0, out=out)
    return out


class BeraDataset(Dataset):
    def __init__(self, img_filenames, depth_filenames):
        self.img_filenames = img_filenames
        self.depth_filenames = depth_filenames

    def __len__(self):
        return len(self.depth_filenames)

    def __getitem__(self, index):
        """Reads sample"""
        image = cv2.imread(self.img_filenames[index])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.load(self.depth_filenames[index], allow_pickle=True)
        label = rebin(label, (128, 128))
        mask = (label != 0).astype(int)
        return {'image': image, 'depth': label, 'mask': mask}
