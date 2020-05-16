from glob import glob
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DatadirParser():
    def __init__(self, data_dir="/mnt/data/davletshina/datasets/Bera_MDE"):
        self.data_dir = f'{data_dir}/splits'
        self.pc_name_prefixes = ["KirbyLeafOff2017PointCloudEntireSite", "KirbyLeafOn2017PointCloudEntireSite"]
        self.img_name_prefixes = ["KirbyLeafOff2017RGBNEntireSite", "KirbyLeafOn2017RGBNEntireSite"]
        self.depth_dir = f'{data_dir}/depth_maps/*'
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
    def __init__(self, data, depth, test_size=0.2, val=True):
        """
        Train-validation-test splitter, stores all the filenames
        :param path_to_data: path to images
        :param val: boolean, true if validation set needed to be split up
        """
        self.val = val
        self.data = pd.DataFrame()
        self.data['image'] = data
        self.data['depth'] = depth
        self.test_size = test_size
        self.random_state = 42
        self.data_train = pd.DataFrame()
        self.data_test = pd.DataFrame()
        self.data_val = pd.DataFrame()
        self._split_data()

    def _split_stats(self, df):
        print(f'Size: {len(df)}')
        print(f'Percentage from original data: {len(df) / len(self.data)}')

    def _split_data(self):
        """
        Creates data_train, data_val, data_test dataframes with filenames
        """

        data_train, data_test, labels_train, labels_test = train_test_split(self.data['image'], self.data['depth'], test_size=self.test_size,
                                                                            random_state=self.random_state)
        if self.val:
            data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train, test_size=self.test_size,
                                                                              random_state=self.random_state)

        print('=============Train subset===============')
        self.data_train['image'] = data_train
        self.data_train['depth'] = labels_train
        self._split_stats(data_train)

        print('=============Test subset===============')
        self.data_test['image'] = data_test
        self.data_test['depth'] = labels_test
        self._split_stats(data_test)

        if self.val:
            print('===========Validation subset============')
            self.data_val['image'] = data_val
            self.data_val['depth'] = labels_val
            self._split_stats(data_val)


class BeraDataset(Dataset):
    def __init__(self, img_filenames, depth_filenames):
        self.img_filenames = img_filenames
        self.depth_filenames = depth_filenames

    def __len__(self) -> int:
        return len(self.depth_filenames)

    def __getitem__(self, index) -> np.array:
        """Reads sample"""
        image = Image.open(self.img_filenames[index])
        print(self.depth_filenames[index])
        label = np.load(self.depth_filenames[index], allow_pickle=True)
        return {'image': image, 'label': label}


## Example usage
parsed = DatadirParser()
img, d = parsed.get_parsed()

spliter = TrainValTestSplitter(img, d)
train = spliter.data_train
test = spliter.data_test
val = spliter.data_val

ds = BeraDataset(train["image"], train["depth"])
item = ds[0]
