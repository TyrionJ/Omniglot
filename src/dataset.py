import os
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

from src.config import data_root, train_file, valid_file


class OmniglotSet(Dataset):
    def __init__(self, T='train', trans=ToTensor()):
        creat_list()

        self.trans = trans
        self.inputs, self.labels = [], []
        self._cache = {}
        file = train_file if T == 'train' else valid_file
        with open(file, 'r') as f:
            self._img_files = [line.split('\n')[0] for line in f.readlines()]

    def __len__(self):
        return len(self._img_files)

    def __getitem__(self, index):
        if index not in self._cache:
            img = self._img_files[index]
            label = int(img.split('/')[-1].split('_')[0]) - 1
            self._cache[index] = (self.trans(Image.open(img)), label)

        return self._cache[index]


def creat_list():
    if not os.path.exists(train_file):
        import random
        train_list, test_list = [], []
        all_data = {}
        alphabets = os.listdir(f'{data_root}/images')
        for alphabet in alphabets:
            chars = os.listdir(f'{data_root}/images/{alphabet}')
            for char in chars:
                for sample in os.listdir(f'{data_root}/images/{alphabet}/{char}'):
                    label = int(sample.split('_')[0])
                    img = f'{data_root}/images/{alphabet}/{char}/{sample}'
                    if label in all_data:
                        all_data[label].append(img)
                    else:
                        all_data[label] = [img]

        for label in all_data:
            train, test = train_test_split(all_data[label], train_size=0.7)
            train_list.extend(train)
            test_list.extend(test)

        random.shuffle(train_list)
        random.shuffle(test_list)
        with open(train_file, 'w') as f:
            f.writelines('\n'.join(train_list))
        with open(valid_file, 'w') as f:
            f.writelines('\n'.join(test_list))
