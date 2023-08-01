
import torch

from PIL import Image
import numpy as np
import scipy.io as scio
from torchvision import transforms
import h5py

def range_data(data):
    data = np.array(data).astype(np.int) - 1
    num_data = data.shape[0]
    return np.reshape(data, (num_data,))


DATA_PATH = 'E:/coco/coco_data/COCO.mat'
TEXT_PATH = 'E:/coco/coco_data/COCO_BoW.npy'

data = h5py.File(DATA_PATH)
# data.close()
FAll_tmp = data['FAll'] #存放图片名(1,87081)
IAll_tmp = data['IAll'] #存放原始特征(87081,3,224,224)
XAll_tmp = data['XAll'] #存放特征提取后的特征(4096,87081)
LAll_tmp = data['LAll'] #存放标签(91,87081)
param_tmp = data['param'] #存放 数据集划分索引


label_set = np.array(LAll_tmp).transpose()
txt_set = np.squeeze(np.load(TEXT_PATH))

param = {}
param['indexQuery'] = range_data(param_tmp['indexQuery'])
param['indexRetrieval'] = range_data(param_tmp['indexDatabase'])

test_index = param['indexQuery']
retrieval_index = range_data(param_tmp['indexDatabase'])

perm = np.random.permutation(retrieval_index.shape[0])
train_index = perm[:10000]


database_index = np.array([i for i in list(range(label_set.shape[0])) if i not in list(test_index)])


indexTest = test_index
indexDatabase = database_index
indexTrain = train_index

coco_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

coco_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

txt_feat_len = txt_set.shape[1]


class MSCOCO(torch.utils.data.Dataset):

    def __init__(self, transform=None, target_transform=None, train=True, database=False):
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.train_labels = label_set[indexTrain]
            self.train_index = indexTrain
            self.txt = txt_set[indexTrain]
        elif database:
            self.train_labels = label_set[indexDatabase]
            self.train_index = indexDatabase
            self.txt = txt_set[indexDatabase]
        else:
            self.train_labels = label_set[indexTest]
            self.train_index = indexTest
            self.txt = txt_set[indexTest]

    def __getitem__(self, index):

        # nuswide = h5py.File(IMG_DIR, 'r', libver='latest', swmr=True)
        img, target = IAll_tmp[self.train_index[index]], self.train_labels[index]
        img = Image.fromarray(np.transpose(img, (2, 1, 0)))
        # nuswide.close()

        txt = self.txt[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, txt, target, index

    def __len__(self):
        return len(self.train_labels)

