import os
from base import BaseDataSet, BaseDataLoader
from utils import palette
import glob
import numpy as np
from PIL import Image
from skimage import io

class PhotovoltaicDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """

    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super().__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root.replace("/", '\\'), self.split)
        self.image_dir = os.path.join(self.root, 'images')
        self.label_dir = os.path.join(self.root, 'GT')
        if self.split == "train":
            file_list = glob.glob(os.path.join(self.image_dir, r'*.tif'))
        elif self.split == "val":
            file_list = glob.glob(os.path.join(self.image_dir, r'*.png'))
        else:
            raise ValueError(f'Invalidate value of split: {self.split}')
        self.files = [line.split(os.sep)[-1][:-4] for line in file_list]

    def _load_data(self, index):
        image_id = self.files[index]
        if self.split == "train":
            image_path = os.path.join(self.image_dir, image_id + '.tif')
        elif self.split == "val":
            image_path = os.path.join(self.image_dir, image_id + '.png')
        else:
            raise ValueError(f'Invalidate value of split: {self.split}')

        label_path = os.path.join(self.label_dir, image_id + '.png')
        # image = np.asarray(Image.open(image_path), dtype=np.float32)
        # label = np.asarray(Image.open(label_path), dtype=np.int32)
        image = np.asarray(io.imread(image_path), dtype=np.float32)
        label = np.asarray(io.imread(label_path)/255, dtype=np.int32)
        # image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class Photovoltaic(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False, shuffle=False, flip=False, rotate=False, blur=False, hsv_jitter=False, augment=False,
                 val_split=None, return_id=False):

        self.MEAN = (0.442729, 0.513404, 0.438429)
        self.STD = (0.117271, 0.103274, 0.092775)

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'hsv_jitter': hsv_jitter,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
        if split in ('train', 'val'):
            self.dataset = PhotovoltaicDataset(**kwargs)
        else:
            raise ValueError(f"Invalid split name {split}")
        super().__init__(self.dataset, batch_size, shuffle, num_workers, val_split)