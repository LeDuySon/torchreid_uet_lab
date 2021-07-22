from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class UET_REID(ImageDataset):
    dataset_dir = 'uet_reid'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        #data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        #if osp.isdir(data_dir):
        #    self.data_dir = data_dir
        #else:
        #    warnings.warn(
        #        'The current data structure is deprecated. Please '
        #        'put data folders such as "bounding_box_train" under '
        #        '"Market-1501-v15.09.15".'
        #    )
        #print("data dir: ", self.dataset_dir)
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')
        
        #print(self.train_dir)
        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, camid=0, relabel=True)
        query = self.process_dir(self.query_dir, camid=0, relabel=False)
        gallery = self.process_dir(self.gallery_dir, camid=1, relabel=False)
        
        # just for testing 
        self.query = query
        print(query)
        #print(train[0])
        #print(gallery[0])
        super(UET_REID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, camid=0, relabel=False):
        id_paths = glob.glob(dir_path + "/*")
        #print("id_paths: ", len(id_paths))
        data = []
        for pid in id_paths:
            ids = int(pid.split("/")[-1]) - 1 # idx start from zero but mot start from 1
            img_paths = glob.glob(pid + "/*.jpg")
            for img_path in img_paths:
                data.append((img_path, ids, camid))
            
        return data

if __name__ == '__main__':
    test = UET_REID("reid-data")
    print(test.query)
