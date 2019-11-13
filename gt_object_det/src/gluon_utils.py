from __future__ import division

import os
import json
import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet.gluon import HybridBlock, nn, data


class GluoncvPreBlock(HybridBlock):
    r""" Extension of HybridBlock for Gluoncv model zoo pretrained models.

    Parameters
    ----------
    model_arch : str
        Base model architecture from model_zoo for GluonCV network.
        
    Example: 
    ----------
    import src.gluon_utils as guts
    
    net = guts.GluoncvPreBlock('yolo3_darknet53_coco')
    net.get_params('models/yolo/batch4epochs50/yolov3_best.params')
    image='fish.jpg'
    size=512
    cid, score, bbox, image = net.image_forward(image,size)
        
    """
    def __init__(self, model_arch, **kwargs):
        super(GluoncvPreBlock, self).__init__(**kwargs)
        self.net = gcv.model_zoo.get_model(model_arch, pretrained_base=False)
        
    """
    Parameters
    ---------
    model_params: str, Pretrained model parameters.  
    """
    def get_params(self, model_params):
        self.net.load_parameters(model_params)
    
    """
    Parameters
    ---------
    image_path: str, Path of image to push through network.
    image_size: int - Size to reshape image for inference.
    """
        
    def load_forward(self,image_path, image_size):
        x, image = gcv.data.transforms.presets.ssd.load_test(image_path, image_size)
        cid, score, bbox = self.net(x)
        return cid, score, bbox, image
    
    """
    Parameters
    ---------
    image_path: NDArray, NDArray of image.
    image_size: int - Size to reshape image for inference.
    """
        
    def transform_forward(self,image, image_size):
        x, image = gcv.data.transforms.presets.ssd.transform_test(image, image_size)
        cid, score, bbox = self.net(x)
        return cid, score, bbox, image


class GroundTruthDataset(data.Dataset):
    """
    Custom Dataset to handle the GroundTruth json file
    """
    def __init__(self,field_name, data_path='data'):
        """
        Parameters
        ---------
        data_path: str, Path to the data folder, default 'data'
        field_name: str, The annotation task name that appears in your json
                    the parent node of `annotations` that holds bbs infos

        """
        self.data_path = data_path
        self.field_name = field_name
        self.image_info = []
        with open(os.path.join(data_path, 'output.manifest')) as f:
            lines = f.readlines()
            for line in lines:
                info = json.loads(line[:-1])
                if len(info[field_name]['annotations']):
                    self.image_info.append(info)

    def __getitem__(self, idx):
        """
        Parameters
        ---------
        idx: int, index requested

        Returns
        -------
        image: nd.NDArray
            The image 
        label: np.NDArray bounding box labels of the form [[x1,y1, x2, y2, class], ...]
        """
        info = self.image_info[idx]
        image = mx.image.imread(os.path.join(data_dir,info['source-ref'].split('/')[-1]))
        boxes = info[field_name]['annotations']
        label = []
        for box in boxes:
            label.append([box['left'], box['top'], 
                box['left']+box['width'], box['top']+box['height'], 0])

        return image, np.array(label)

    def __len__(self):
        return len(self.image_info)

