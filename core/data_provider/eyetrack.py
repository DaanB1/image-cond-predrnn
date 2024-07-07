__author__ = 'DaanB1'

import numpy as np
import os
import cv2
import logging
from torch.utils.data import Dataset, DataLoader
from torch import randint

logger = logging.getLogger(__name__)

class EyetrackDataset(Dataset):
    def __init__(self, input_param, vid_names):
        self.vid_names = vid_names
        self.root = input_param['paths'][0]
        self.image_width = input_param['image_width']
        self.seq_len = input_param['seq_length']
        self.vid_sizes = {}

        for video in self.vid_names:
            length = len(os.listdir(os.path.join(self.root, "Frame Data", video)))
            self.vid_sizes[video] = length

    def get_frame_heatmap_pair(self, video, idx):
        frame_path = os.path.join(self.root, "Frame Data", video, str(idx) + ".png")
        heatmap_path = os.path.join(self.root, "Heatmap Data", video, str(idx) + ".npy")
        frame_data = cv2.imread(frame_path)
        frame_data = frame_data.resize(self.image_width, self.image_width)
        frame_data = np.array(frame_data)
        heatmap_data = np.expand_dims(np.load(heatmap_path), axis=-1)
        return frame_data, heatmap_data

    def __len__(self):
        return sum(self.vid_sizes.values())

    # Returns frame data and heatmap data in shape [seq_len, channels, width, height]
    def __getitem__(self, idx):
        video_name = None
        for key, value in self.vid_sizes.items():
            if idx - value < 0:
                video_name = key
                break
            else:
                idx = idx - value

        #If unable to create full sequence, retry with anohter random index
        if idx - self.seq_len < 0:
            return self.__getitem__(randint(self.__len__(), [1])[0])

        seq_range = range(max(0, idx - self.seq_len), idx)
        frame_data = np.zeros((len(seq_range), self.image_width, self.image_width, 3))
        heatmap_data = np.zeros((len(seq_range), self.image_width, self.image_width, 1))
        for i, id in enumerate(seq_range):
            try:
                frame, heatmap = self.get_frame_heatmap_pair(video_name, id)
            except:
                return self.__getitem__(randint(self.__len__(), [1])[0])
            frame_data[i, :, :, :] = frame
            heatmap_data[i, :, :, :] = heatmap

        return np.concatenate((heatmap_data, frame_data), axis=-1)


class InputHandle:
    def __init__(self, video_names, input_param):
        self.dataset = EyetrackDataset(input_param, video_names)
        self.dataloader = DataLoader(self.dataset, input_param['minibatch_size'], shuffle=True)
        self.iter = iter(self.dataloader)
        self.payload = None
        self.hasnext = True

    def total(self):
        return len(self.dataset)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        self.iter = iter(self.dataloader)
        self.next()
        self.hasnext = True

    def next(self):
        value = next(self.iter, None)
        if value is None:
            self.hasnext = False
        self.payload = value

    def no_batch_left(self):
        return not self.hasnext

    def get_batch(self):
        return self.payload.numpy(force=True)

#Hardcoded train-test split
def get_train_input_handle(input_params):
    videos = ["AH SALES", "Allianz Direct", "Beterbed SALES-1", "Dacia Sandero", "DELTA"]
    return InputHandle(videos, input_params)

def get_test_input_handle(input_params):
    videos = ["DELA Hertest", "Bever SALES-1"]
    return InputHandle(videos, input_params)
