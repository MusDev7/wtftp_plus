import json
import logging
import os
import random
# import coordinate_conversion as cc
import numpy as np
import torch

import torch.utils.data as tu_data


class DataGenerator:
    def __init__(self, data_path, minibatch_len, interval=1,
                 train=True, test=True, dev=True, train_shuffle=True, test_shuffle=False, dev_shuffle=True, MAD_test=False, test_name="test", llzscore=False, scaled=True):
        assert os.path.exists(data_path)
        self.scaled = scaled
        self.attr_names = ['lon', 'lat', 'alt', 'spdx', 'spdy', 'spdz']
        self.llzscore = llzscore
        self.data_path = data_path
        self.interval = interval
        self.minibatch_len = minibatch_len
        self.data_range = np.load('data_range.npy', allow_pickle=True).item()
        assert type(self.data_range) is dict
        self.rng = random.Random(123)
        if train:
            self.train_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'train'), shuffle=train_shuffle))
        if dev:
            self.dev_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, test_name), shuffle=dev_shuffle))
        if test:
            if MAD_test:
                self.test_set = [mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'test_split/Maintain'), shuffle=test_shuffle)),
                                 mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'test_split/Ascend'), shuffle=test_shuffle)),
                                 mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'test_split/Descend'), shuffle=test_shuffle))]
            else:
                self.test_set = self.dev_set 

        print('data range:', self.data_range)

    def readtxt(self, data_path, shuffle=True):
        assert os.path.exists(data_path)
        data = []
        for root, dirs, file_names in os.walk(data_path):
            for file_name in file_names:
                if not file_name.endswith('txt'):
                    continue
                with open(os.path.join(root, file_name)) as file:
                    lines = file.readlines()
                    lines = lines[self.interval-1::self.interval]
                    if len(lines) == self.minibatch_len:
                        data.append(lines)
                    elif len(lines) < self.minibatch_len:
                        continue
                    else:
                        for i in range(len(lines)-self.minibatch_len+1):
                            data.append(lines[i:i+self.minibatch_len])
        print(f'{len(data)} items loaded from \'{data_path}\'')
        if shuffle:
            random.shuffle(data)
        return data

    def scale(self, inp, attr):
        assert type(attr) is str and attr in self.attr_names
        data_range = self.data_range
        if attr == "lon" and self.llzscore:
            inp = (inp-data_range[attr]['mu'])/data_range[attr]['sigma']
        elif attr == "lat" and self.llzscore:
            inp = (inp-data_range[attr]['mu'])/data_range[attr]['sigma']
        else:
            inp = (inp-data_range[attr]['min'])/(data_range[attr]['max']-data_range[attr]['min'])
        return inp

    def unscale(self, inp, attr):
        assert type(attr) is str and attr in self.attr_names
        data_range = self.data_range
        if attr == "lon" and self.llzscore:
            inp = inp*data_range[attr]['sigma']+data_range[attr]['mu']
        elif attr == "lat" and self.llzscore:
            inp = inp*data_range[attr]['sigma']+data_range[attr]['mu']
        else:
            inp = inp*(data_range[attr]['max']-data_range[attr]['min'])+data_range[attr]['min']
        return inp

    def collate(self, inp):
        '''
        :param inp: batch * n_sequence * n_attr
        :return:
        '''
        oup = []
        for minibatch in inp:
            tmp = []
            for line in minibatch:
                items = line.strip().split("|")
                lon, lat, alt, spdx, spdy, spdz = float(items[4]), float(items[5]), (float(items[6]) / 10), \
                                                  float(items[7]), float(items[8]), float(items[9])
                tmp.append([lon, lat, alt, spdx, spdy, spdz])
            minibatch = np.array(tmp)
            # print(minibatch.shape)
            if self.scaled:
                for i in range(minibatch.shape[-1]):
                    minibatch[:, i] = self.scale(minibatch[:, i], self.attr_names[i])
            oup.append(minibatch)
        return np.array(oup)


class mini_DataGenerator(tu_data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
