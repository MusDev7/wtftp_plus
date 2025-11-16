import torch
import math
import time
import numpy as np
from pytorch_wavelets import DWT1DForward, DWT1DInverse


def ecef2enu(ref_lon, ref_lat, ref_ecefX, ref_ecefY, ref_ecefZ, trg_ecefX, trg_ecefY, trg_ecefZ):
    dX = trg_ecefX - ref_ecefX
    dY = trg_ecefY - ref_ecefY
    dZ = trg_ecefZ - ref_ecefZ
    try:
        x = -np.sin(ref_lon) * dX + np.cos(ref_lon) * dY
        y = -np.sin(ref_lat)*np.cos(ref_lon) * dX - np.sin(ref_lat)*np.sin(ref_lon) * dY + np.cos(ref_lat) * dZ
        z = np.cos(ref_lat)*np.cos(ref_lon) * dX + np.cos(ref_lat)*np.sin(ref_lon) * dY + np.sin(ref_lat) * dZ
    except:
        x = -torch.sin(ref_lon) * dX + torch.cos(ref_lon) * dY
        y = -torch.sin(ref_lat) * torch.cos(ref_lon) * dX - torch.sin(ref_lat) * torch.sin(ref_lon) * dY + torch.cos(ref_lat) * dZ
        z = torch.cos(ref_lat) * torch.cos(ref_lon) * dX + torch.cos(ref_lat) * torch.sin(ref_lon) * dY + torch.sin(ref_lat) * dZ
    return x, y, z

def gc2ecef(lon, lat, alt):
    a = 6378.137 # km
    b = 6356.752
    e_square = 1 - (b ** 2) / (a ** 2)
    try:
        lat = np.radians(lat)
        lon = np.radians(lon)
        N = a / np.sqrt(1 - e_square*(np.sin(lat)**2))
        X = (N+alt) * np.cos(lat) * np.cos(lon)
        Y = (N+alt) * np.cos(lat) * np.sin(lon)
        Z = ((b**2)/(a**2) * N + alt) * np.sin(lat)
    except:
        lat = torch.deg2rad(lat)
        lon = torch.deg2rad(lon)
        N = a / torch.sqrt(1 - e_square * (torch.sin(lat) ** 2))
        X = (N + alt) * torch.cos(lat) * torch.cos(lon)
        Y = (N + alt) * torch.cos(lat) * torch.sin(lon)
        Z = ((b ** 2) / (a ** 2) * N + alt) * torch.sin(lat)
    return X, Y, Z

def generate_scales_idx(length, level, halfLenFilter):
    scales_idx = [0]
    time_len = [length]
    tmp = 0
    for _ in range(level):
        length = int(math.floor((length-1)/2) + halfLenFilter)
        time_len.append(length)
        tmp += length
        scales_idx.append(tmp)
    scales_idx.append(tmp+length)
    return scales_idx, time_len


def convert_WTCs(x, level, wavelet='haar', mode='symmetric'):
    """

    :param x: BatchSize * AttrSize * TimeSize
    :param level:
    :param mode:
    :return:
    """
    dwt = DWT1DForward(J=level, wave=wavelet, mode=mode).to(x.device)
    low, highs = dwt(x)
    return low, torch.cat(highs, dim=-1)


def reconvert_WTCs(low, highs, time_len, wavelet='haar', mode='symmetric'):
    """

    :param low: BatchSize * AttrSize * (ScaleSize @ TimeSize)
    :param highs: BatchSize * AttrSize * (ScaleSize @ TimeSize)
    :param scales_idx:
    :param wavelet:
    :param mode:
    :return:
    """
    idwt = DWT1DInverse(wave=wavelet, mode=mode).to(low.device)
    highs = torch.split(highs, time_len[1:], dim=-1)
    return idwt((low, highs))

def progress_bar(step, n_step, str, start_time=time.perf_counter(), bar_len=20):
    '''
    :param bar_len: length of the bar
    :param step: from 0 to n_step-1
    :param n_step: number of steps
    :param str: info to be printed
    :param start_time: time to begin the progress_bar
    :return:
    '''
    step = step+1
    a = "*" * int(step * bar_len / n_step)
    b = " " * (bar_len - int(step * bar_len / n_step))
    c = step / n_step * 100
    dur = time.perf_counter() - start_time
    print("\r{:^3.0f}%[{}{}]{:.2f}s {}".format(c, a, b, dur, str), end="")
    if step == n_step:
        print('')
