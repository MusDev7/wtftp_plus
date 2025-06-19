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


def Direction(traj, threshold_heading=1, threshold_alt=10):
    """

    :param traj: B * T * D (in D, lon, lat and alt measured in degree, degree and meter located in the first 3 elements)
    :param threshold_heading:
    :param threshold_alt:
    :return:
    """
    e = 1e-6
    time_len = traj.size(-2)
    traj_middle = traj[..., time_len//2, :]
    vec_entire = traj[..., -1, :]-traj[..., 0, :]
    vec_middle = traj_middle-traj[..., 0, :]
    theta_entire = torch.arccos(vec_entire[...,0]/((vec_entire[...,0]**2+vec_entire[...,1]**2)**0.5+e))*180/torch.pi
    theta_middle = torch.arccos(vec_middle[...,0]/((vec_middle[...,0]**2+vec_middle[...,1]**2)**0.5+e))*180/torch.pi
    left = (theta_entire - theta_middle) >= threshold_heading
    right = (theta_entire - theta_middle) <= -threshold_heading
    direct = ~(left | right)
    heading = torch.stack([left, direct, right], dim=-1)
    ascend = vec_entire[..., 2] >= threshold_alt
    descend = vec_entire[..., 2] <= -threshold_alt
    maintain = ~(ascend | descend)
    alt = torch.stack([ascend, maintain, descend], dim=-1)
    return heading, alt


class convertIntention:
    def __init__(self, num_pitch, num_yaw, device='cpu'):
        assert num_pitch % 2 == 1 and num_yaw % 4 == 0
        self.num_pitch = num_pitch
        self.num_yaw = num_yaw
        self.device = device
        self.candidate_pitch_intention = None
        self.candidate_yaw_intention = None
        self.pitch_yaw_intention()

    def pitch_yaw_intention(self):
        candidate_pitch = torch.linspace(-math.pi / 2, math.pi / 2, self.num_pitch, device=self.device).float()  # range: (-pi/2, pi/2)
        candidate_pitch_intention = torch.zeros(self.num_pitch, 2, device=self.device).float()
        for i in range(self.num_pitch):
            candidate_pitch_intention[i, 0] = torch.cos(candidate_pitch[i])
            candidate_pitch_intention[i, 1] = torch.sin(candidate_pitch[i])
        self.candidate_pitch_intention = candidate_pitch_intention

        candidate_yaw = torch.linspace(0, 2 * math.pi - 2 * math.pi / self.num_yaw, self.num_yaw,
                                       device=self.device).float()  # range: (0, 2*pi)
        # print(candidate_yaw)
        candidate_yaw_intention = torch.zeros(self.num_yaw, 2, device=self.device).float()
        for i in range(self.num_yaw):
            candidate_yaw_intention[i, 0] = torch.cos(candidate_yaw[i])
            candidate_yaw_intention[i, 1] = torch.sin(candidate_yaw[i])
        self.candidate_yaw_intention = candidate_yaw_intention

    @torch.no_grad()
    def convert(self, inp):
        """
        Create intention from input
        :param inp: this tensor is the velocity (batch_size, 3), (spdx, spdy, spdz)
        :return:
        """
        pitch_vector = torch.zeros(inp.shape[0], 1, 2, device=self.device).float()
        length = torch.norm(inp, dim=1)
        pitch_vector[:, 0, 0] = torch.norm(inp[:, :2], dim=1)/length
        pitch_vector[:, 0, 1] = inp[:, 2]/length
        length = torch.norm(inp[:, :2], dim=1, keepdim=True).repeat(1, 2)
        yaw_vector = (inp[:, :2]/length).reshape(inp.shape[0], 1, 2)
        # print(pitch_vector)
        # print(yaw_vector)
        candidate_pitch_intention = self.candidate_pitch_intention.clone().reshape(1, self.num_pitch, 2).repeat(
            inp.shape[0], 1, 1)  # shape: (batch_size, num_pitch, 2)
        candidate_yaw_intention = self.candidate_yaw_intention.clone().reshape(1, self.num_yaw, 2).repeat(
            inp.shape[0], 1, 1)  # shape: (batch_size, num_yaw, 2)
        similarity_pitch = torch.bmm(pitch_vector, candidate_pitch_intention.permute(0, 2, 1)).squeeze(1)  # shape: (batch_size, num_intention)
        pitch_idx = torch.argmax(similarity_pitch, dim=1)
        similarity_yaw = torch.bmm(yaw_vector, candidate_yaw_intention.permute(0, 2, 1)).squeeze(1)  # shape: (batch_size, num_intention)
        yaw_idx = torch.argmax(similarity_yaw, dim=1)
        # print('similarity_pitch', similarity_pitch)
        # print('similarity_yaw', similarity_yaw)
        output = (pitch_idx * self.num_yaw + yaw_idx).long()
        return output, pitch_idx, yaw_idx

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


if __name__ == '__main__':
    inp = torch.randn(2, 3)
    print(inp)
    Intnt = convertIntention(num_pitch=5, num_yaw=8, device='cpu')
    print(Intnt.candidate_pitch_intention)
    print(Intnt.candidate_yaw_intention)
    print(Intnt.convert(inp))
