import torch
import pywt
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import DataGenerator
from utils import *
import argparse
import json
import time
import datetime
import os
import logging
import shutil
import math
from sklearn.metrics import accuracy_score

try:
    from tensorboardX import SummaryWriter
except:
    pass
torch.autograd.set_detect_anomaly(True)


class Hpara(object):
    def __init__(self, wavelet, d_input, d_output, d_model, n_head, n_layer_decoder, n_layer_encoder, theta_liwt,
                 theta_lwtLo, theta_lwtHi, dropout, batch_first, level_encoder, level_decoder, extendMode, lr,
                 data_path, pre_len, his_len, interval, logdir, cpu, logging, log_mode,
                 saving_model_num, comments, epoch, batch_size, llZscore, *args, **kwargs):
        self.batch_size = batch_size
        self.epoch = epoch
        self.scale_idx_encoder, self.time_len_encoder = \
            generate_scales_idx(his_len, level_encoder, pywt.Wavelet(wavelet).dec_len // 2)
        self.scale_idx_decoder, self.time_len_decoder = \
            generate_scales_idx(pre_len, level_decoder, pywt.Wavelet(wavelet).dec_len // 2)
        self.wavelet = wavelet
        self.d_input = d_input
        self.d_output = d_output
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer_decoder = n_layer_decoder
        self.n_layer_encoder = n_layer_encoder
        self.theta_liwt = 1
        self.theta_lwtLo = 1
        self.theta_lwtHi = 1
        self.dropout = dropout
        self.batch_first = batch_first
        self.level_encoder = level_encoder
        self.level_decoder = level_decoder
        self.extendMode = extendMode
        self.lr = lr
        self.data_path = data_path
        self.pre_len = pre_len
        self.his_len = his_len
        self.interval = interval
        self.logdir = logdir
        self.cpu = cpu
        self.logging = logging
        self.log_mode = log_mode
        self.saving_model_num = saving_model_num
        self.comments = comments
        # additional zscore parameters
        self.llZscore = llZscore
        self.mean_lon = 107.48713366276365
        self.std_lon = 2.765318627809872
        self.mean_lat = 28.32332017253662
        self.std_lat = 2.7269063733767744

class Client:
    def __init__(self, model_path, client_args):
        self.SEED = None
        self.device = torch.device(
            f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() and client_args.cuda else 'cpu')
        self.iscuda = torch.cuda.is_available() and client_args.cuda
        self.model = torch.load(model_path, map_location=self.device)
        self.args = self.model.local["args"]
        self.data_set = None
        self.log_path = client_args.logdir
        self.client_args = client_args
        self.MAE = nn.L1Loss()
        self.MSE = nn.MSELoss()
        log_name = f'test{"(mad)" if self.client_args.mad else ""}_{self.args.comments}.log'
        logging.basicConfig(filename=os.path.join(self.log_path, log_name),
                            filemode="w", format='%(asctime)s   %(message)s', level=logging.DEBUG)

    def run(self):
        self.data_set = DataGenerator(data_path=self.args.data_path,
                                      minibatch_len=self.args.pre_len + self.args.his_len,
                                      interval=self.args.interval,
                                      llzscore=self.args.llZscore, scaled=True, dev_shuffle=False, train=False,
                                      test_name=self.client_args.test_name, MAD_test=self.client_args.mad)

        self.test(self.model)

    def dsp(self,deg=4):
        scal = 1
        lAmbda = 0.6
        L2 = 1e-6
        x = scal * torch.arange(-7, 8, 1, dtype=torch.float64)
        X = torch.stack([x ** degg for degg in range(deg, -1, -1)], dim=0).T
        W = torch.diag_embed(lAmbda ** torch.arange(0, self.args.pre_len, 1, dtype=torch.float64))
        mm = X.T @ W @ X + L2 * torch.diag_embed(torch.ones(deg + 1, dtype=torch.float64))
        # mm[mm < 1e-15] = 0.
        mminv = torch.linalg.inv(mm)
        W_dsp = X @ mminv @ X.T @ W
        W_dsp = W_dsp.numpy()
        return W_dsp

    @torch.no_grad()
    def test(self, model):
        if not self.client_args.mad:
            test_data = DataLoader(dataset=self.data_set.test_set, batch_size=self.args.batch_size, shuffle=False,
                                   collate_fn=self.data_set.collate)
        else:
            test_data = [DataLoader(dataset=self.data_set.test_set[i], batch_size=self.args.batch_size, shuffle=False,
                                    collate_fn=self.data_set.collate) for i in range(3)]
        model.eval()

        # import matplotlib.pyplot as plt
        if not self.client_args.mad:
            self.gene_pred(test_data, "AllStage")
        else:
            self.gene_pred(test_data[0], "Maintain")
            self.gene_pred(test_data[1], "Ascend")
            self.gene_pred(test_data[2], "Descend")


    def gene_pred(self, test_data, comments=""):
        model = self.model
        his_set = []
        output_set = []
        trg_set = []
        coeffLY_set = []
        from tqdm import tqdm
        for i, item in tqdm(enumerate(test_data), desc="testing"):
            item = torch.from_numpy(item).float().to(self.device)

            obs = item[:, :-self.args.pre_len, :]
            tgt = item[:, -self.args.pre_len:, :]


            de = torch.arange(1, self.args.pre_len + 1).reshape(1, self.args.pre_len, 1). \
            repeat(obs.shape[0], 1, obs.shape[-1]).to(obs.device)
            lo_init = (obs[:, -1:, :] - obs[:, -2:-1, :]).repeat(1, self.args.time_len_decoder[-1], 1)
            tgt_coeff_set, norm_de, (
                prior_t,
                coeff_set_allLayer, weights_attn_sasa_allLayerDecoder, weights_attn_sa_allLayerDecoder) = model(
                obs, lo_init)
            coeffLY_set.append(coeff_set_allLayer[-1])



            pred_lowWTCs = tgt_coeff_set[:, -self.args.time_len_decoder[-1]:, :].transpose(1, 2)  # B * A * T
            pred_highWTCs = tgt_coeff_set[:, :-self.args.time_len_decoder[-1], :].transpose(1, 2)  # B * A * T

            pred_traj = reconvert_WTCs(pred_lowWTCs, pred_highWTCs,
                                       self.args.time_len_decoder, wavelet=self.args.wavelet,
                                       mode=self.args.extendMode).transpose(1, 2)[:, :self.args.pre_len, :]

            # recover the traj

            de = torch.arange(1, self.args.pre_len + 1).reshape(1, self.args.pre_len, 1).repeat(
                    obs.shape[0], 1, obs.shape[-1]).to(obs.device)
            pred_traj *= de
            pred_traj += obs[:, -1:, :].repeat(1, self.args.pre_len, 1)
            his_set.append(obs)
            output_set.append(pred_traj)
            trg_set.append(tgt)

        his_set = torch.cat(his_set, dim=0)
        output_set = torch.cat(output_set, dim=0)
        trg_set = torch.cat(trg_set, dim=0)

        W_dsp_ll = self.dsp(4)
        W_dsp_a = W_dsp_ll
        # cal loss
        dsploss_unscaled = {'rmse': {}, 'mae': {}, 'mre':{}}
        trg_cloned = trg_set.detach()
        pre_cloned = output_set.detach()

        test_step = [1, 3, 6, 9, 15]
        print_str = f'Description: {self.args.comments}\n'
        for i, name in enumerate(self.data_set.attr_names):
            trg_cloned[..., i] = self.data_set.unscale(trg_cloned[..., i], name)
            pre_cloned[..., i] = self.data_set.unscale(pre_cloned[..., i], name)
        pre_dsp = pre_cloned.clone()
        pre_dsp[...,:2] = torch.from_numpy(W_dsp_ll).float().to(self.device)@pre_cloned[...,:2]
        pre_dsp[...,2:3] = torch.from_numpy(W_dsp_a).float().to(self.device)@pre_cloned[...,2:3]
        for s in test_step:
            if s > self.args.pre_len:
                break
            for i, name in enumerate(self.data_set.attr_names):
                non_zero = trg_cloned[:, :s, i] > 0
                dsploss_unscaled['rmse'][name] = math.sqrt(
                    float(self.MSE(trg_cloned[:, :s, i],pre_dsp[:, :s, i])))
                dsploss_unscaled['mae'][name] = float(self.MAE(trg_cloned[:, :s, i], pre_dsp[:, :s, i]))
                dsploss_unscaled['mre'][name] = 100*float((torch.mean(torch.abs(trg_cloned[:, :s, i]-pre_dsp[:, :s, i])[non_zero]/trg_cloned[:, :s, i][non_zero])).cpu())
                X_t, Y_t, Z_t = gc2ecef(trg_cloned.cpu().numpy()[:, :s, 0], trg_cloned.cpu().numpy()[:, :s, 1], trg_cloned.cpu().numpy()[:, :s, 2] / 100)
                X_dsp, Y_dsp, Z_dsp = gc2ecef(pre_dsp.cpu().numpy()[:, :s, 0], pre_dsp.cpu().numpy()[:, :s, 1], pre_dsp.cpu().numpy()[:, :s, 2] / 100)
                MDE_dsp = np.mean(np.sqrt((X_dsp - X_t) ** 2 + (Y_dsp - Y_t) ** 2 + (Z_dsp - Z_t) ** 2))
            print_str += f'({comments})Evaluation-Stage Step {s}:\n' \
                         f'({comments})aveMSE(scaled): in each attr(RMSE, unscaled): {dsploss_unscaled["rmse"]}\n' \
                         f'({comments})aveMAE(scaled): in each attr(MAE, unscaled): {dsploss_unscaled["mae"]}\n' \
                         f'({comments})in each attr(MRE, unscaled): {dsploss_unscaled["mre"]}\n' \
                         f'({comments})\033[1;32;40m MDE(unscaled): {MDE_dsp:.10f}  \033[0m\n'
        self.log(print_str)
        print(print_str)

        visualize = str(input("visualize? [Y/n]"))
        if visualize != 'n':
            import matplotlib.pyplot as plt
            sel = int(input('sel: '))
            while sel != -1:

                sel_traj = pre_cloned[sel, ...].detach().cpu().numpy()
                sel_trg_traj = trg_cloned[sel, ...].detach().cpu().numpy()
                sel_his_traj = his_set[sel, ...].detach().cpu().numpy()

                for j, name in enumerate(self.data_set.attr_names):
                    sel_his_traj[..., j] = self.data_set.unscale(sel_his_traj[..., j], name)

                def _draw(p_traj, title=str(sel)):
                    fig = plt.figure(figsize=(12, 10))
                    elev_azim_set = [[90, 0], [0, 0], [0, 90], [None,
                                                                None]]  # represent top view, lateral view(lat), lateral view(lon) and default, respectively
                    for n, elev_azim in enumerate(elev_azim_set):
                        ax = fig.add_subplot(2, 2, n + 1, projection='3d')
                        ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
                        ax.plot3D(sel_his_traj[..., 0], sel_his_traj[..., 1], sel_his_traj[..., 2], marker='o',
                                  markeredgecolor='dodgerblue',
                                  label='his')
                        ax.plot3D(sel_trg_traj[..., 0], sel_trg_traj[..., 1], sel_trg_traj[..., 2], marker='*',
                                  markeredgecolor='blueviolet',
                                  label='tgt')
                        ax.plot3D(p_traj[..., 0], p_traj[..., 1], p_traj[..., 2], marker='p',
                                  markeredgecolor='orangered',
                                  label='pre')
                        ax.set_xlabel('lon')
                        ax.set_ylabel('lat')
                        ax.set_zlabel('alt')
                        zmargin = 10
                        ax.set_zlim(min([min(it) for it in
                                         [sel_his_traj[..., 2], sel_trg_traj[..., 2], sel_traj[..., 2]]]) - zmargin,
                                    max([max(it) for it in
                                         [sel_his_traj[..., 2], sel_trg_traj[..., 2], sel_traj[..., 2]]]) + zmargin)
                        ax.legend()
                        plt.suptitle(title)

                poly_traj = pre_dsp[sel, ...].detach().cpu().numpy()
                _draw(poly_traj, f'{sel}')
                plt.show()
                sel = int(input('sel: '))

    def log(self, log_str):
        logging.debug(log_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='directory of pt files')
    parser.add_argument('--netdir', type=str, default="", help='exact model file')
    parser.add_argument('--cuda', action="store_true", help="using gpu")
    parser.add_argument('--test_name', type=str, default="test", help="test dir name")
    parser.add_argument('--mad', action="store_true", help="mad test")
    args = parser.parse_args()
    if len(args.netdir) == 0:
        pt_files = sorted([file for file in os.listdir(args.logdir) if file.endswith(".pt")])
        assert pt_files.__len__() != 0
        for n, pt in enumerate(pt_files):
            print(f"{n}: {pt}")
        try:
            n = int(input("Choose the model: "))
            n = n if 0 <= n <= 9 else -1
        except:
            n = -1
        model_path = os.path.join(args.logdir, pt_files[n])
    else:
        model_path = args.netdir

    test_client = Client(model_path, args)
    test_client.run()

