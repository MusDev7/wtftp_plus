# coding=utf-8
import pywt

from model import *
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

try:
    from tensorboardX import SummaryWriter
except:
    pass
torch.autograd.set_detect_anomaly(True)


class Hpara(object):
    def __init__(self, wavelet, d_input, d_output, d_model, n_head, n_layer_decoder, n_layer_encoder,
                 dropout, batch_first, level_encoder, level_decoder, extendMode, lr,
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
    def __init__(self, opt):
        self.args = Hpara(**opt)
        self.SEED = None
        self.opt = opt
        self.device = torch.device(
            f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() and not self.args.cpu else 'cpu')
        self.iscuda = torch.cuda.is_available() and not self.args.cpu
        self.data_set = None
        if self.args.saving_model_num > 0:
            self.model_names = []
        if self.args.logging:
            self.log_path = self.args.logdir + f'/{datetime.datetime.now().strftime("%y-%m-%d-%H-%M")}' + \
                            ('' if len(self.args.comments) == 0 else ('-' + self.args.comments))
            self.TX_log_path = self.log_path + '/Tensorboard'
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            if os.path.exists(self.TX_log_path):
                shutil.rmtree(self.TX_log_path)
                os.mkdir(self.TX_log_path)
            log_name = 'train.log'
            logging.basicConfig(filename=os.path.join(self.log_path, log_name),
                                filemode=self.args.log_mode, format='%(asctime)s   %(message)s', level=logging.DEBUG)
            try:
                self.TX_logger = SummaryWriter(log_dir=self.log_path + '/Tensorboard')
            except:
                pass

    def run(self, is_training=True):
        if is_training:
            self.data_set = DataGenerator(data_path=self.args.data_path,
                                          minibatch_len=self.args.pre_len + self.args.his_len,
                                          interval=self.args.interval,
                                          llzscore=self.args.llZscore, scaled=True)
            self.train()

    def train(self):
        model = WTFTP_plus(self.args)
        print_str = f'parameter amount: {sum([p.numel() for p in model.parameters() if p.requires_grad])}'
        print(print_str)
        self.log(print_str)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        opt_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        self.MSE = nn.MSELoss(reduction='mean')
        self.MAE = nn.L1Loss(reduction='mean')
        self.CE = nn.CrossEntropyLoss(reduction='mean')
        # start training
        self.log(self.summary(model))
        model.to(self.device)
        for epoch in range(self.opt["epoch"]):
            model.train()
            train_data = DataLoader(dataset=self.data_set.train_set, batch_size=self.args.batch_size, shuffle=True,
                                    collate_fn=self.data_set.collate)
            start_time = time.perf_counter()
            batchs_len = len(train_data)
            loss_set = []
            high_loss_set = []
            low_loss_set = []
            mde_loss_set = []
            lon_aeloss_set = []
            lat_aeloss_set = []
            alt_aeloss_set = []
            weight_attn = {"sasa_en": [], "sa_en": [], "sasa_de": [], "sa_de": []}
            print('\n' + f'Epoch {epoch}/{self.args.epoch - 1}'.center(50, '-'))
            print(f'Description: {self.args.comments}')
            print(f'lr: {float(opt_lr_scheduler.get_last_lr()[0])}')
            if self.args.logging:
                logging.debug('\n' + f'Epoch {epoch}/{self.args.epoch - 1}'.center(50, '-'))
                logging.debug(f'Description: {self.args.comments}')
            for i, item in enumerate(train_data):
                item = torch.from_numpy(item).float().to(self.device)

                obs = item[:, :-self.args.pre_len, :]
                tgt = item[:, -self.args.pre_len:, :]

                # target lowWTCs, highWTCs, lo_init
                de = torch.arange(1, self.args.pre_len + 1).reshape(1, self.args.pre_len, 1). \
                    repeat(obs.shape[0], 1, obs.shape[-1]).to(obs.device)
                input_traj = (tgt - obs[:, -1:, :].repeat(1, self.args.pre_len, 1)) / de
                lowWTCs, highWTCs = convert_WTCs(input_traj.transpose(1, 2),
                                                 level=self.args.level_decoder, wavelet=self.args.wavelet,
                                                 mode=self.args.extendMode)

                lo_init = (obs[:, -1:, :] - obs[:, -2:-1, :]).repeat(1, self.args.time_len_decoder[-1], 1)

                # fed into the model
                tgt_coeff_set, norm_de, (
                    prior_t,
                    coeff_set_allLayer, weights_attn_sasa_allLayerDecoder, weights_attn_sa_allLayerDecoder) = model(obs,
                                                                                                                    lo_init)

                pred_lowWTCs = tgt_coeff_set[:, -self.args.time_len_decoder[-1]:, :].transpose(1, 2)  # B * A * T
                pred_highWTCs = tgt_coeff_set[:, :-self.args.time_len_decoder[-1], :].transpose(1, 2)  # B * A * T

                lowWTCs_loss, highWTCs_loss = self.compute_loss(lowWTCs, highWTCs, pred_lowWTCs, pred_highWTCs,
                                                                self.args.scale_idx_decoder)
                L2 = (epoch + 1) / (1 + self.opt["epoch"]) * 1e-4 * norm_de.mean(dim=0).sum()

                total_loss = lowWTCs_loss + highWTCs_loss + L2

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                print_str = f'Total Loss: {total_loss.item():.7f}, ' + \
                            f'Low-Freq Loss: {lowWTCs_loss.item():.7f}, ' + \
                            f'High-Freq Loss: {highWTCs_loss.item():.7f} ' + \
                            f'L2: {L2.item()*10000:.7f}e-4 '

                progress_bar(i, batchs_len, print_str, start_time)

                if self.args.logging:
                    if i % (len(train_data) // 15) == 0 or i == len(train_data) - 1:
                        self.log(print_str)
                    loss_set.append(total_loss.detach().cpu())
                    high_loss_set.append(highWTCs_loss.detach().cpu())
                    low_loss_set.append(lowWTCs_loss.detach().cpu())
            # lr scheduler
            opt_lr_scheduler.step()
            # dev stage
            dev_data = DataLoader(dataset=self.data_set.dev_set, batch_size=self.args.batch_size, shuffle=False,
                                  collate_fn=self.data_set.collate)

            model.eval()
            with torch.no_grad():
                output_set = []
                trg_set = []
                for i, item in enumerate(dev_data):
                    item = torch.from_numpy(item).float().to(self.device)

                    obs = item[:, :-self.args.pre_len, :]
                    tgt = item[:, -self.args.pre_len:, :]

                    # lo_init
                    lo_init = (obs[:, -1:, :] - obs[:, -2:-1, :]).repeat(1, self.args.time_len_decoder[-1], 1)

                    tgt_coeff_set, norm_de, (
                        prior_t,
                        coeff_set_allLayer, weights_attn_sasa_allLayerDecoder, weights_attn_sa_allLayerDecoder) = model(
                        obs, lo_init)

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

                    output_set.append(pred_traj)
                    trg_set.append(tgt)

                    weight_attn["sasa_de"].append(torch.stack(weights_attn_sasa_allLayerDecoder, dim=0))
                    weight_attn["sa_de"].append(torch.stack(weights_attn_sa_allLayerDecoder, dim=0))

                output_set = torch.cat(output_set, dim=0)
                trg_set = torch.cat(trg_set, dim=0)

                weight_attn["sasa_de"] = torch.cat(weight_attn["sasa_de"], dim=1)
                weight_attn["sa_de"] = torch.cat(weight_attn["sa_de"], dim=1)
                # cal loss
                dev_mse = self.MSE(trg_set, output_set)
                dev_mae = self.MAE(trg_set, output_set)

                loss_unscaled = {'rmse': {}, 'mae': {}}
                trg_cloned = trg_set.detach()
                pre_cloned = output_set.detach()
                test_step = [1, 3, 9, 15]
                print_str = ''
                for s in test_step:
                    if s > self.args.pre_len:
                        break
                    tgt_denorm = trg_cloned.clone()[:, :s, :]
                    pre_denorm = pre_cloned.clone()[:, :s, :]
                    for i, name in enumerate(self.data_set.attr_names):
                        tgt_denorm[..., i] = self.data_set.unscale(tgt_denorm[..., i], name)
                        pre_denorm[..., i] = self.data_set.unscale(pre_denorm[..., i], name)
                        # calculate the MAE and RMSE
                        loss_unscaled['rmse'][name] = math.sqrt(float(self.MSE(tgt_denorm[..., i], pre_denorm[..., i])))
                        loss_unscaled['mae'][name] = float(self.MAE(tgt_denorm[..., i], pre_denorm[..., i]))
                    # calculate the MDE
                    tgt_denorm = tgt_denorm.cpu().numpy()
                    pre_denorm = pre_denorm.cpu().numpy()
                    X, Y, Z = gc2ecef(pre_denorm[..., 0], pre_denorm[..., 1], pre_denorm[..., 2] / 100)
                    X_t, Y_t, Z_t = gc2ecef(tgt_denorm[..., 0], tgt_denorm[..., 1], tgt_denorm[..., 2] / 100)
                    MDE = np.mean(np.sqrt((X - X_t) ** 2 + (Y - Y_t) ** 2 + (Z - Z_t) ** 2))
                    
                    print_str += f'Evaluation-Stage Step {s}:\n' \
                                 f'aveMSE(scaled): {dev_mse:.10f}, in each attr(RMSE, unscaled): {loss_unscaled["rmse"]}\n' \
                                 f'aveMAE(scaled): {dev_mae:.10f}, in each attr(MAE, unscaled): {loss_unscaled["mae"]}\n' \
                                 f'\033[1;32;40m MDE(unscaled): {MDE:.10f} \033[0m\n'
                print(print_str)
                if self.args.logging:
                    self.log(print_str)
            # tensorboard log
            if self.args.logging:
                mde_loss_set.append(MDE)
                lon_aeloss_set.append(loss_unscaled['mae']['lon'])
                lat_aeloss_set.append(loss_unscaled['mae']['lat'])
                alt_aeloss_set.append(loss_unscaled['mae']['alt'])
                try:
                    self.TX_logger.add_scalar('train_loss', torch.tensor(loss_set).mean().numpy(), epoch + 1)
                    self.TX_logger.add_scalar('high_loss', torch.tensor(high_loss_set).mean().numpy(), epoch + 1)
                    self.TX_logger.add_scalar('low_loss', torch.tensor(low_loss_set).mean().numpy(), epoch + 1)
                    self.TX_logger.add_scalar('maxStep_val_MDE', torch.tensor(mde_loss_set).mean().numpy(), epoch + 1)
                    self.TX_logger.add_scalar('maxStep_val_lon_aeLoss', torch.tensor(lon_aeloss_set).mean().numpy(),
                                              epoch + 1)
                    self.TX_logger.add_scalar('maxStep_val_lat_aeLoss', torch.tensor(lat_aeloss_set).mean().numpy(),
                                              epoch + 1)
                    self.TX_logger.add_scalar('maxStep_val_alt_aeLoss', torch.tensor(alt_aeloss_set).mean().numpy(),
                                              epoch + 1)
                    
                    for layer, (sasa_de, sa_de) in enumerate(zip(weight_attn["sasa_de"], weight_attn["sa_de"])):
                        self.TX_logger.add_image(f'weights_attn_sasa_decoder_layer{layer}',
                                                 sasa_de.detach().cpu().mean(dim=0, keepdim=True).numpy(), epoch + 1)
                        self.TX_logger.add_image(f'weights_attn_sa_decoder_layer{layer}',
                                                 sa_de.detach().cpu().mean(dim=0, keepdim=True).numpy(), epoch + 1)
                except Exception as e:
                    self.log(e)
                    print(e)
            # save model
            self.saving_model(model, self.log_path, f'epoch_{epoch}.pt')

    def compute_loss(self, lowWTCs, highWTCs,
                     pred_lowWTCs, pred_highWTCs, scales_idx, highWTCs_lossWeights=None):
        """

        :param highWTCs_lossWeights:
        :param prior_mean: BatchSize * NScales * FeatureSize
        :param prior_logVar: ditto
        :param posterior_mean: ditto
        :param posterior_logVar: ditto
        :param lowWTCs:
        :param highWTCs:
        :param pred_lowWTCs:
        :param pred_highWTCs:
        :param scales_idx:
        :return:
        """
        if highWTCs_lossWeights is None:
            highWTCs_lossWeights = torch.ones(len(scales_idx) - 2).float()
        lowWTCs_loss = self.MSE(lowWTCs, pred_lowWTCs)

        highWTCs_loss = torch.tensor(0, device=highWTCs.device).float()

        for scale, (leftBound, rightBound) in enumerate(zip(scales_idx[:-2], scales_idx[1:-1])):
            highWTCs_loss = highWTCs_loss + highWTCs_lossWeights[scale] * \
                            self.MSE(pred_highWTCs[:, :, leftBound:rightBound], highWTCs[:, :, leftBound:rightBound])

        return lowWTCs_loss, highWTCs_loss

    def log(self, log_str):
        logging.debug(log_str)

    def summary(self, model):
        output = f'TRAIN DETAILS'.center(50, '-') + \
                 f'\ntraining on device {self.device} with the random seed being {self.SEED}\n'
        for arg in vars(self.args):
            output += f'{arg}: {getattr(self.args, arg)}\n'
        for arg in model.local:
            output += f'{arg}: {model.local[arg]}\n'
        # output += str(model.args)
        print(output)
        return output

    def saving_model(self, model, model_path, this_model_name):
        if self.opt["saving_model_num"] > 0:
            self.model_names.append(this_model_name)
            if len(self.model_names) > self.opt["saving_model_num"]:
                removed_model_name = self.model_names[0]
                del self.model_names[0]
                os.remove(os.path.join(model_path, removed_model_name))  # remove the oldest model
            torch.save(model, os.path.join(model_path, this_model_name))  # save the latest model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./optDir/opt.json', help='options file')
    opt_file = parser.parse_args()
    try:
        with open(opt_file.opt, 'r') as f:
            opt = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('No options file "opt_backup.json" found')
    if len(opt["comments"]) == 0:
        opt["comments"] = input("Enter the training description: ")
    train_client = Client(opt)
    train_client.run()
