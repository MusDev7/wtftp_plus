import pywt
import torch

from module import *

class DomainTransformDecoderLayer(nn.Module):
    def __init__(self, scale_idx, time_len, wavelet, d_model, n_head, theta_liwt=0.01, theta_lwtLo=0.01,
                 theta_lwtHi=0.01, dropout=0.0, batch_first=True):
        super(DomainTransformDecoderLayer, self).__init__()
        self.scale_idx = scale_idx
        self.scaleAwareSelfAttention = ScaleAwareSelfAttentionUpdate(d_model, n_head, dropout, batch_first, temporal_mask=True)
        self.ln1s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(len(time_len))])
        self.learnableInverseWaveletTransforms = \
            nn.ModuleList([
                LearnableInverseWaveletTransform(n, k, wavelet, theta_liwt) for (n, k) in
                zip(time_len[:-1], time_len[1:])
            ])  # order from the low-level to the high-level
        self.ln2 = nn.LayerNorm(d_model)
        # self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1,
        #                       padding_mode='replicate')
        self.selfAttention = MHAttention(d_model=d_model, h=n_head, dropout=dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.learnableWaveletTransforms = \
            nn.ModuleList([
                LearnableWaveletTransform(n, k, wavelet, theta_lwtLo, theta_lwtHi) for (n, k) in
                zip(time_len[:-1], time_len[1:])
            ])  # order from the low-level to the high-level
        self.ln4s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(len(time_len))])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                          nn.Linear(d_model, d_model))
            for _ in range(len(time_len))
        ])
        self.ln5s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(len(time_len))])
        #==================BYPASS====================
        self.saBypass = MHAttention(n_head, d_model, dropout)
        self.lnBypass = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(len(scale_idx) - 1)
        ])

    def forward(self, x, prior_t):
        """
        x: B * T * D
        """
        normWLo_sum = torch.tensor(0.0, device=x.device)
        normWHi_sum = torch.tensor(0.0, device=x.device)
        normBLo_sum = torch.tensor(0.0, device=x.device)
        normBHi_sum = torch.tensor(0.0, device=x.device)
        normW_sum = torch.tensor(0.0, device=x.device)
        normB_sum = torch.tensor(0.0, device=x.device)

        # ================= Scale-Aware Self-Attention =================
        x_, weights_attn_sasa = self.scaleAwareSelfAttention(x, self.scale_idx)
        tmp = []
        for (i, j, ln) in zip(self.scale_idx[:-1], self.scale_idx[1:], self.ln1s):
            # x[:, i:j, :] = ln(x[:, i:j, :])
            tmp.append(ln(x_[:, i:j, :]+x[:, i:j, :]))
        x = torch.cat(tmp, dim=1)

        # ================= BYPASS =================
        statesBypass, _ = self.saBypass(x, prior_t, prior_t)
        coeff_set_bypass = []
        for (i, j, ln) in zip(self.scale_idx[:-1], self.scale_idx[1:], self.lnBypass):
            coeff_set_bypass.append(ln(statesBypass[:, i:j, :]+x[:, i:j, :]))
        # ================= Inverse Wavelet Transform =================
        coeff_set_toBeFused = []
        lo = x[:, self.scale_idx[-2]:self.scale_idx[-1], :]
        for i in range(len(self.scale_idx) - 2):
            hi = x[:, self.scale_idx[-(3 + i)]:self.scale_idx[-(2 + i)], :]
            coeff_set_toBeFused.append({"lo": lo, "hi": hi})
            lo, normW, normB = self.learnableInverseWaveletTransforms[::-1][i](lo, hi)
            normW_sum += normW
            normB_sum += normB
        coeff_set_toBeFused.append({"lo": lo, "hi": None})
        t = self.ln2(lo)
        # ================= Time-Domain Self-Attention =================
        states, weights_attn_sa = self.selfAttention(t, t, t)
        enhanced_t = self.ln3(states + t)
        # ===================== Wavelet transform =====================
        coeff_set = []
        enhanced_lo = enhanced_t
        for i in range(len(self.scale_idx) - 2):
            enhanced_lo, enhanced_hi, normWLo, normWHi, normBLo, normBHi = self.learnableWaveletTransforms[i]\
                (enhanced_lo+coeff_set_toBeFused[-i-1]["lo"])
            coeff_set.append(self.ln4s[i](enhanced_hi+coeff_set_toBeFused[-i-2]["hi"]))
            normWLo_sum += normWLo
            normWHi_sum += normWHi
            normBLo_sum += normBLo
            normBHi_sum += normBHi
        coeff_set.append(self.ln4s[-1](enhanced_lo+coeff_set_toBeFused[0]["lo"]))
        # ===================== FFN =====================
        for i, (ffn, ln) in enumerate(zip(self.ffns, self.ln5s)):
            coeff_set[i] = ln(ffn(coeff_set[i]+coeff_set_bypass[i])+coeff_set[i]+coeff_set_bypass[i])

        coeff_set = torch.cat(coeff_set, dim=1)
        return coeff_set, weights_attn_sasa, weights_attn_sa, torch.stack([normWLo_sum,
                                                                           normWHi_sum,
                                                                           normBLo_sum,
                                                                           normBHi_sum,
                                                                           normW_sum,
                                                                           normB_sum])


class ImprovedTransformerDecoder(nn.Module):
    def __init__(self, scale_idx, time_len, wavelet, d_input, d_output, d_model, n_head, n_layer, theta_liwt=0.01,
                 theta_lwtLo=0.01, theta_lwtHi=0.01, dropout=0.0, batch_first=True):
        super(ImprovedTransformerDecoder, self).__init__()
        self.scale_idx = scale_idx
        self.time_len = time_len
        self.embed_lo = nn.Sequential(nn.Linear(d_input, d_model // 2), nn.ReLU(),
                                      nn.Linear(d_model // 2, d_model))
        self.embedding_hi = nn.Parameter(
            nn.init.normal_(torch.empty(size=(1, len(time_len) - 1, d_model)), mean=0.02, std=0.1))
        self.positionEncoder = PositionalEncoding(d_model)
        self.scaleEncoder = ScaleEncoding(d_model)
        self.decoder = nn.ModuleList([
            DomainTransformDecoderLayer(scale_idx, time_len, wavelet, d_model, n_head, theta_liwt=theta_liwt,
                                        theta_lwtLo=theta_lwtLo, theta_lwtHi=theta_lwtHi, dropout=dropout,
                                        batch_first=batch_first)
            for _ in range(n_layer)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(),
                          nn.Linear(d_model // 2, d_output))
            for _ in range(len(time_len))
        ])

    def forward(self, x_lo, prior_t):
        B = x_lo.shape[0]
        x_lo = self.embed_lo(x_lo).repeat(1, self.scale_idx[-1]-self.scale_idx[-2], 1)
        x = []
        for i, l in enumerate(self.time_len[1:]):
            x.append(self.embedding_hi[:, i:i + 1, :].repeat(B, l, 1))
        x.append(x_lo)
        x = torch.cat(x, dim=1)
        x = self.positionEncoder(x)
        x = self.scaleEncoder(x, self.scale_idx)
        weights_attn_sasa_allLayer = []
        weights_attn_sa_allLayer = []
        coeff_set_allLayer = []
        norm_set = []
        coeff_set = x
        for decoder_layer in self.decoder:
            coeff_set, weights_attn_sasa, weights_attn_sa, norm = decoder_layer(coeff_set, prior_t)
            norm_set.append(norm)
            coeff_set_allLayer.append(coeff_set)
            weights_attn_sasa_allLayer.append(weights_attn_sasa)
            weights_attn_sa_allLayer.append(weights_attn_sa)
        tgt_coeff_set = coeff_set_allLayer[-1].clone()
        oup_coeff_set = []
        for (i, j, ffn) in zip(self.scale_idx[:-1], self.scale_idx[1:], self.ffns):
            oup_coeff_set.append(ffn(tgt_coeff_set[:, i:j, :]))

        oup_coeff_set = torch.cat(oup_coeff_set, dim=1)
        norm_set = torch.stack(norm_set, dim=0)
        return oup_coeff_set, coeff_set_allLayer, weights_attn_sasa_allLayer, weights_attn_sa_allLayer, norm_set


class WTFTP_plus(nn.Module):
    def __init__(self, args):
        super(WTFTP_plus, self).__init__()
        self.local = locals()
        self.encoder = nn.Sequential(nn.Sequential(nn.Linear(args.d_input, args.d_model // 2), nn.ReLU(),
                                                   nn.Linear(args.d_model // 2, args.d_model)),
                                     nn.TransformerEncoder(
                                         nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.n_head,
                                                                    dim_feedforward=args.d_model,
                                                                    dropout=args.dropout,
                                                                    batch_first=args.batch_first),
                                         num_layers=args.n_layer_encoder)
                                     )
        self.decoder = ImprovedTransformerDecoder(args.scale_idx_decoder, args.time_len_decoder, args.wavelet,
                                                args.d_input,
                                                args.d_output, args.d_model, args.n_head, args.n_layer_decoder,
                                                args.theta_liwt,
                                                args.theta_lwtLo, args.theta_lwtHi, args.dropout, args.batch_first,
                                                )

    def forward(self, x, lo_init):
        # prior_t, prior_t_allLayer, enhanced_t_allLayer, weights_attn_sasa_allLayerEncoder, weights_attn_sa_allLayerEncoder, norm_set_encoder= self.encoder(x)
        prior_t = self.encoder(x)
        tgt_coeff_set, coeff_set_allLayer, weights_attn_sasa_allLayerDecoder, weights_attn_sa_allLayerDecoder, norm_set_decoder\
            = self.decoder(
            lo_init, prior_t)
        return tgt_coeff_set, norm_set_decoder, (
        prior_t, coeff_set_allLayer, weights_attn_sasa_allLayerDecoder, weights_attn_sa_allLayerDecoder)


if __name__ == '__main__':
    from utils import generate_scales_idx, convert_WTCs

    wavelet = "db1"
    scales_idx_en, time_len_en = generate_scales_idx(9, 1, pywt.Wavelet(wavelet).dec_len / 2)
    scales_idx_de, time_len_de = generate_scales_idx(15, 2, pywt.Wavelet(wavelet).dec_len / 2)
    x = torch.rand((10, 9, 6))
    lo_init = torch.mean(x, dim=1, keepdim=True).repeat(1, time_len_de[-1], 1)


    class Args:
        def __init__(self):
            self.scale_idx_encoder = scales_idx_en
            self.time_len_encoder = time_len_en
            self.scale_idx_decoder = scales_idx_de
            self.time_len_decoder = time_len_de
            self.wavelet = wavelet
            self.d_input = 6
            self.d_output = 6
            self.d_model = 64
            self.n_head = 4
            self.n_layer_decoder = 2
            self.n_layer_encoder = 2
            self.theta_liwt = 0.01
            self.theta_lwtLo = 0.01
            self.theta_lwtHi = 0.01
            self.dropout = 0.
            self.batch_first = True


    args = Args()
    wtftp_plus = WTFTP_plus(args)
    tgt_coeff_set, norm_set_decoder, (
        prior_t, coeff_set_allLayer, weights_attn_sasa_allLayerDecoder, weights_attn_sa_allLayerDecoder) = wtftp_plus(x, lo_init)
    print("")
