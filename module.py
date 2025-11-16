import torch.nn as nn
import torch
import math
import pywt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0).repeat(x.shape[0], 1, 1)
        return x


class ScaleEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(ScaleEncoding, self).__init__()
        self.scales = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)

    def forward(self, x, scales_idx):
        tmp = []
        for scale, (leftBound, rightBound) in enumerate(zip(scales_idx[:-1], scales_idx[1:])):
            tmp.append(x[:, leftBound:rightBound, :] + self.scales(torch.LongTensor([scale]).unsqueeze(0).
                                                                   repeat(x.shape[0], rightBound-leftBound).to(x.device)))
        return torch.cat(tmp, dim=1)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, score_adjust=None, dropout=None):
        scores = (torch.matmul(query, key.transpose(-2, -1)) + (score_adjust if score_adjust is not None else 0.)) \
                 / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        p_attn = torch.functional.F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MHAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, require_mean_attn=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.require_mean_attn = require_mean_attn

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, score_adjust=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, score_adjust=score_adjust, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        if self.require_mean_attn:
            attn = attn.mean(dim=1)

        return self.output_linear(x), attn


class ScaleAwareSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0, batch_first=True):
        super(ScaleAwareSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout,
                                          batch_first=batch_first)

    def _mask_generation(self, scales_idx):
        mask = torch.zeros((scales_idx[-1], scales_idx[-1]))
        leftBound = 0
        for scale, rightBound in enumerate(scales_idx):
            mask[leftBound:rightBound, leftBound:rightBound] = 1
            leftBound = rightBound
        mask = mask == 0
        return mask

    def forward(self, x, scales_idx):
        """

        :param scales_idx:
        :param x:
        :return:
        """
        mask = self._mask_generation(scales_idx=scales_idx).to(x.device)
        x_t, weights_attn = self.attn(x, x, x, attn_mask=mask)
        return x, weights_attn


class ScaleAwareSelfAttentionUpdate(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0, batch_first=True, temporal_mask=False):
        super(ScaleAwareSelfAttentionUpdate, self).__init__()
        self.temporal_mask = temporal_mask
        self.attn = MHAttention(d_model=d_model, h=n_head, dropout=dropout)

    def _mask_generation(self, scales_idx):
        mask = torch.zeros((scales_idx[-1], scales_idx[-1]))
        leftBound = 0
        for scale, rightBound in enumerate(scales_idx):
            mask[leftBound:rightBound, leftBound:rightBound] = \
                torch.tril(torch.ones(rightBound-leftBound,rightBound-leftBound)) if self.temporal_mask else 1
            leftBound = rightBound
        mask = mask == 0
        return mask

    def forward(self, x, scales_idx, score_adjust=None):
        """

        :param scales_idx:
        :param x:
        :return:
        """
        mask = self._mask_generation(scales_idx=scales_idx).to(x.device)
        x_t, weights_attn = self.attn(x, x, x, mask=mask, score_adjust=score_adjust)
        return x, weights_attn


class LearnableInverseWaveletTransform(nn.Module):
    def __init__(self, n, k, wavelet="haar", theta=0.01):
        super(LearnableInverseWaveletTransform, self).__init__()
        self.n = n
        self.k = k
        self.wavelet = wavelet
        self.theta = 1.0
        self.adaptiveParameters = nn.Parameter(nn.init.normal_(torch.empty(size=(2*self.k, self.n)), mean=0.02, std=0.1))
        self.bias = nn.Parameter(nn.init.normal_(torch.empty(size=(self.n,)), mean=0.02, std=0.1))
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU(1/5.5)
        self.transformMatrix, self.mask = self.buildTransformMatrix()
        self.transformMatrix = nn.Parameter(self.transformMatrix, requires_grad=False)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def buildTransformMatrix(self):
        W = torch.zeros(size=(2*self.k, self.n), dtype=torch.float)
        mask = torch.ones(size=(2*self.k, self.n), dtype=torch.float)
        h_coeff = pywt.Wavelet(self.wavelet).rec_lo
        g_coeff = pywt.Wavelet(self.wavelet).rec_hi
        filter_len = len(h_coeff)
        for i in range(self.k):
            lag = 2*i
            if lag < self.n:
                W[2*i, lag:min(lag+filter_len, self.n)] = torch.tensor(data=h_coeff[:min(lag+filter_len, self.n)-lag],
                                                                       dtype=torch.float)
                mask[2*i, lag:min(lag+filter_len, self.n)] = 0.
            W[2*i+1, max(0, lag+2-filter_len):min(lag+2, self.n)] = torch.tensor(
                data=g_coeff[-2-lag:filter_len-max(0, lag+2-self.n)], dtype=torch.float)
            mask[2*i+1, max(0, lag+2-filter_len):min(lag+2, self.n)] = 0.
        return W, mask

    def forward(self, lo, hi):
        """
        lo, hi: B * T(k) * D
        """
        B = lo.shape[0]
        D = lo.shape[2]
        c = torch.empty(size=(B, self.k*2, D), device=lo.device)
        c[:, list(range(0, self.k*2, 2)), :] = lo
        c[:, list(range(1, self.k*2, 2)), :] = hi
        a = self.leakyReLU(torch.bmm(c.transpose(1, 2),(self.transformMatrix + self.mask * self.theta *
                                                        self.adaptiveParameters).unsqueeze(0).repeat(c.shape[0],1,1)) + self.bias)
        return a.transpose(1,2), torch.norm(self.mask * self.adaptiveParameters, 2), torch.norm(self.bias, 2)  # B * T(n) * D



class LearnableWaveletTransform(nn.Module):
    def __init__(self, n, k, wavelet="haar", thetaLo=0.01, thetaHi=0.01):
        super(LearnableWaveletTransform, self).__init__()
        self.n = n
        self.k = k
        self.wavelet = wavelet
        self.thetaLo = 1.0
        self.thetaHi = 1.0
        self.adaptiveParametersLo = nn.Parameter(nn.init.normal_(torch.empty(size=(self.k, self.n)), mean=0.02, std=0.1))
        self.adaptiveParametersHi = nn.Parameter(nn.init.normal_(torch.empty(size=(self.k, self.n)), mean=0.02, std=0.1))
        self.biasLo = nn.Parameter(nn.init.normal_(torch.empty(size=(self.k,)), mean=0.02, std=0.1))
        self.biasHi = nn.Parameter(nn.init.normal_(torch.empty(size=(self.k,)), mean=0.02, std=0.1))
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU(1/5.5)
        (self.transformMatrixLo, self.transformMatrixHi), (self.maskLo, self.maskHi) = self.buildTransformMatrix()
        self.transformMatrixLo = nn.Parameter(self.transformMatrixLo, requires_grad=False)
        self.transformMatrixHi = nn.Parameter(self.transformMatrixHi, requires_grad=False)
        self.maskLo = nn.Parameter(self.maskLo, requires_grad=False)
        self.maskHi = nn.Parameter(self.maskHi, requires_grad=False)

    def buildTransformMatrix(self):
        W = torch.zeros(size=(2*self.k, self.n), dtype=torch.float)
        mask = torch.ones(size=(2*self.k, self.n), dtype=torch.float)
        h_coeff = pywt.Wavelet(self.wavelet).rec_lo
        g_coeff = pywt.Wavelet(self.wavelet).rec_hi
        filter_len = len(h_coeff)
        for i in range(self.k):
            lag = 2*i
            if lag < self.n:
                W[2*i, lag:min(lag+filter_len, self.n)] = torch.tensor(data=h_coeff[:min(lag+filter_len, self.n)-lag], dtype=torch.float)
                mask[2*i, lag:min(lag+filter_len, self.n)] = 0.
            W[2*i+1, max(0, lag+2-filter_len):min(lag+2, self.n)] = torch.tensor(data=g_coeff[-2-lag:filter_len-max(0, lag+2-self.n)],
                                                                                 dtype=torch.float)
            mask[2*i+1, max(0, lag+2-filter_len):min(lag+2, self.n)] = 0.
        WLo = W[list(range(0, 2*self.k, 2)), :]
        maskLo = mask[list(range(0, 2*self.k, 2)), :]
        WHi = W[list(range(1, 2*self.k, 2)), :]
        maskHi = mask[list(range(1, 2*self.k, 2)), :]
        return (WLo, WHi), (maskLo, maskHi)

    def forward(self, a):
        """
        a: B * T(n) * D
        """
        lo = self.leakyReLU(torch.bmm(a.transpose(1, 2),
                                      (self.transformMatrixLo + self.maskLo * self.thetaLo * self.adaptiveParametersLo)
                                      .transpose(0,1).unsqueeze(0).repeat(a.shape[0],1,1)) + self.biasLo)
        hi = self.leakyReLU(torch.bmm(a.transpose(1, 2),
                                      (self.transformMatrixHi + self.maskHi * self.thetaHi * self.adaptiveParametersHi)
                                      .transpose(0,1).unsqueeze(0).repeat(a.shape[0],1,1)) + self.biasHi)
        return lo.transpose(1,2), hi.transpose(1,2), \
               torch.norm(self.maskLo * self.adaptiveParametersLo, 2), torch.norm(self.maskHi * self.adaptiveParametersHi, 2), \
               torch.norm(self.biasLo, 2), torch.norm(self.biasHi, 2)  # B * T(k) * D



