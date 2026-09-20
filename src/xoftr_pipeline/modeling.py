"""XoFTR (Tuzcuoğlu, Köksal, Sofu, Kalkan and Alatan, CVPRW 2024) inference network, vendored from
https://github.com/OnderT/XoFTR at commit e0fbea431b30be9742effbf5577c90aa8eb938f9 (Apache-2.0):
``src/xoftr/backbone/resnet.py``, ``src/xoftr/utils/position_encoding.py`` and ``src/xoftr/xoftr_module/*``
concatenated in dependency order with the package-relative imports removed, plus the inference configuration
from ``src/config/default.py`` (``get_cfg_defaults(inference=True)`` lowered to a plain dict, as
image-matching-models builds it). No training utilities, datasets, Lightning or kornia code is carried; the
six ``einops.rearrange`` patterns upstream uses are provided by a local ``rearrange`` shim (plain ``reshape`` /
``permute``), so torch is the only dependency.

The state-dict key layout of the Hub checkpoints (``vismatch/xoftr``, formerly ``image-matching-models/xoftr``)
matches this module exactly: ``backbone.*``, ``pos_encoding.*``, ``loftr_coarse.*``, ``coarse_matching.*``,
``fine_process.*``, ``fine_matching.*``.
"""
# ruff: noqa: E501, N801, N802, N803, N806, E741, B905, F841, E712  -- vendored code kept as upstream wrote it, for auditability

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Module


def rearrange(tensor: torch.Tensor, pattern: str, **axes: int) -> torch.Tensor:
    """The six ``einops.rearrange`` patterns the upstream modules use, as plain ``view`` / ``permute`` so the
    vendored code reads exactly as upstream without an einops dependency. Any other pattern is refused."""
    if pattern == "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c":
        return tensor.reshape(tensor.shape[0], axes["h0c"], axes["w0c"], axes["h1c"], axes["w1c"])
    if pattern == "b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)":
        return tensor.reshape(tensor.shape[0], axes["h0c"] * axes["w0c"], axes["h1c"] * axes["w1c"])
    if pattern == "n (h w) c -> n c h w":
        n, _, c = tensor.shape
        return tensor.reshape(n, axes["h"], axes["w"], c).permute(0, 3, 1, 2)
    if pattern == "n c h w -> n (h w) 1 c":
        return tensor.flatten(2).permute(0, 2, 1).unsqueeze(2)
    if pattern == "n (c ww) l -> n l ww c":
        n, cww, l = tensor.shape
        return tensor.reshape(n, cww // axes["ww"], axes["ww"], l).permute(0, 3, 2, 1)
    if pattern == "n c h w -> n (h w) c":
        return tensor.flatten(2).permute(0, 2, 1)
    raise ValueError(f"unsupported rearrange pattern: {pattern!r}")

UPSTREAM_REPOSITORY = "https://github.com/OnderT/XoFTR"
UPSTREAM_COMMIT = "e0fbea431b30be9742effbf5577c90aa8eb938f9"
RESOLUTION = (8, 2)


def default_config(*, coarse_thr: float = 0.3, fine_thr: float = 0.1, denser: bool = False) -> dict:
    """``get_cfg_defaults(inference=True)`` -> ``lower_config`` -> ``["xoftr"]`` from upstream ``src/config/default.py``,
    with the three knobs image-matching-models exposes."""
    return {
        "resolution": RESOLUTION,
        "fine_window_size": 5,
        "medium_window_size": 3,
        "resnet": {"initial_dim": 128, "block_dims": [128, 196, 256]},
        "coarse": {
            "inference": True,
            "d_model": 256,
            "d_ffn": 256,
            "nhead": 8,
            "layer_names": ["self", "cross"] * 4,
            "attention": "linear",
        },
        "match_coarse": {
            "inference": True,
            "d_model": 256,
            "thr": coarse_thr,
            "border_rm": 2,
            "match_type": "dual_softmax",
            "dsmax_temperature": 0.1,
            "train_coarse_percent": 0.2,
            "train_pad_num_gt_min": 200,
        },
        "fine": {
            "denser": denser,
            "inference": True,
            "dsmax_temperature": 0.1,
            "thr": fine_thr,
            "mlp_hidden_dim_coef": 2,
            "nhead_fine_level": 8,
            "nhead_medium_level": 7,
        },
        "loss": {
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "pos_weight": 1.0,
            "neg_weight": 1.0,
            "coarse_weight": 0.5,
            "fine_weight": 0.3,
            "sub_weight": 1 * 10**4,
        },
    }



# ----------------------------------------------------------------------------------------------------
# upstream src/xoftr/backbone/resnet.py
# ----------------------------------------------------------------------------------------------------

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class ResNet_8_2(nn.Module):
    """
    ResNet, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        x3_out = self.layer3_outconv(x3)

        return x3_out, x2, x1


# ----------------------------------------------------------------------------------------------------
# upstream src/xoftr/utils/position_encoding.py
# ----------------------------------------------------------------------------------------------------

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))

        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


# ----------------------------------------------------------------------------------------------------
# upstream src/xoftr/xoftr_module/linear_attention.py
# ----------------------------------------------------------------------------------------------------

"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""



def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


# ----------------------------------------------------------------------------------------------------
# upstream src/xoftr/xoftr_module/transformer.py
# ----------------------------------------------------------------------------------------------------

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super().__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


# ----------------------------------------------------------------------------------------------------
# upstream src/xoftr/xoftr_module/coarse_matching.py
# ----------------------------------------------------------------------------------------------------

INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        d_model = config['d_model']
        self.thr = config['thr']
        self.inference = config['inference']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level XoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']
        self.final_proj = nn.Linear(d_model, d_model, bias=True)

        self.temperature = config['dsmax_temperature']

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """

        feat_c0 = self.final_proj(feat_c0)
        feat_c1 = self.final_proj(feat_c1)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                    feat_c1) / self.temperature
        if mask_c0 is not None:
            sim_matrix.masked_fill_(
                ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                -INF)
        if self.inference:
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match_inference(sim_matrix, data))
        else:
            conf_matrix_0_to_1 = F.softmax(sim_matrix, 2)
            conf_matrix_1_to_0 = F.softmax(sim_matrix, 1)
            data.update({'conf_matrix_0_to_1': conf_matrix_0_to_1,
                        'conf_matrix_1_to_0': conf_matrix_1_to_0
                        })
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match_training(conf_matrix_0_to_1, conf_matrix_1_to_0, data))

    @torch.no_grad()
    def get_coarse_match_training(self, conf_matrix_0_to_1, conf_matrix_1_to_0, data):
        """
        Args:
            conf_matrix_0_to_1 (torch.Tensor): [N, L, S]
            conf_matrix_1_to_0 (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix_0_to_1.device

        # confidence thresholding
        # {(nearest neighbour for 0 to 1) U (nearest neighbour for 1 to 0)}
        mask = torch.logical_or((conf_matrix_0_to_1 > self.thr) * (conf_matrix_0_to_1 == conf_matrix_0_to_1.max(dim=2, keepdim=True)[0]),
                               (conf_matrix_1_to_0 > self.thr) * (conf_matrix_1_to_0 == conf_matrix_1_to_0.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        mconf = torch.maximum(conf_matrix_0_to_1[b_ids, i_ids, j_ids], conf_matrix_1_to_0[b_ids, i_ids, j_ids])

        # random sampling of training samples for fine-level XoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # the sampling is performed across all pairs in a batch without manually balancing
            # samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min, ),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                    len(data['spv_b_ids']),
                    (max(num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min), ),
                    device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches

    @torch.no_grad()
    def get_coarse_match_inference(self, sim_matrix, data):
        """
        Args:
            sim_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }

        # softmax for 0 to 1
        conf_matrix_ = F.softmax(sim_matrix, 2)

        # confidence thresholding and nearest neighbour for 0 to 1
        mask = (conf_matrix_ > self.thr) * (conf_matrix_ == conf_matrix_.max(dim=2, keepdim=True)[0])

        # unlike training, reuse the same conf martix to decrease the vram consumption
        # softmax for 0 to 1
        conf_matrix_ = F.softmax(sim_matrix, 1)

        # update mask {(nearest neighbour for 0 to 1) U (nearest neighbour for 1 to 0)}
        mask = torch.logical_or(mask,
                                 (conf_matrix_ > self.thr) * (conf_matrix_ == conf_matrix_.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                    **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        # mconf = torch.maximum(conf_matrix_0_to_1[b_ids, i_ids, j_ids], conf_matrix_1_to_0[b_ids, i_ids, j_ids])

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches are the current coarse level predictions
        coarse_matches.update({
            'm_bids': b_ids,  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c,
            'mkpts1_c': mkpts1_c,
        })

        return coarse_matches


# ----------------------------------------------------------------------------------------------------
# upstream src/xoftr/xoftr_module/fine_process.py
# ----------------------------------------------------------------------------------------------------

class Mlp(nn.Module):
    """Multi-Layer Perceptron (MLP)"""

    def __init__(self,
                 in_dim,
                 hidden_dim=None,
                 out_dim=None,
                 act_layer=nn.GELU):
        """
        Args:
            in_dim: input features dimension
            hidden_dim: hidden features dimension
            out_dim: output features dimension
            act_layer: activation function
        """
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.view(*x_size[:-1], self.out_dim)
        return x


class VanillaAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 proj_bias=False):
        super().__init__()
        """
        Args:
            dim: feature dimension
            num_heads: number of attention head
            proj_bias: bool use query, key, value bias
        """
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_temp = self.head_dim ** -0.5
        self.kv_proj = nn.Linear(dim, dim * 2, bias=proj_bias)
        self.q_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.merge = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv=None):
        """
        Args:
            x_q (torch.Tensor): [N, L, C]
            x_kv (torch.Tensor): [N, S, C]
        """
        if x_kv is None:
            x_kv = x_q
        bs, _, dim = x_q.shape
        bs, _, dim = x_kv.shape
        # [N, S, 2, H, D] => [2, N, H, S, D]
        kv = self.kv_proj(x_kv).reshape(bs, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # [N, L, H, D] => [N, H, L, D]
        q = self.q_proj(x_q).reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = kv[0].transpose(-2, -1).contiguous(), kv[1].contiguous() # [N, H, D, S], [N, H, S, D]
        attn = (q @ k) * self.softmax_temp # [N, H, L, S]
        attn = attn.softmax(dim=-1)
        x_q = (attn @ v).transpose(1, 2).reshape(bs, -1, dim)
        x_q = self.merge(x_q)
        return x_q


class CrossBidirectionalAttention(nn.Module):
    def __init__(self, dim, num_heads, proj_bias = False):
        super().__init__()
        """
        Args:
            dim: feature dimension
            num_heads: number of attention head
            proj_bias: bool use query, key, value bias
        """

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_temp = self.head_dim ** -0.5
        self.qk_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.v_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.merge = nn.Linear(dim, dim, bias=proj_bias)
        self.temperature = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        # print(self.temperature)

    def map_(self, func, x0, x1):
        return func(x0), func(x1)

    def forward(self, x0, x1):
        """
        Args:
            x0 (torch.Tensor): [N, L, C]
            x1 (torch.Tensor): [N, S, C]
        """
        bs = x0.size(0)

        qk0, qk1 = self.map_(self.qk_proj, x0, x1)
        v0, v1 = self.map_(self.v_proj, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous(),
            (qk0, qk1, v0, v1))

        qk0, qk1 = qk0 * self.softmax_temp**0.5, qk1 * self.softmax_temp**0.5
        sim = qk0 @ qk1.transpose(-2,-1).contiguous()
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        x0 = attn01 @ v1
        x1 = attn10 @ v0
        x0, x1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2),
                        x0, x1)
        x0, x1 = self.map_(self.merge, x0, x1)

        return x0, x1


class SwinPosEmbMLP(nn.Module):
    def __init__(self,
                 dim):
        super().__init__()
        self.pos_embed = None
        self.pos_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(512, dim, bias=False))

    def forward(self, x):
        seq_length = x.shape[1]
        if self.pos_embed is None or self.training:
            seq_length = int(seq_length**0.5)
            coords = torch.arange(0, seq_length, device=x.device, dtype = x.dtype)
            grid = torch.stack(torch.meshgrid([coords, coords])).contiguous().unsqueeze(0)
            grid -= seq_length // 2
            grid /= (seq_length // 2)
            self.pos_embed = self.pos_mlp(grid.flatten(2).transpose(1,2))
        x = x + self.pos_embed
        return x


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_coef, use_pre_pos_embed=False):
        super().__init__()
        self.mlp = Mlp(in_dim=dim*2, hidden_dim=dim*mlp_hidden_coef, out_dim=dim, act_layer=nn.GELU)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = VanillaAttention(dim, num_heads=num_heads)
        self.pos_embed = SwinPosEmbMLP(dim)
        self.pos_embed_pre = SwinPosEmbMLP(dim) if use_pre_pos_embed else nn.Identity()

    def forward(self, x, x_pre):
        ww = x.shape[1]
        ww_pre = x_pre.shape[1]
        x = self.pos_embed(x)
        x_pre = self.pos_embed_pre(x_pre)
        x = torch.cat((x, x_pre), dim=1)
        x = x + self.gamma*self.norm1(self.mlp(torch.cat([x, self.attn(self.norm2(x))], dim=-1)))
        x, x_pre = x.split([ww, ww_pre], dim=1)
        return x, x_pre


class WindowCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_coef):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_dim=dim*2, hidden_dim=dim*mlp_hidden_coef, out_dim=dim, act_layer=nn.GELU)
        self.cross_attn = CrossBidirectionalAttention(dim, num_heads=num_heads, proj_bias=False)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x0, x1):
        m_x0, m_x1 = self.cross_attn(self.norm1(x0), self.norm1(x1))
        x0 = x0 + self.gamma*self.norm2(self.mlp(torch.cat([x0, m_x0], dim=-1)))
        x1 = x1 + self.gamma*self.norm2(self.mlp(torch.cat([x1, m_x1], dim=-1)))
        return x0, x1


class FineProcess(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        block_dims = config['resnet']['block_dims']
        self.block_dims = block_dims
        self.W_f = config['fine_window_size']
        self.W_m = config['medium_window_size']
        nhead_f = config["fine"]['nhead_fine_level']
        nhead_m = config["fine"]['nhead_medium_level']
        mlp_hidden_coef = config["fine"]['mlp_hidden_dim_coef']

        # Networks
        self.conv_merge = nn.Sequential(nn.Conv2d(block_dims[2]*2, block_dims[1], kernel_size=1, stride=1, padding=0, bias=False),
                                        nn.Conv2d(block_dims[1], block_dims[1], kernel_size=3, stride=1, padding=1, groups=block_dims[1], bias=False),
                                        nn.BatchNorm2d(block_dims[1])
                                        )
        self.out_conv_m = nn.Conv2d(block_dims[1], block_dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.out_conv_f = nn.Conv2d(block_dims[0], block_dims[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.self_attn_m = WindowSelfAttention(block_dims[1], num_heads=nhead_m,
                                                mlp_hidden_coef=mlp_hidden_coef, use_pre_pos_embed=False)
        self.cross_attn_m = WindowCrossAttention(block_dims[1], num_heads=nhead_m,
                                                  mlp_hidden_coef=mlp_hidden_coef)
        self.self_attn_f = WindowSelfAttention(block_dims[0], num_heads=nhead_f,
                                                mlp_hidden_coef=mlp_hidden_coef, use_pre_pos_embed=True)
        self.cross_attn_f = WindowCrossAttention(block_dims[0], num_heads=nhead_f,
                                                  mlp_hidden_coef=mlp_hidden_coef)
        self.down_proj_m_f = nn.Linear(block_dims[1], block_dims[0], bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def pre_process(self, feat_f0, feat_f1, feat_m0, feat_m1, feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data):
        W_f = self.W_f
        W_m = self.W_m
        data.update({'W_f': W_f,
                'W_m': W_m})

        # merge coarse features before and after loftr layer, and down proj channel dimesions
        feat_c0 = rearrange(feat_c0, 'n (h w) c -> n c h w', h =data["hw0_c"][0], w =data["hw0_c"][1])
        feat_c1 = rearrange(feat_c1, 'n (h w) c -> n c h w', h =data["hw1_c"][0], w =data["hw1_c"][1])
        feat_c0 = self.conv_merge(torch.cat([feat_c0, feat_c0_pre], dim=1))
        feat_c1 = self.conv_merge(torch.cat([feat_c1, feat_c1_pre], dim=1))
        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) 1 c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) 1 c')

        stride_f = data['hw0_f'][0] // data['hw0_c'][0]
        stride_m = data['hw0_m'][0] // data['hw0_c'][0]

        if feat_m0.shape[2] == feat_m1.shape[2] and feat_m0.shape[3] == feat_m1.shape[3]:
            feat_m = self.out_conv_m(torch.cat([feat_m0, feat_m1], dim=0))
            feat_m0, feat_m1 = torch.chunk(feat_m, 2, dim=0)
            feat_f = self.out_conv_f(torch.cat([feat_f0, feat_f1], dim=0))
            feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        else:
            feat_m0 = self.out_conv_m(feat_m0)
            feat_m1 = self.out_conv_m(feat_m1)
            feat_f0 = self.out_conv_f(feat_f0)
            feat_f1 = self.out_conv_f(feat_f1)

        # 1. unfold (crop windows) all local windows
        feat_m0_unfold = F.unfold(feat_m0, kernel_size=(W_m, W_m), stride=stride_m, padding=W_m//2)
        feat_m0_unfold = rearrange(feat_m0_unfold, 'n (c ww) l -> n l ww c', ww=W_m**2)
        feat_m1_unfold = F.unfold(feat_m1, kernel_size=(W_m, W_m), stride=stride_m, padding=W_m//2)
        feat_m1_unfold = rearrange(feat_m1_unfold, 'n (c ww) l -> n l ww c', ww=W_m**2)

        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W_f**2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W_f**2)

        # 2. select only the predicted matches
        feat_c0 = feat_c0[data['b_ids'], data['i_ids']] # [n, ww, cm]
        feat_c1 = feat_c1[data['b_ids'], data['j_ids']]

        feat_m0_unfold = feat_m0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cm]
        feat_m1_unfold = feat_m1_unfold[data['b_ids'], data['j_ids']]

        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

        return feat_c0, feat_c1, feat_m0_unfold, feat_m1_unfold, feat_f0_unfold, feat_f1_unfold

    def forward(self, feat_f0, feat_f1, feat_m0, feat_m1, feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data):
        """
        Args:
            feat_f0 (torch.Tensor): [N, C, H, W]
            feat_f1 (torch.Tensor): [N, C, H, W]
            feat_m0 (torch.Tensor): [N, C, H, W]
            feat_m1 (torch.Tensor): [N, C, H, W]
            feat_c0 (torch.Tensor): [N, L, C]
            feat_c1 (torch.Tensor): [N, S, C]
            feat_c0_pre (torch.Tensor): [N, C, H, W]
            feat_c1_pre (torch.Tensor): [N, C, H, W]
            data (dict): with keys ['hw0_c', 'hw1_c', 'hw0_m', 'hw1_m', 'hw0_f', 'hw1_f', 'b_ids', 'j_ids']
        """

        # upstream note: "Check for this case" (kept as written; a to-do marker in the vendored source)
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W_f**2, self.block_dims[0], device=feat_f0.device)
            feat1 = torch.empty(0, self.W_f**2, self.block_dims[0], device=feat_f0.device)
            return feat0, feat1

        feat_c0, feat_c1, feat_m0_unfold, feat_m1_unfold, \
            feat_f0_unfold, feat_f1_unfold = self.pre_process(feat_f0, feat_f1, feat_m0, feat_m1,
                                                               feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data)

        # self attention (c + m)
        feat_m_unfold, _ = self.self_attn_m(torch.cat([feat_m0_unfold, feat_m1_unfold], dim=0),
                                                         torch.cat([feat_c0, feat_c1], dim=0))
        feat_m0_unfold, feat_m1_unfold = torch.chunk(feat_m_unfold, 2, dim=0)

        # cross attention (m0 <-> m1)
        feat_m0_unfold, feat_m1_unfold = self.cross_attn_m(feat_m0_unfold, feat_m1_unfold)

        # down proj m
        feat_m_unfold = self.down_proj_m_f(torch.cat([feat_m0_unfold, feat_m1_unfold], dim=0))
        feat_m0_unfold, feat_m1_unfold = torch.chunk(feat_m_unfold, 2, dim=0)

        # self attention (m + f)
        feat_f_unfold, _ = self.self_attn_f(torch.cat([feat_f0_unfold, feat_f1_unfold], dim=0),
                                                         torch.cat([feat_m0_unfold, feat_m1_unfold], dim=0))
        feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_f_unfold, 2, dim=0)

        # cross attention (f0 <-> f1)
        feat_f0_unfold, feat_f1_unfold = self.cross_attn_f(feat_f0_unfold, feat_f1_unfold)

        return feat_f0_unfold, feat_f1_unfold


# ----------------------------------------------------------------------------------------------------
# upstream src/xoftr/xoftr_module/fine_matching.py
# ----------------------------------------------------------------------------------------------------

class FineSubMatching(nn.Module):
    """Fine-level and Sub-pixel matching"""

    def __init__(self, config):
        super().__init__()
        self.temperature = config['fine']['dsmax_temperature']
        self.W_f = config['fine_window_size']
        self.denser = config['fine']['denser']
        self.inference = config['fine']['inference']
        dim_f = config['resnet']['block_dims'][0]
        self.fine_thr = config['fine']['thr']
        self.fine_proj = nn.Linear(dim_f, dim_f, bias=False)
        self.subpixel_mlp = nn.Sequential(nn.Linear(2*dim_f, 2*dim_f, bias=False),
                                           nn.ReLU(),
                                           nn.Linear(2*dim_f, 4, bias=False))

    def forward(self, feat_f0_unfold, feat_f1_unfold, data):
        """
        Args:
            feat_f0_unfold (torch.Tensor): [M, WW, C]
            feat_f1_unfold (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """

        feat_f0 = self.fine_proj(feat_f0_unfold)
        feat_f1 = self.fine_proj(feat_f1_unfold)

        M, WW, C = feat_f0.shape
        W_f = self.W_f

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
                'mconf_f': torch.zeros(0, device=feat_f0_unfold.device),
                # 'mkpts0_f_train': data['mkpts0_c'],
                # 'mkpts1_f_train': data['mkpts1_c'],
                # 'conf_matrix_fine': torch.zeros(1, W_f*W_f, W_f*W_f, device=feat_f0.device)
            })
            return

        # normalize
        feat_f0, feat_f1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_f0, feat_f1])
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0,
                                      feat_f1) / self.temperature

        conf_matrix_fine = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data.update({'conf_matrix_fine': conf_matrix_fine})

        # predict fine-level and sub-pixel matches from conf_matrix
        data.update(**self.get_fine_sub_match(conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data))

    def get_fine_sub_match(self, conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data):
        """
        Args:
            conf_matrix_fine (torch.Tensor): [M, WW, WW]
            feat_f0_unfold (torch.Tensor): [M, WW, C]
            feat_f1_unfold (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'm_bids' (torch.Tensor): [M]
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """

        with torch.no_grad():
            W_f = self.W_f

            # 1. confidence thresholding
            mask = conf_matrix_fine > self.fine_thr

            if mask.sum() == 0:
                mask[0,0,0] = 1
                conf_matrix_fine[0,0,0] = 1

            if not self.denser:
                # match only the highest confidence
                mask = mask \
                    * (conf_matrix_fine == conf_matrix_fine.amax(dim=[1,2], keepdim=True))
            else:
                # 2. mutual nearest, match all features in fine window
                mask = mask \
                    * (conf_matrix_fine == conf_matrix_fine.max(dim=2, keepdim=True)[0]) \
                    * (conf_matrix_fine == conf_matrix_fine.max(dim=1, keepdim=True)[0])

            # 3. find all valid fine matches
            # this only works when at most one `True` in each row
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix_fine[b_ids, i_ids, j_ids]

            # 4. update with matches in original image resolution

            # indices from coarse matches
            b_ids_c, i_ids_c, j_ids_c = data['b_ids'], data['i_ids'], data['j_ids']

            # scale (coarse level / fine-level)
            scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

            # coarse level matches scaled to fine-level (1/2)
            mkpts0_c_scaled_to_f = torch.stack(
            [i_ids_c % data['hw0_c'][1], torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale_f_c

            mkpts1_c_scaled_to_f = torch.stack(
                [j_ids_c % data['hw1_c'][1], torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            # updated b_ids after second thresholding
            updated_b_ids = b_ids_c[b_ids]

            # scales (image res / fine level)
            scale = data['hw0_i'][0] / data['hw0_f'][0]
            scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

            # fine-level discrete matches on window coordiantes
            mkpts0_f_window = torch.stack(
            [i_ids % W_f, torch.div(i_ids, W_f, rounding_mode='trunc')],
            dim=1)

            mkpts1_f_window = torch.stack(
            [j_ids % W_f, torch.div(j_ids, W_f, rounding_mode='trunc')],
            dim=1)

        # sub-pixel refinement
        sub_ref = self.subpixel_mlp(torch.cat([feat_f0_unfold[b_ids, i_ids],
                                                     feat_f1_unfold[b_ids, j_ids]], dim=-1))
        sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=-1)
        sub_ref0 = torch.tanh(sub_ref0) * 0.5
        sub_ref1 = torch.tanh(sub_ref1) * 0.5

        # final sub-pixel matches by (coarse-level + fine-level windowed + sub-pixel refinement)
        mkpts0_f_train = (mkpts0_f_window + mkpts0_c_scaled_to_f[b_ids] - (W_f//2) + sub_ref0) * scale0
        mkpts1_f_train = (mkpts1_f_window + mkpts1_c_scaled_to_f[b_ids] - (W_f//2) + sub_ref1) * scale1
        mkpts0_f = mkpts0_f_train.clone().detach()
        mkpts1_f = mkpts1_f_train.clone().detach()

        # These matches is the current prediction (for visualization)
        sub_pixel_matches = {
            'm_bids': b_ids_c[b_ids[mconf != 0]],  # mconf == 0 => gt matches
            'mkpts0_f': mkpts0_f[mconf != 0],
            'mkpts1_f': mkpts1_f[mconf != 0],
            'mconf_f': mconf[mconf != 0]
        }

        # These matches are used for training
        if not self.inference:
            sub_pixel_matches.update({
                'mkpts0_f_train': mkpts0_f_train[mconf != 0],
                'mkpts1_f_train': mkpts1_f_train[mconf != 0],
            })

        return sub_pixel_matches


# ----------------------------------------------------------------------------------------------------
# upstream src/xoftr/xoftr.py
# ----------------------------------------------------------------------------------------------------

class XoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = ResNet_8_2(config['resnet'])
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)


    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data)

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
