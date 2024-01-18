import torch
import warnings
from typing import Dict, Optional, Tuple, Union
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from copy import deepcopy
from abc import ABCMeta, abstractmethod


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module: nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    _abbr_ = 'conv_block'
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,  # 3
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: tuple = ('conv', 'norm', 'act')):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode  # False
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # reset padding to 0 for conv module
        conv_padding = padding
        self.conv = torch.nn.modules.Conv2d(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=conv_padding, dilation=dilation,
                                            groups=groups, bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            # self.norm_name, norm = build_norm_layer(
            #     norm_cfg, norm_channels)  # type: ignore
            self.norm_name = 'bn'
            norm = nn.BatchNorm2d(norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None  # type: ignore

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            # self.activate = build_activation_layer(act_cfg_)
            self.activate = nn.modules.ReLU(inplace=True)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x: torch.Tensor, statistic: bool = False, activate: bool = True, norm: bool = True):
        if statistic:
            for layer in self.order:
                if layer == 'conv':
                    if self.with_explicit_padding:
                        x = self.padding_layer(x)
                    x = self.conv(x)  # [batch_size*num_segments,channel,...]
                elif layer == 'norm' and norm and self.with_norm:
                    dims = [0] + list(range(2,len(x.shape)))
                    mean = x.mean(dim=dims).cpu()  #
                    var = x.var(dim=dims).cpu()
                    x = self.norm(x)
                elif layer == 'act' and activate and self.with_activation:
                    x = self.activate(x)
            return x, mean, var
        else:
            for layer in self.order:
                if layer == 'conv':
                    if self.with_explicit_padding:
                        x = self.padding_layer(x)
                    x = self.conv(x)
                elif layer == 'norm' and norm and self.with_norm:
                    x = self.norm(x)
                elif layer == 'act' and activate and self.with_activation:
                    x = self.activate(x)
            return x


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x, statistic=False, **kwargs):
        if statistic:
            mean = kwargs.get('mean', [])
            var = kwargs.get('var', [])
            identity = x
            out, mean_temp_1, var_temp_1 = self.conv1(x, True)
            mean.append(mean_temp_1)
            var.append(var_temp_1)
            out, mean_temp_2, var_temp_2 = self.conv2(out, True)
            mean.append(mean_temp_2)
            var.append(var_temp_2)
            out, mean_temp_3, var_temp_3 = self.conv3(out, True)
            mean.append(mean_temp_3)
            var.append(var_temp_3)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            out = self.relu(out)
            return out, mean, var
        else:
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            return self.relu(out)


class TAM(nn.Module):
    def __init__(self,
                 in_channels,
                 num_segments,
                 alpha=2,
                 adaptive_kernel_size=3,
                 beta=4,
                 conv1d_kernel_size=3,
                 adaptive_convolution_stride=1,
                 adaptive_convolution_padding=1,
                 init_std=0.001):
        super().__init__()

        assert beta > 0 and alpha > 0
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.alpha = alpha
        self.adaptive_kernel_size = adaptive_kernel_size
        self.beta = beta
        self.conv1d_kernel_size = conv1d_kernel_size
        self.adaptive_convolution_stride = adaptive_convolution_stride
        self.adaptive_convolution_padding = adaptive_convolution_padding
        self.init_std = init_std

        self.G = nn.Sequential(
            nn.Linear(num_segments, num_segments * alpha, bias=False),
            nn.BatchNorm1d(num_segments * alpha), nn.ReLU(inplace=True),
            nn.Linear(num_segments * alpha, adaptive_kernel_size, bias=False),
            nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels // beta,
                conv1d_kernel_size,
                stride=1,
                padding=conv1d_kernel_size // 2,
                bias=False), nn.BatchNorm1d(in_channels // beta),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // beta, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        # [n, c, h, w]
        n, c, h, w = x.size()
        num_segments = self.num_segments
        num_batches = n // num_segments
        assert c == self.in_channels

        # [num_batches, c, num_segments, h, w]
        x = x.view(num_batches, num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # [num_batches * c, num_segments, 1, 1]
        theta_out = F.adaptive_avg_pool2d(
            x.view(-1, num_segments, h, w), (1, 1))

        # [num_batches * c, 1, adaptive_kernel_size, 1]
        conv_kernel = self.G(theta_out.view(-1, num_segments)).view(
            num_batches * c, 1, -1, 1)

        # [num_batches, c, num_segments, 1, 1]
        local_activation = self.L(theta_out.view(-1, c, num_segments)).view(
            num_batches, c, num_segments, 1, 1)

        # [num_batches, c, num_segments, h, w]
        new_x = x * local_activation

        # [1, num_batches * c, num_segments, h * w]
        y = F.conv2d(
            new_x.view(1, num_batches * c, num_segments, h * w),
            conv_kernel,
            bias=None,
            stride=(self.adaptive_convolution_stride, 1),
            padding=(self.adaptive_convolution_padding, 0),
            groups=num_batches * c)

        # [n, c, h, w]
        y = y.view(num_batches, c, num_segments, h, w)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)

        return y


class TABlock(nn.Module):
    def __init__(self, block, num_segments, tam_cfg=dict()):
        super().__init__()
        self.tam_cfg = deepcopy(tam_cfg)
        self.block = block
        self.num_segments = num_segments
        self.tam = TAM(
            in_channels=block.conv1.out_channels,
            num_segments=num_segments,
            **self.tam_cfg)

        if not isinstance(self.block, Bottleneck):
            raise NotImplementedError('TA-Blocks have not been fully '
                                      'implemented except the pattern based '
                                      'on Bottleneck block.')

    def forward(self, x, statistic=False, **kwargs):
        if statistic:
            mean = kwargs.get('mean', [])
            var = kwargs.get('var', [])
            identity = x
            out, mean_temp_1, var_temp_1 = self.block.conv1(x, True)
            mean.append(mean_temp_1)
            var.append(var_temp_1)
            out = self.tam(out)
            out, mean_temp_2, var_temp_2 = self.block.conv2(out, True)
            mean.append(mean_temp_2)
            var.append(var_temp_2)
            out, mean_temp_3, var_temp_3 = self.block.conv3(out, True)
            mean.append(mean_temp_3)
            var.append(var_temp_3)
            if self.block.downsample is not None:
                identity = self.block.downsample(x)
            out = out + identity
            out = self.block.relu(out)
            return out, mean, var
        else:
            identity = x
            out = self.block.conv1(x)
            out = self.tam(out)
            out = self.block.conv2(out)
            out = self.block.conv3(out)
            if self.block.downsample is not None:
                identity = self.block.downsample(x)
            out = out + identity
            return self.block.relu(out)


class TANetAdapt(nn.Module):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    def __init__(self,
                 depth,  # 50
                 num_segments=16,
                 tam_cfg=dict(),
                 pretrained=None,
                 torchvision_pretrain=True,
                 in_channels=3,
                 num_stages=4,
                 out_indices=(2, 3),  # last two block
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 partial_bn=False,
                 with_cp=False):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.torchvision_pretrain = torchvision_pretrain
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.strides = strides
        self.dilations = dilations
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self.__make_stem_layer()
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = self.__make_res_layer(self.block, self.inplanes, planes, num_blocks, stride=stride,
                                              dilation=dilation, style=self.style, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                              act_cfg=act_cfg, with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)
        self.num_segments = num_segments
        self.tam_cfg = deepcopy(tam_cfg)


    def __make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def __make_res_layer(self,
                         block,
                         inplanes,
                         planes,
                         blocks,
                         stride=1,
                         dilation=1,
                         style='pytorch',
                         conv_cfg=None,
                         norm_cfg=None,
                         act_cfg=None,
                         with_cp=False):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                dilation,
                downsample,
                style=style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    1,
                    dilation,
                    style=style,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))

        return nn.Sequential(*layers)

    def forward(self, x):
        img_shape = x.shape[-3:]
        # x = x.view(-1, *img_shape)
        x = x.reshape(-1, *img_shape)
        x = self.conv1(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        return x  # Tensor list list

    def init_weights(self):
        # super().init_weights()
        self.make_tam_modeling()

    def make_tam_modeling(self):
        """Replace ResNet-Block with TA-Block."""

        def make_tam_block(stage, num_segments, tam_cfg=dict()):
            blocks = list(stage.children())
            for i, block in enumerate(blocks):
                blocks[i] = TABlock(block, num_segments, deepcopy(tam_cfg))
            return nn.Sequential(*blocks)

        for i in range(self.num_stages):
            layer_name = f'layer{i + 1}'
            res_layer = getattr(self, layer_name)
            setattr(self, layer_name,
                    make_tam_block(res_layer, self.num_segments, self.tam_cfg))


class AvgConsensus(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)


class TSMHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=16,
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.5,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool
        self.input_size = kwargs.get('input_size', 224)
        self.downsample = self.input_size // 32  # 下采样32倍
        consensus_ = consensus.copy()
        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def forward(self, x):
        x = x.reshape(-1, self.in_channels, self.downsample, self.downsample)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        feat = x.view(-1, self.num_segments, self.in_channels)
        feat = feat.mean(dim=1)
        cls_score = self.fc_cls(x)
        if self.is_shift and self.temporal_pool:
            cls_score = cls_score.view((-1, self.num_segments // 2) + cls_score.size()[1:])
        else:
            cls_score = cls_score.view((-1, self.num_segments) + cls_score.size()[1:])
        cls_score = self.consensus(cls_score)
        return feat, cls_score.squeeze(1)


class BaseHead(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 num_classes,
                 in_channels,
                 multi_class=False,
                 label_smooth_eps=0.0,
                 topk=(1, 5)):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""


class TSNHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segs,
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.4,
                 init_std=0.001,
                 **kwargs):
        super().__init__(num_classes, in_channels)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.in_channels = in_channels
        self.num_segs = num_segs
        self.input_size = kwargs.get('input_size', 224)
        self.downsample = self.input_size // 32  # 下采样32倍

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, x):
        x = x.reshape(-1, self.in_channels, self.downsample, self.downsample)
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, self.num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        feat = x.view(x.size(0), -1)
        # [N, in_channels]
        cls_score = self.fc_cls(feat)
        # [N, num_classes]
        return feat, cls_score



