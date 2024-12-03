import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
from pytorch_model_summary import summary
import numpy as np


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)

        Usage:
        self.embedding_xyz = Embedding(3, 10) # 10 is the default number <- 3 means having x, y, z
        self.embedding_dir = Embedding(3, 4) # 4 is the default number <- for angles, here we are not using it
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class SineLayer(nn.Module):
    # See Siren paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, KSize, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Conv1d(in_features, out_features, KSize, bias=bias, padding='same')
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, KSize,
                 first_omega_0=10, hidden_omega_0=3.): # here first_omega and hidden_omega are hyperparameters
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, KSize,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, KSize,
                                      is_first=False, omega_0=hidden_omega_0))

        final_linear = nn.Conv1d(hidden_features, out_features, 1)

        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                         np.sqrt(6 / hidden_features) / hidden_omega_0)

        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output


class Siren_Block(nn.Module):
    """
    Input is [batch, emb, time]
    """

    def __init__(self, in_channels, hidden_channels=5):
        super(Siren_Block, self).__init__()
        self.HIDDEN_CHANNELS = hidden_channels
        self.act = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, self.HIDDEN_CHANNELS, 1,padding='same')
        self.siren = Siren(in_channels, hidden_features=256, hidden_layers=1, out_features=self.HIDDEN_CHANNELS,
                           KSize=1)
    def forward(self, x):
        """
        """
        # ------ siren --------
        # notes: here can also add residual connections as you want
        # res = self.residual(x)
        # res = self.act(res)
        x = self.siren(x)

        return x


class ConvBlock(nn.Module):
    """
    Input is [batch, emb, time]
    simple conv block from wav2vec 2.0
        - conv
        - layer norm by embedding axis
        - activation
    To do:
        add res blocks.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, p_conv_drop=0.1):
        super(ConvBlock, self).__init__()

        # use it instead stride.

        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                bias=False,
                                padding='same')

        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=p_conv_drop)

        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        """
        - conv
        - norm
        - activation
        - downsample

        """

        x = self.conv1d(x)
        # norm by last axis.
        x = torch.transpose(x, -2, -1)
        x = self.norm(x)
        x = torch.transpose(x, -2, -1)
        x = self.activation(x)
        x = self.drop(x)
        x = self.downsample(x)

        return x


class UpConvBlock(nn.Module):
    def __init__(self, scale, **args):
        super(UpConvBlock, self).__init__()
        self.conv_block = ConvBlock(**args)
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.upsample(x)
        return x


class AutoEncoder1D(nn.Module):
    """
    This is implementation of AutoEncoder1D model for time serias regression

    decoder_reduce  -size of reducing parameter on decoder stage. We do not want use a lot of features here.
    """

    def __init__(self,
                 n_electrodes=30,
                 n_freqs=16,
                 n_channels_out=21,
                 channels=[8, 16, 32, 32],
                 kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4],
                 dilation=[1, 1, 1],
                 decoder_reduce=1,
                 hidden_channels=5,
                 ):

        super(AutoEncoder1D, self).__init__()

        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_freqs * n_electrodes
        self.n_channels_out = n_channels_out

        self.model_depth = len(channels) - 1

        self.artur_block = Siren_Block(in_channels=self.n_electrodes, hidden_channels=hidden_channels)
        self.spatial_reduce = ConvBlock(self.artur_block.HIDDEN_CHANNELS, channels[0], kernel_size=3)

        # create downsample blcoks in Sequentional manner.
        self.downsample_blocks = nn.ModuleList([ConvBlock(channels[i],
                                                          channels[i + 1],
                                                          kernel_sizes[i],
                                                          stride=strides[i],
                                                          dilation=dilation[i]) for i in range(self.model_depth)])

        # make the same but in another side w/o last conv.
        channels = [ch // decoder_reduce for ch in channels[:-1]] + channels[-1:]
        # channels
        self.upsample_blocks = nn.ModuleList([UpConvBlock(scale=strides[i],
                                                          in_channels=channels[i + 1],
                                                          out_channels=channels[i],
                                                          kernel_size=kernel_sizes[i]) for i in
                                              range(self.model_depth - 1, -1, -1)])

        self.conv1x1_one = nn.Conv1d(channels[0], self.n_channels_out, kernel_size=1, padding='same')


    def forward(self, x):
        """
        """
        batch, elec, time = x.shape

        x = self.artur_block(x)
        x = self.spatial_reduce(x)

        # encode information
        for i in range(self.model_depth):
            x = self.downsample_blocks[i](x)

        for i in range(self.model_depth):
            x = self.upsample_blocks[i](x)

        x = self.conv1x1_one(x)

        return x


class Model(nn.Module):
    """
    This is implementation of AutoEncoder1D model for time serias regression

    decoder_reduce  -size of reducing parameter on decoder stage. We do not want use a lot of features here.
    """

    def __init__(self, dict_setting):
        super(Model, self).__init__()
        dict_setting_new = dict_setting.copy()
        self.n_channels_out = dict_setting_new['n_channels_out']

        dict_setting_new.pop('n_channels_out')

        # print('HOW: ', dict_setting)
        self.models = nn.ModuleList(
            [AutoEncoder1D(n_channels_out=1, **dict_setting_new) for i in range(self.n_channels_out)])
        # print('HUI', len(self.models))

    def forward(self, x):
        """
        """
        batch, elec, time = x.shape
        preds = [model(x) for model in self.models]
        preds = torch.cat(preds, dim=1)

        return preds