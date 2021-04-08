import torch
import torch.nn as nn

from dsp import minmax_scale
from hparams import hparams as hp


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=hp.model.encoder.input_channels,
                      out_channels=hp.model.encoder.expand_channels,
                      kernel_size=(1, 1),
                      bias=False),
            nn.GroupNorm(hp.model.encoder.expand_channels // 16, hp.model.encoder.expand_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hp.model.encoder.expand_channels,
                      out_channels=1,
                      kernel_size=(1, 1),
                      bias=False),
        )
        self.lstm = nn.LSTM(input_size=2 * hp.dsp.num_mels + 2 * hp.dsp.num_audio_features,
                            hidden_size=hp.model.encoder.lstm_hidden_dims,
                            num_layers=hp.model.encoder.lstm_num_layers,
                            batch_first=True,
                            bidirectional=True)
        # self.fc = nn.Linear(in_features=2 * hp.model.encoder.lstm_num_layers * hp.model.encoder.lstm_hidden_dims +
        #                                 hp.dsp.num_transient_spectrum_bins,
        #                     out_features=hp.model.encoder.encoding_dims)
        self.fc = nn.Linear(in_features=2 * hp.model.encoder.lstm_num_layers * hp.model.encoder.lstm_hidden_dims,
                            out_features=hp.model.encoder.encoding_dims)

    def forward(self, x, audio_features):  # , transient_spectrum):
        L, R = torch.split(x, 3, dim=1)
        L = self.in_conv(L).squeeze(1).transpose(1, 2)
        R = self.in_conv(R).squeeze(1).transpose(1, 2)
        audio_features = audio_features.transpose(1, 2)
        x = torch.cat((L, R, audio_features), dim=2)
        self.lstm.flatten_parameters()
        _, (h, _) = self.lstm(x)
        h = h.view(1, -1)
        # x = torch.cat((h, transient_spectrum), dim=1)
        return self.fc(h)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.num_mels = hp.dsp.num_mels
        self.lstm = nn.LSTM(input_size=hp.model.encoder.encoding_dims,
                            hidden_size=hp.model.decoder.lstm_hidden_dims,
                            num_layers=hp.model.decoder.lstm_num_layers,
                            batch_first=True,
                            bidirectional=False)
        self.fc = nn.Linear(in_features=hp.model.decoder.lstm_hidden_dims,
                            out_features=2 * hp.dsp.num_mels)

    def forward(self, encoding):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(encoding)
        x = torch.sigmoid(self.fc(x))
        x = x.view(1, self.num_mels, -1, 2)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inputs, features):  # , transient_spectrum):
        encoding = self.encoder(inputs, features)  # , transient_spectrum)
        decoder_input = encoding.unsqueeze(1).repeat(1, inputs.shape[-2], 1)
        reconstruction = self.decoder(decoder_input)
        return minmax_scale(reconstruction,
                            series_min=0.,
                            series_max=1.,
                            new_min=hp.dsp.min_vol,
                            new_max=hp.dsp.max_vol), \
               encoding

# class ConvLSTMCell(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size, bias):
#         """
#         Initialize ConvLSTM cell.
#         Parameters
#         ----------
#         input_dim: int
#             Number of channels of input tensor.
#         hidden_dim: int
#             Number of channels of hidden state.
#         kernel_size: (int, int)
#             Size of the convolutional kernel.
#         bias: bool
#             Whether or not to add the bias.
#         """
#
#         super(ConvLSTMCell, self).__init__()
#
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#
#         self.k_size = kernel_size
#         self.bias = bias
#
#         self.pad = nn.ReflectionPad1d(kernel_size // 2)
#         self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
#                               out_channels=4 * self.hidden_dim,
#                               kernel_size=self.k_size,
#                               bias=self.bias)
#
#     def forward(self, input_tensor, cur_state):
#         h_cur, c_cur = cur_state
#
#         combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
#         combined = self.pad(combined)
#         combined_conv = self.conv(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = torch.tanh(cc_g)
#
#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)
#
#         return h_next, c_next
#
#
# class DenseLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size):
#         super(DenseLayer, self).__init__()
#
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.kernel_size = kernel_size
#         self.fw = ConvLSTMCell(input_dim=self.input_dim,
#                                hidden_dim=self.hidden_dim,
#                                kernel_size=self.kernel_size,
#                                bias=False)
#         if hp.model.densesynth.bidirectional:
#             self.bw = ConvLSTMCell(input_dim=self.input_dim,
#                                    hidden_dim=self.hidden_dim,
#                                    kernel_size=self.kernel_size,
#                                    bias=False)
#
#     def forward(self, x):
#         h_fw, c_fw = self.init_hidden(batch_size=x.shape[0], num_mels=x.shape[2])
#         output_inner_fw = []
#         if hp.model.densesynth.bidirectional:
#             h_bw, c_bw = self.init_hidden(batch_size=x.shape[0], num_mels=x.shape[2])
#             output_inner_bw = []
#         for t in range(x.shape[-1]):
#             h_fw, c_fw = self.fw(input_tensor=x[:, :, :, t],
#                                  cur_state=[h_fw, c_fw])
#             output_inner_fw.append(h_fw)
#             if hp.model.densesynth.bidirectional:
#                 h_bw, c_bw = self.bw(input_tensor=x[:, :, :, (-(t + 1))],
#                                      cur_state=[h_bw, c_bw])
#                 output_inner_bw.insert(0, h_bw)
#
#         x = torch.stack(output_inner_fw, dim=3)
#         if hp.model.densesynth.bidirectional:
#             x += torch.stack(output_inner_bw, dim=3)
#         return x
#
#     def init_hidden(self, batch_size, num_mels):
#         return torch.zeros(batch_size, self.hidden_dim, num_mels, device=self.fw.conv.weight.device), \
#                torch.zeros(batch_size, self.hidden_dim, num_mels, device=self.fw.conv.weight.device)
#
#
# class DenseBlock(nn.Module):
#     def __init__(self, input_dim, growth_rate, kernel_sizes):
#         super(DenseBlock, self).__init__()
#         self.input_dim = input_dim
#         self.num_layers = len(kernel_sizes)
#         self.growth_rate = growth_rate
#         self.kernel_sizes = kernel_sizes
#         self.bottleneck_dims = growth_rate * 2
#         self.convs = nn.ModuleList()
#         for i in range(self.num_layers):
#             self.convs.append(
#                 nn.Sequential(
#                     nn.BatchNorm2d(self.input_dim + i * self.growth_rate),
#                     nn.LeakyReLU(),
#                     nn.Conv2d(in_channels=self.input_dim + i * self.growth_rate,
#                               out_channels=self.bottleneck_dims,
#                               kernel_size=(1, 1),
#                               bias=False),
#                     DenseLayer(input_dim=self.bottleneck_dims,
#                                hidden_dim=self.growth_rate,
#                                kernel_size=self.kernel_sizes[i])
#                 )
#             )
#         self.transition = nn.Sequential(
#             nn.BatchNorm2d(self.input_dim + self.num_layers * self.growth_rate),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=self.input_dim + self.num_layers * self.growth_rate,
#                       out_channels=(self.input_dim + self.num_layers * self.growth_rate) // 2,
#                       kernel_size=(1, 1),
#                       bias=False)
#         )
#
#     def forward(self, x):
#         for i in range(self.num_layers):
#             out = self.convs[i](x)
#             x = torch.cat((x, out), dim=1)
#         return self.transition(x)
#
#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.num_blocks = hp.model.encoder.num_blocks
#         self.in_conv = nn.Conv2d(in_channels=hp.model.encoder.input_dims,
#                                  out_channels=hp.model.encoder.growth_rates[0] * 2,
#                                  kernel_size=(1, 1),
#                                  bias=False)
#         self.blocks = nn.ModuleList()
#         input_dims = [hp.model.encoder.growth_rates[0] * 2]
#         for i in range(self.num_blocks):
#             k = hp.model.encoder.growth_rates[i]
#             kernel_sizes = hp.model.encoder.kernel_sizes[i]
#             input_dims.append((input_dims[-1] + len(kernel_sizes) * k) // 2)
#             self.dense_blocks.append(
#                 DenseBlock(input_dim=input_dims[i],
#                            growth_rate=k,
#                            kernel_sizes=kernel_sizes)
#             )
#         self.out_conv = nn.Sequential(
#             nn.BatchNorm1d(input_dims[-1]),
#             nn.LeakyReLU(),
#             nn.Conv1d(in_channels=input_dims[-1],
#                       out_channels=1,
#                       kernel_size=(1, 1),
#                       bias=True)
#         )
#
#     def forward(self, x):
#         x = self.in_conv(x)
#         for block in self.blocks:
#             x = block(x)
#         x = x[:, :, :, -1]
#         x = self.out_conv(x)
#         return torch.tanh(x.squeeze(1))
