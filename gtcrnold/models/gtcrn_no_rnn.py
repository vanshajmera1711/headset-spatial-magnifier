"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.tra(h1)

        x =  self.shuffle(h1, x2)
        
        return x


# class GRNN(nn.Module):
#     """Grouped RNN"""
#     def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
#         self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

#     def forward(self, x, h=None):
#         """
#         x: (B, seq_length, input_size)
#         h: (num_layers, B, hidden_size)
#         """
#         if h== None:
#             if self.bidirectional:
#                 h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
#             else:
#                 h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
#         x1, x2 = torch.chunk(x, chunks=2, dim=-1)
#         h1, h2 = torch.chunk(h, chunks=2, dim=-1)
#         h1, h2 = h1.contiguous(), h2.contiguous()
#         y1, h1 = self.rnn1(x1, h1)
#         y2, h2 = self.rnn2(x2, h2)
#         y = torch.cat([y1, y2], dim=-1)
#         h = torch.cat([h1, h2], dim=-1)
#         return y, h
    
    
# class DPGRNN(nn.Module):
#     """Grouped Dual-path RNN"""
#     def __init__(self, input_size, width, hidden_size, **kwargs):
#         super(DPGRNN, self).__init__(**kwargs)
#         self.input_size = input_size
#         self.width = width
#         self.hidden_size = hidden_size

#         self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
#         self.intra_fc = nn.Linear(hidden_size, hidden_size)
#         self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

#         self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
#         self.inter_fc = nn.Linear(hidden_size, hidden_size)
#         self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
    
#     def forward(self, x):
#         """x: (B, C, T, F)"""
#         ## Intra RNN
#         x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
#         intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
#         intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
#         intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
#         intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
#         intra_x = self.intra_ln(intra_x)
#         intra_out = torch.add(x, intra_x)

#         ## Inter RNN
#         x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
#         inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
#         inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
#         inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
#         inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
#         inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
#         inter_x = self.inter_ln(inter_x) 
#         inter_out = torch.add(intra_out, inter_x)
        
#         dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
#         return dual_out

class StackDownsampler(nn.Module):
    """
    Downsamples by stacking consecutive frequency bins into the channel dimension.
    Converts (B, C, T, 257) -> (B, C*2, T, 129)
    """
    def __init__(self, in_freq=257, out_freq=129):
        super().__init__()
        self.in_freq = in_freq
        self.out_freq = out_freq
        
        # Calculate how many bins will be paired
        # (129 - 1) * 2 = 128 * 2 = 256
        self.paired_freq = (out_freq - 1) * 2
        
    def forward(self, x):
        # Input x shape: (B, C, T, F_in), e.g., (B, 3, T, 257)
        B, C, T, _ = x.shape
        
        # Split into the part we can pair and the part that is odd
        x_paired = x[..., :self.paired_freq] # (B, C, T, 256)
        x_odd = x[..., self.paired_freq:]   # (B, C, T, 1)
        
        # Reshape the paired part to (B, C, T, 128, 2)
        x_paired_reshaped = x_paired.reshape(B, C, T, self.out_freq - 1, 2)
        
        # Permute to (B, C, 2, T, 128) to bring stack dim next to channel dim
        x_paired_permuted = x_paired_reshaped.permute(0, 1, 4, 2, 3)
        
        # Reshape to stack channels -> (B, C*2, T, 128) e.g. (B, 6, T, 128)
        x_stacked = x_paired_permuted.reshape(B, C * 2, T, self.out_freq - 1)
        
        # Pad the odd bin to match the new channel dim (C*2)
        # We pad dim 1 (channels) from C to C*2 with (0, C)
        x_odd_padded = torch.nn.functional.pad(x_odd, (0, 0, 0, 0, 0, C), 'constant', 0)
        # x_odd_padded shape: (B, 6, T, 1)
        
        # Concatenate along the frequency dimension
        # (B, 6, T, 128) + (B, 6, T, 1) -> (B, 6, T, 129)
        output = torch.cat([x_stacked, x_odd_padded], dim=-1)
        
        return output
    
class StackUpsampler(nn.Module):
    """
    Upsamples by un-stacking the channel dimension back into the frequency dimension.
    Converts (B, C*2, T, 129) -> (B, C, T, 257)
    """
    def __init__(self, in_freq=129, out_freq=257):
        super().__init__()
        self.out_freq = out_freq
        self.in_freq = in_freq
        
        # Calculate the paired frequency bins
        self.paired_freq = out_freq - 1 # 256
        
    def forward(self, x):
        # Input x shape: (B, C_in, T, F_in), e.g., (B, 4, T, 129)
        B, C_in, T, _ = x.shape
        
        # Check for even number of channels
        if C_in % 2 != 0:
            raise ValueError(f"StackUpsampler requires an even number of input channels, but got {C_in}")
            
        C_out = C_in // 2
        
        # Split the stacked part and the odd, padded bin
        x_stacked = x[..., :self.in_freq - 1] # (B, C_in, T, 128)
        x_odd_padded = x[..., self.in_freq - 1:]  # (B, C_in, T, 1)
        
        # Un-pad the odd bin by just taking the first C_out channels
        x_odd_unpadded = x_odd_padded[:, :C_out, ...] # (B, C_out, T, 1)

        # Reshape stacked part to (B, C_out, 2, T, 128)
        x_unstacked_reshaped = x_stacked.reshape(B, C_out, 2, T, self.in_freq - 1)
        
        # Permute to (B, C_out, T, 128, 2)
        x_unstacked_permuted = x_unstacked_reshaped.permute(0, 1, 3, 4, 2)
        
        # Reshape to (B, C_out, T, 256)
        x_unstacked = x_unstacked_permuted.reshape(B, C_out, T, self.paired_freq)

        # Concatenate along the frequency dimension
        # (B, C_out, T, 256) + (B, C_out, T, 1) -> (B, C_out, T, 257)
        output = torch.cat([x_unstacked, x_odd_unpadded], dim=-1)
        
        return output

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(in_channels, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 4, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRN(nn.Module):
    def __init__(self, n_channels: int):
        """
        Initialize the multi-channel capable GTCRN.
        
        :param n_in_channels: The number of input channels. This should be
                              2 * (number of audio channels) from the
                              stacked real/imag spectrogram.
        """
        super().__init__()
        
        self.n_in_channels = n_channels*2
        
        # self.erb = ERB(65, 64) # This ERB logic is not used, so we keep it commented.
        
        # Calculate the number of channels after freq_downsampler and SFE
        # 1. freq_downsampler stacks 2 bins, so channels become n_in_channels * 2
        # 2. SFE has kernel_size=3, so channels become (n_in_channels * 2) * 3
        encoder_in_channels = (self.n_in_channels * 2) * 3
        
        self.freq_downsampler = StackDownsampler(in_freq=257, out_freq=129)
        self.freq_upsampler = StackUpsampler(in_freq=129, out_freq=257)
        
        self.sfe = SFE(3, 1)

        # Pass the calculated input channels to the Encoder
        self.encoder = Encoder(in_channels=encoder_in_channels)
        
        # self.dpgrnn1 = DPGRNN(16, 33, 16)
        # self.dpgrnn2 = DPGRNN(16, 33, 16)
        
        self.decoder = Decoder()

        self.mask = Mask()
        self.output_type = 'CRM' # Model outputs a complex mask

    def forward(self, sta_spec):
        """
        Forward pass for multi-channel GTCRN.

        :param sta_spec: Stacked complex spectrogram [B, C_in, F, T]
                         where C_in is 2 * num_audio_channels.
        :return: A single-channel complex mask [B, 2, F, T]
        """
        # sta_spec shape is [B, C_in, F, T], e.g. [B, 6, 257, T]
        # Permute to [B, C_in, T, F] for processing
        feat = sta_spec.permute(0, 1, 3, 2) # [B, 6, T, 257]

        # --- Removed single-channel feature extraction ---
        # The (mag, real, imag) logic is gone.
        # We now use all input channels as features.
        # -----------------------------------------------

        # feat = self.erb.bm(feat)  # (B, C_in, T, 129)
        
        # Downsample frequency, stacking into channels
        feat = self.freq_downsampler(feat) # [B, C_in*2, T, 129], e.g. [B, 12, T, 129]
        
        # Subband Feature Extraction
        feat = self.sfe(feat)     # [B, (C_in*2)*3, T, 129], e.g. [B, 36, T, 129]

        # Encoder
        feat, en_outs = self.encoder(feat)
        
        # Dual-path RNNs
        # feat = self.dpgrnn1(feat) # [B, 16, T, 33]
        # feat = self.dpgrnn2(feat) # [B, 16, T, 33]

        # Decoder
        m_feat = self.decoder(feat, en_outs) # [B, 4, T, 129]
        
        # Upsample frequency, un-stacking from channels
        m = self.freq_upsampler(m_feat) # [B, 2, T, 257]
        
        # m = self.erb.bs(m_feat) #(B,2,T,F)
        
        # Permute back to [B, 2, F, T]
        return m.permute(0,1,3,2)

if __name__ == "__main__":
    model = GTCRN().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print(flops, params)

    """causality check"""
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)
    
    x1 = torch.stft(x1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    x2 = torch.stft(x2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y1 = model(x1)[0]
    y2 = model(x2)[0]
    y1 = torch.istft(y1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y2 = torch.istft(y2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    
    print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
