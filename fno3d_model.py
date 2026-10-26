import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv3d(nn.Module):
    """
    High-Frequency 3D Fourier Convolution Layer.
    Expanded Fourier mode capacity preserves sharp convective gradients.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Height (Z) modes
        self.modes2 = modes2 # Lat (H) modes
        self.modes3 = modes3 # Lon (W) modes

        self.scale = 1.0 / (in_channels * out_channels)
        
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def _compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_float = x.to(torch.float32)
        x_ft = torch.fft.rfftn(x_float, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batchsize, self.out_channels, 
            x.size(-3), x.size(-2), x.size(-1) // 2 + 1, 
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self._compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self._compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self._compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self._compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x_out = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x_out.to(x.dtype)

class FNO3D_Radar(nn.Module):
    def __init__(self, in_steps=2, out_steps=2, channels=2, width=32, modes1=12, modes2=20, modes3=20):
        super(FNO3D_Radar, self).__init__()
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.in_channels = in_steps * channels
        self.out_channels = out_steps * channels

        self.fc0 = nn.Conv3d(self.in_channels, width, 1)

        # 3D Spectral Layers with increased mode truncation limits
        self.conv0 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv3 = SpectralConv3d(width, width, modes1, modes2, modes3)

        self.w0 = nn.Conv3d(width, width, 1)
        self.w1 = nn.Conv3d(width, width, 1)
        self.w2 = nn.Conv3d(width, width, 1)
        self.w3 = nn.Conv3d(width, width, 1)

        self.fc1 = nn.Conv3d(width, 128, 1)
        self.fc2 = nn.Conv3d(128, self.out_channels, 1)

    def forward(self, x):
        B, T, C, Z, H, W = x.shape
        x = x.view(B, T * C, Z, H, W)

        x = self.fc0(x)

        x1 = self.conv0(x) + self.w0(x)
        x = F.gelu(x1)

        x1 = self.conv1(x) + self.w1(x)
        x = F.gelu(x1)

        x1 = self.conv2(x) + self.w2(x)
        x = F.gelu(x1)

        x1 = self.conv3(x) + self.w3(x)
        x = F.gelu(x1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        out = x.view(B, self.out_steps, C, Z, H, W)
        return out