import torch.nn as nn
import torch
from torchvision.models import vgg16

from modeling import uformer


class network(nn.Module):
    def __init__(self, inchnls=3, outchnls=3, initialchnls=16, rows=3, columns=6, norm=False, device='cuda'):
        """ GridNet constructor.
        Args:
          inchnls: input channels; default is 3.
          outchnls: output channels; default is 3.
          initialchnls: initial number of feature channels; default is 16.
          rows: number of rows; default is 3.
          columns: number of columns; default is 6 (should be an even number).
          norm: apply batch norm as used in Ref. 1; default is False (i.e., Ref. 2)
        """

        super(network, self).__init__()
        assert columns % 2 == 0, 'use even number of columns'
        assert columns > 1, 'use number of columns > 1'
        assert rows > 1, 'use number of rows > 1'

        self.device = device

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.rows = rows
        self.columns = columns

        # encoder
        for r in range(rows):
            res_blocks = nn.ModuleList([])
            down_blocks = nn.ModuleList([])
            for c in range(int(columns / 2)):
                if r == 0:
                    if c == 0:
                        res_blocks.append(ForwardBlock(in_dim=inchnls,
                                                       out_dim=initialchnls,
                                                       norm=norm).to(device=self.device))
                    else:
                        res_blocks.append(ResidualBlock(in_dim=initialchnls, norm=norm).to(
                            device=self.device))
                    down_blocks.append(SubsamplingBlock(
                        in_dim=initialchnls, norm=norm).to(device=self.device))
                else:
                    if c > 0:
                        res_blocks.append(ResidualBlock(
                            in_dim=initialchnls * (2 ** r), norm=norm).to(
                            device=self.device))
                    else:
                        res_blocks.append(nn.ModuleList([]))
                    if r < (rows - 1):
                        down_blocks.append(SubsamplingBlock(
                            in_dim=initialchnls * (2 ** r), norm=norm).to(
                            device=self.device))
                    else:
                        down_blocks.append(nn.ModuleList([]))

            self.encoder.append(res_blocks)
            self.encoder.append(down_blocks)

        # decoder
        for r in range((rows - 1), -1, -1):
            res_blocks = nn.ModuleList([])
            up_blocks = nn.ModuleList([])
            for c in range(int(columns / 2), columns):
                if r == 0:
                    res_blocks.append(ResidualBlock(in_dim=initialchnls,
                                                    norm=norm).to(device=self.device))
                    up_blocks.append(nn.ModuleList([]))
                elif r > 0:
                    res_blocks.append(ResidualBlock(
                        in_dim=initialchnls * (2 ** r), norm=norm).to(
                        device=self.device))
                    up_blocks.append(UpsamplingBlock(
                        in_dim=initialchnls * (2 ** r), norm=norm).to(
                        device=self.device))

            self.decoder.append(res_blocks)
            self.decoder.append(up_blocks)

        self.output = ForwardBlock(in_dim=initialchnls, out_dim=outchnls,
                                   norm=norm).to(device=self.device)

    def forward(self, x):
        """ Forward function

    Args:
      x: input image

    Returns:
      output: output image
    """
        latent_downscaled = []
        latent_upscaled = []
        latent_forward = []

        for i in range(0, len(self.encoder), 2):
            res_blcks = self.encoder[i]
            branch_blcks = self.encoder[i + 1]
            if not branch_blcks[0]:
                not_last = False
            else:
                not_last = True
            for j, (res_blck, branch_blck) in enumerate(zip(res_blcks, branch_blcks)):
                if i == 0 and j == 0:
                    x_latent = res_blck(x)
                elif i == 0:
                    x_latent = res_blck(x_latent)
                elif j == 0:
                    x_latent = latent_downscaled[j]
                else:
                    x_latent = res_blck(x_latent)
                    x_latent = x_latent + latent_downscaled[j]
                if i == 0:
                    latent_downscaled.append(branch_blck(x_latent))
                elif not_last:
                    latent_downscaled[j] = branch_blck(x_latent)
            latent_forward.append(x_latent)

        latent_forward.reverse()

        for k, i in enumerate(range(0, len(self.decoder), 2)):
            res_blcks = self.decoder[i]
            branch_blcks = self.decoder[i + 1]
            if not branch_blcks[0]:
                not_last = False
            else:
                not_last = True
            for j, (res_blck, branch_blck) in enumerate(zip(res_blcks, branch_blcks)):
                if j == 0:
                    latent_x = latent_forward[k]
                x_latent = res_blck(latent_x)
                if i > 0:
                    x_latent = x_latent + latent_upscaled[j]
                if i == 0:
                    latent_upscaled.append(branch_blck(x_latent))
                elif not_last:
                    latent_upscaled[j] = branch_blck(x_latent)

        output = self.output(x_latent)
        return output


class SubsamplingBlock(nn.Module):
    """ SubsamplingBlock"""

    def __init__(self, in_dim, norm=False):
        super(SubsamplingBlock, self).__init__()
        self.output = None
        if norm:
            self.block = nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.PReLU(init=0.25),
                nn.Conv2d(in_dim, int(in_dim * 2), kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(int(in_dim * 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(in_dim * 2), int(in_dim * 2), kernel_size=3, padding=1))
        else:
            self.block = nn.Sequential(
                nn.PReLU(init=0.25),
                nn.Conv2d(in_dim, int(in_dim * 2), kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(in_dim * 2), int(in_dim * 2), kernel_size=3, padding=1))

    def forward(self, x):
        return self.block(x)


class UpsamplingBlock(nn.Module):
    """ UpsamplingBlock"""

    def __init__(self, in_dim, norm=False):
        super(UpsamplingBlock, self).__init__()
        self.output = None
        if norm:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                nn.BatchNorm2d(in_dim),
                nn.PReLU(init=0.25),
                nn.Conv2d(in_dim, int(in_dim / 2), kernel_size=3, padding=1),
                nn.BatchNorm2d(int(in_dim / 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(in_dim / 2), int(in_dim / 2), kernel_size=3, padding=1))
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                nn.PReLU(init=0.25),
                nn.Conv2d(in_dim, int(in_dim / 2), kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(in_dim / 2), int(in_dim / 2), kernel_size=3, padding=1))

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """ ResidualBlock"""

    def __init__(self, in_dim, out_dim=None, norm=False):
        super(ResidualBlock, self).__init__()
        self.output = None
        intermediate_dim = int(in_dim * 2)
        if out_dim is None:
            out_dim = in_dim
        if norm:
            self.block = nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.PReLU(init=0.25),
                nn.Conv2d(in_dim, intermediate_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(intermediate_dim),
                nn.PReLU(init=0.25),
                nn.Conv2d(intermediate_dim, out_dim, kernel_size=3, padding=1))
        else:
            self.block = nn.Sequential(
                nn.PReLU(init=0.25),
                nn.Conv2d(in_dim, intermediate_dim, kernel_size=3, padding=1),
                nn.PReLU(init=0.25),
                nn.Conv2d(intermediate_dim, out_dim, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.block(x)


class ForwardBlock(nn.Module):
    """ ForwardBlock"""

    def __init__(self, in_dim, out_dim=None, norm=False):
        super(ForwardBlock, self).__init__()
        self.output = None
        intermediate_dim = int(in_dim * 2)
        if out_dim is None:
            out_dim = in_dim
        if norm:
            self.block = nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.PReLU(init=0.25),
                nn.Conv2d(in_dim, intermediate_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(intermediate_dim),
                nn.PReLU(init=0.25),
                nn.Conv2d(intermediate_dim, out_dim, kernel_size=3, padding=1))
        else:
            self.block = nn.Sequential(
                nn.PReLU(init=0.25),
                nn.Conv2d(in_dim, intermediate_dim, kernel_size=3, padding=1),
                nn.PReLU(init=0.25),
                nn.Conv2d(intermediate_dim, out_dim, kernel_size=3, padding=1))

    def forward(self, x):
        return self.block(x)


class WBnet(nn.Module):
    def __init__(self, inchnls=9, initialchnls=8, rows=4, columns=6,
                 norm=False, device='cuda'):
        """ Network constructor.
    """
        self.outchnls = int(inchnls / 3)
        self.inchnls = inchnls
        self.device = device
        super(WBnet, self).__init__()
        assert columns % 2 == 0, 'use even number of columns'
        assert columns > 1, 'use number of columns > 1'
        assert rows > 1, 'use number of rows > 1'
        self.net = network(inchnls=self.inchnls, outchnls=self.outchnls,
                                   initialchnls=initialchnls, rows=rows, columns=columns,
                                   norm=norm, device=self.device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ Forward function"""
        weights = self.net(x)
        weights = torch.clamp(weights, -1000, 1000)
        weights = self.softmax(weights)
        out_img = torch.unsqueeze(weights[:, 0, :, :], dim=1) * x[:, :3, :, :]
        for i in range(1, int(x.shape[1] // 3)):
            out_img += torch.unsqueeze(weights[:, i, :, :],
                                       dim=1) * x[:, (i * 3):3 + (i * 3), :, :]
        return out_img, weights


class WBnetUformer(nn.Module):
    def __init__(self, inchnls=9, initialchnls=16, ps=64, rows=4, columns=6, norm=False, style_remover="efdm", device='cuda'):
        """ Network constructor.
        """
        self.outchnls = int(inchnls / 3)
        self.inchnls = inchnls
        self.device = device
        super(WBnetUformer, self).__init__()
        assert columns % 2 == 0, 'use even number of columns'
        assert columns > 1, 'use number of columns > 1'
        assert rows > 1, 'use number of rows > 1'
        # self.net = gridnet.network(inchnls=self.inchnls, outchnls=self.outchnls,
        #                    initialchnls=initialchnls, rows=rows, columns=columns,
        #                    norm=norm, device=self.device)

        vgg_feats = vgg16(pretrained=True).features.eval().cuda()
        vgg_feats = nn.Sequential(*[module for module in vgg_feats][:35]).eval()
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.net = uformer.StyleUformer(vgg_feats=vgg_feats, img_size=ps, in_chans=inchnls, dd_in=inchnls, embed_dim=initialchnls, depths=depths, style_proj_n_ch=128, win_size=8, mlp_ratio=4., 
                                        token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False, style_remover=style_remover).to(self.device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ Forward function"""
        weights = self.net(x)
        weights = torch.clamp(weights, -1000, 1000)
        weights = self.softmax(weights)
        out_img = torch.unsqueeze(weights[:, 0, :, :], dim=1) * x[:, :3, :, :]
        for i in range(1, int(x.shape[1] // 3)):
            out_img += torch.unsqueeze(weights[:, i, :, :],
                                       dim=1) * x[:, (i * 3):3 + (i * 3), :, :]
        return out_img, weights


def build_model(wb_model_path, wb_settings, device):
    wb_network = WBnetUformer(device=device, inchnls=3 * len(wb_settings), ps=128)
    state_dict = torch.load(wb_model_path, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    wb_network.load_state_dict(new_state_dict)
    wb_network.to(device=device)
    wb_network.eval()
    return wb_network


if __name__ == '__main__':
    x = torch.rand(8, 15, 64, 64).cuda()
    net = WBnet(15, 32)
    y, w = net(x)
    print(y.shape, w.shape)
