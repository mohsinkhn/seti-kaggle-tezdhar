import timm
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers.weight_init import trunc_normal_


class SimpleBB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=1, num_classes=1)
        # del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.fc1(self.drop(self.flat(self.avg(x))))


class FeaturesBB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.flat(self.avg(x))


class FasterBB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=1, num_classes=1)
        self.model.blocks[0][0].conv_dw.stride = (2, 2)
        # del self.model.classifier
        # self.pool = nn.MaxPool2d((1, 2), (1, 1), padding=0)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        #del self.model.conv_head, self.model.bn2
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.fc1(self.drop(self.flat(self.avg(x))))


class Simple3BB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.fc1(self.drop(self.flat(self.avg(x))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Hybrid3BB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # del self.model.classifier
        dim1 = self.model.num_features
        dim2 = 256
        self.fc0 = nn.Linear(dim1, dim2)
        self.attn = Attention(dim2, 1)
        self.norm1 = nn.LayerNorm(dim2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, dim2))

        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(dim2, 1)
        self.init_weights()

    def init_weights(self):
        head_bias = 0.
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x = self.model.forward_features(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc0(x)
        x = x + self.drop(self.attn(self.norm1(x + self.pos_embed)))
        x = x.mean(1)
        return self.fc1(self.drop(x))


class Hybrid4BB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # del self.model.classifier
        dim1 = self.model.num_features
        dim2 = 512
        self.fc0 = nn.Linear(dim1, dim2)
        self.attn = Attention(dim2, 2)
        self.norm1 = nn.LayerNorm(dim2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, dim2))

        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(dim2, 1)
        self.init_weights()

    def init_weights(self):
        head_bias = 0.
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x = self.model.forward_features(x)
        b, c, h, w = x.shape
        x = x.mean(3)
        x = x.view(b, c, -1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc0(x)
        x = x + self.drop(self.attn(self.norm1(x + self.pos_embed)))
        x = x.mean(1)
        return self.fc1(self.drop(x))


class Hybrid5BB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # del self.model.classifier
        dim1 = self.model.num_features
        dim2 = 512
        self.fc0 = nn.Linear(dim1, dim2)
        self.attn = Attention(dim2, 2)
        self.norm1 = nn.LayerNorm(dim2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 24, dim2))

        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(dim2, 1)
        self.init_weights()

    def init_weights(self):
        head_bias = 0.
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x = self.model.forward_features(x)
        b, c, h, w = x.shape
        x = x.mean(3)
        x = x.view(b, c, -1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc0(x)
        x = x + self.drop(self.attn(self.norm1(x + self.pos_embed)))
        x = x.mean(1)
        return self.fc1(self.drop(x))


class Hybrid5eBB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # del self.model.classifier
        dim1 = self.model.num_features
        dim2 = 512
        self.fc0 = nn.Linear(dim1, dim2)
        self.attn = Attention(dim2, 2)
        self.norm1 = nn.LayerNorm(dim2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 24 * 16, dim2))

        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(dim2, 1)
        self.init_weights()

    def init_weights(self):
        head_bias = 0.
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x = self.model.forward_features(x)
        b, c, h, w = x.shape
        # x = x.mean(3)
        x = x.view(b, c, -1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc0(x)
        x = x + self.drop(self.attn(self.norm1(x + self.pos_embed)))
        x = x.mean(1)
        return self.fc1(self.drop(x))


class Hybrid6BB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # del self.model.classifier
        dim1 = self.model.num_features
        dim2 = 512
        self.fc0 = nn.Linear(dim1, dim2)
        self.attn = Attention(dim2, 2)
        self.norm1 = nn.LayerNorm(dim2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 32, dim2))

        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(dim2, 1)
        self.init_weights()

    def init_weights(self):
        head_bias = 0.
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x = self.model.forward_features(x)
        b, c, h, w = x.shape
        x = x.mean(3)
        x = x.view(b, c, -1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc0(x)
        x = x + self.drop(self.attn(self.norm1(x + self.pos_embed)))
        x = x.mean(1)
        return self.fc1(self.drop(x))


class HybridFeatures(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # del self.model.classifier
        dim1 = self.model.num_features
        dim2 = 256
        self.fc0 = nn.Linear(dim1, dim2)
        self.attn = Attention(dim2, 1)
        self.norm1 = nn.LayerNorm(dim2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, dim2))

        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(dim2, 1)
        self.init_weights()

    def init_weights(self):
        head_bias = 0.
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x = self.model.forward_features(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc0(x)
        x = x + self.drop(self.attn(self.norm1(x + self.pos_embed)))
        x = x.mean(1)
        return x


class Simple3Features(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.flat(self.avg(x))


class MultiBB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=3, num_classes=1)
        # 128 * 512 x 6
        dim1 = self.model.num_features
        dim2 = 512
        self.fc0 = nn.Linear(dim1, dim2)
        self.attn = Attention(dim2, 2)
        self.norm1 = nn.LayerNorm(dim2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 384, dim2))

        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc1 = nn.Linear(dim2, 1)
        self.init_weights()

    def init_weights(self):
        head_bias = 0.
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        b, n, c, h, w = x.shape
        outs = []
        for i in range(n):
            out = self.model.forward_features(x[:, i])  # b x c x 4 x 16
            out = out.view(out.shape[0], out.shape[1], -1)  # b x c x 64
            out = out.permute(0, 2, 1).contiguous()
            out = self.fc0(out)  # b x 64 x 512
            outs.append(out)

        x = torch.cat(outs, 1)  # b x 384 x 512
        x = x + self.drop(self.attn(self.norm1(x + self.pos_embed)))
        x = x.mean(1)
        return self.fc1(self.drop(x))


class BackgroundAttenuation(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=1, num_classes=1)
        del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(hparams['dropout'])
        self.bn2 = nn.BatchNorm2d(320)
        self.fc1 = nn.Linear(320, 1)

    def forward(self, x1, x2):
        x1 = self.model.conv_stem(x1)
        x2 = self.model.conv_stem(x2)
        x2attn = torch.sigmoid(x2)
        x1 = x1 * (1 - x2attn)
        x1 = self.model.blocks(self.model.act1(self.model.bn1(x1)))
        x2 = self.model.blocks(self.model.act2(self.model.bn1(x2)))
        x1 = x1 * (1 - torch.sigmoid(x2))
        x1 = self.model.global_pool(self.model.act2(self.bn2(x1)))
        return self.fc1(self.drop(x1))


class TfmBB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, num_classes=1)

    def forward(self, x):
        return self.model(x)


class Ch2(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=2, num_classes=1)

    def forward(self, x):
        return self.model(x)


class SimpleStride1(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=1, num_classes=1)
        self.model.conv_stem.stride = (1, 1)
        # del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.fc1(self.flat(self.avg(x)))


class SimpleSum(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=1, num_classes=1)
        self.model.conv_stem.stride = (2, 1)
        del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.mean(2).mean(2)
        return self.fc1(x)


class EffMod(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, in_chans=1, num_classes=1)
        del self.model.classifier, self.model.conv_head
        self.fc1 = nn.Linear(320, 1)

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.blocks(x)
        return self.fc1(self.model.global_pool(x))


class SimpleBB2(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model('resnest14d', pretrained=True, in_chans=1, num_classes=1)
        self.model.conv1.kernel = (11, 3)
        # del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.fc1(self.flat(self.avg(x)))


class MaxMeanBB(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=1, num_classes=1)
        del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.max(2)[0]
        x = x.mean(2)
        return self.fc1(self.flat(x))


class Ch3(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, num_classes=1)
        del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=hparams['dropout'])
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.fc1(self.drop(self.flat(self.avg(x))))


class Ch6(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, num_classes=1)
        del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=hparams['dropout'])
        self.fc1 = nn.Linear(self.model.num_features * 2, 1)

    def forward(self, x):
        x1 = self.model.forward_features(x[:, :3])
        x2 = self.model.forward_features(x[:, 3:])
        x1 = self.flat(self.avg(x1) + self.max(x1))
        x2 = self.flat(self.avg(x2) + self.max(x2))
        x = torch.cat((x1, x2), -1)
        return self.fc1(self.drop(x))


class SetiCNN9(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(34, 3), stride=(2, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act1 = nn.SiLU(inplace=True)
        block1 = []
        for _ in range(3):
            layer = nn.Sequential(
                nn.Conv2d(24, 24, kernel_size=(7, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True)
            )
            block1.append(layer)
        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(
            nn.Conv2d(24, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(144, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True)
        )
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.mean(x)
        return self.fc(self.flat(x))


class Ch3Effb7stem(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        model1 = timm.create_model('tf_efficientnetv2_l_in21k')
        self.model = timm.create_model(hparams['backbone'], pretrained=True, num_classes=1)
        self.model.conv_stem = model1.conv_stem
        self.bn1 = model1.bn1
        del model1
        del self.model.classifier
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.fc1(self.flat(self.avg(x)))


class Ch6(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, num_classes=1, in_chans=6)

    def forward(self, x):
        x = self.model(x)
        return x


class Ch9(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = timm.create_model(hparams['backbone'], pretrained=True, in_chans=9)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(hparams['dropout'])
        self.fc = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.mean(x)
        x = x.view(x.size()[0], -1)
        x = self.drop(x)
        return self.fc(x)
