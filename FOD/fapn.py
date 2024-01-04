import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d


class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2,  g* 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2*2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)
    
    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)
        
        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up*2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm
    
class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )
class FaPNFusion(nn.Module):
    def __init__(self, in_channels,channel):
        super().__init__()
        self.align_modules = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])
        for ch in in_channels:
            self.align_modules.append(FAM(ch,channel))
            self.output_convs.append(ConvModule(channel,channel,3,1,1))
    def forward(self,xfeatures,yfeatures):
        outs= []
        for x,y,align_module,output_conv in zip(xfeatures,yfeatures,self.align_modules,self.output_convs):
            out = align_module(x,y)
            out = output_conv(out)
            outs.append(out)
        return outs

class FaPNHead(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        in_channels = in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], channel, 1)])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[1:]:
            self.align_modules.append(FAM(ch, channel))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        #self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.align_modules[0](features[0])
        
        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat, out)
            out = output_conv(out)
        out = self.dropout(out)
        #out = self.conv_seg(self.dropout(out))
        out = nn.functional.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        return out

class TFFAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c2, c2)
        self.lateral_ref_conv = FSM(c1,c2)
        self.offset = nn.Conv2d(c2*2, c2, 1, bias=False) # offset할때 c2두배로하거나 res channel 가져가거나 두가지방법있음
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)
    
    def forward(self, feat_l, feat_ref, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)
        
        feat_arm = self.lateral_conv(feat_l)
        feat_ref_arm = self.lateral_ref_conv(feat_ref)

        offset = self.offset(torch.cat([feat_ref_arm, feat_up*2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm

class TFFaPNHead(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        # in_channels = [256,512,1024,2048]
        in_channels = in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(channel, channel, 1)])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[1:]:
            self.align_modules.append(TFFAM(ch, channel))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        #self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features_tf,features_res) -> Tensor:
        features_tf = features_tf[::-1]
        features_res = features_res[::-1]

        out = self.align_modules[0](features_tf[0])
        
        for feat_tf, feat_res, align_module, output_conv in zip(features_tf[1:], features_res[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat_tf,feat_res, out)
            out = output_conv(out)
        out = self.dropout(out)
        #out = self.conv_seg(self.dropout(out))
        out = nn.functional.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        return out
    

    
class TFFAMFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        #self.lateral_conv = FSM(c2, c2)
        self.lateral_ref_conv = FSM(c1,c2)
        self.offset = nn.Conv2d(c2*2, c2, 1, bias=False) # offset할때 c2두배로하거나 res channel 가져가거나 두가지방법있음
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)
    
    def forward(self, feat_ref, feat_s):
        feat_up = feat_s
        if feat_ref.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_ref.shape[2:], mode='bilinear', align_corners=False)
        
        feat_ref_arm = self.lateral_ref_conv(feat_ref)

        offset = self.offset(torch.cat([feat_ref_arm, feat_up*2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align
class TFFaPNFusionHead(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        from FOD.Fusion import Fusion
        # in_channels = [256,512,1024,2048]
        in_channels = in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(channel, channel, 1)])
        self.fusions=[]

        for ch in in_channels[1:]:
            self.align_modules.append(TFFAMFusion(ch, channel))
            self.fusions.append(Fusion(256))
        self.fusions = nn.ModuleList(self.fusions)

    def forward(self, features_tf,features_res) -> Tensor:
        features_tf = features_tf[::-1]
        features_res = features_res[::-1]

        out = self.align_modules[0](features_tf[0])

        i=0
        for feat_tf, feat_res, align_module in zip(features_tf[1:], features_res[1:], self.align_modules[1:]):
            out = align_module(feat_res, out)
            out = self.fusions[i](feat_tf,out)
            i+=1

        return out
    
if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))
    from backbones.resnet import ResNet
    backbone = ResNet('50')
    head = FaPNHead([256, 512, 1024, 2048], 128, 19)
    x = torch.randn(2, 3, 384, 384)
    features = backbone(x)
    out = head(features)
    out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
    print(out.shape)