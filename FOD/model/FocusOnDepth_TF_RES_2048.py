import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from FOD.Reassemble import Reassemble
from FOD.Fusion import Fusion, FeatureFusion
from FOD.Head import HeadDepth, HeadSeg
import json
torch.manual_seed(0)
with open('config.json', 'r') as f:
    config = json.load(f)
mode = config['General']['type']

class FocusOnDepth(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 1024,
                 resample_dim       = 256,
                 read               = 'projection',
                 num_layers_encoder = 24,
                 hooks              = [5, 11, 17, 23],
                 reassemble_s       = [4,8,16,32],
                 transformer_dropout= 0,
                 nclasses           = 2,
                 type               = mode,
                 model_timm         = "vit_large_patch16_384",
                 model              = "TF",
                 resnet_type        = "resnet50",):
        """
        neck : "simple","bifpn"
        head : "simple", "fapn"
        model : "T","R","TR"
        feature_fusion : "simple", "fapn"
        resnet_type : 50, ~
        """    

        """
        Focus on Depth
        type : {"full", "depth", "segmentation"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        #Splitting img into patches
        # channels, image_height, image_width = image_size
        # assert image_height % patch_size == 0 and image_width % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # num_patches = (image_height // patch_size) * (image_width // patch_size)
        # patch_dim = channels * patch_size * patch_size
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear(patch_dim, emb_dim),
        # )
        # #Embedding
        # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        #Transformer
        # encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dropout=transformer_dropout, dim_feedforward=emb_dim*4)
        # self.transformer_encoders = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_encoder)
        self.model = model
        self.type_ = type
        self.len_layer = len(reassemble_s)

        #build model
        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)
        self.resnet = timm.create_model(resnet_type,pretrained=True)

        #Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)
        self.activation_res = {}
        self.hooks_res = [1,2,3,4]
        self._get_layers_from_hooks_res(self.hooks_res)
        
        #Reassembles
        emb_dims = [256,512,1024,2048]
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.fusions.append(Fusion(resample_dim))
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        self.feature_fusion = FeatureFusion(emb_dims,256,True)

        if type == "full":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        else:
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

    def forward(self, img):
        # x = self.to_patch_embedding(img)
        # b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # t = self.transformer_encoders(x)
        tf_list =[]
        res_list = []
        len_fusion = self.len_layer

        # backbones
        t = self.transformer_encoders(img)
        for i in np.arange(0,len_fusion):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            tf_list.append(self.reassembles[i](activation_result))

        r = self.resnet(img)
        for i in np.arange(0,len_fusion):
            hook_to_take = 'r'+str(self.hooks_res[i])
            activation_result = self.activation_res[hook_to_take]
            res_list.append(activation_result)
        
        reassemble_list = self.feature_fusion(tf_list,res_list)

        previous_stage=None
        for i in np.arange(len_fusion-1,-1,-1):
            fusion_result = self.fusions[i](reassemble_list[i], previous_stage)
            previous_stage = fusion_result
        
        out_depth = None
        out_segmentation = None
        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation != None:
            out_segmentation = self.head_segmentation(previous_stage)
        return out_depth, out_segmentation

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            #self.transformer_encoders.layers[h].register_forward_hook(get_activation('t'+str(h)))
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))

    def _get_layers_from_hooks_res(self,hooks):
        def get_activation_res(name):
            def hook(model, input, output):
                self.activation_res[name] = output
            return hook
        for h in hooks:
            eval("self.resnet.layer"+str(h)).register_forward_hook(get_activation_res('r'+str(h)))