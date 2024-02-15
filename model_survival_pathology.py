from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from grid_attention_layer import GridAttentionBlock3D_TORR as AttentionBlock3D
from visualize_attention import HookBasedFeatureExtractor

class MTCNN3Dpathology(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(MTCNN3Dpathology, self).__init__()
        self.image_encoder = VolumeEncoder(n_latents)
        self.n_latents     = n_latents

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    def forward(self, image=None):
        '''
        predictions, attention  = self.infer(image)
        return predictions, attention 
        '''
        predictions  = self.infer(image)
        return predictions 

    def infer(self, image=None): 
        batch_size = image.size(0) 
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA

        preds, attention = self.image_encoder(image)

        return preds, attention 


class VolumeEncoder(nn.Module):
    """Parametrizes q(z|x).

    This is the standard DCGAN architecture.

    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, n_latents):
        super(VolumeEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 4, 2, 1, bias=False)
        self.activation1 = Swish()
        self.conv2 = nn.Conv3d(32, 64, 4, 2, 1, bias=False)
        self.activation2 = Swish()
        self.conv3 = nn.Conv3d(64, 128, 4, 2, 1, bias=False)
        self.activation3 = Swish()
        self.conv4 = nn.Conv3d(128, 256, 4, 1, 0, bias=False)
        self.activation4 = Swish()

        #TODO, change size 256 * 44 * 44 * 37
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5 * 1, 512),
            Swish(),
            nn.Dropout(p=0.1),

            nn.Linear(512, 32),
            Swish(),
            nn.Dropout(p=0.1),

            )
        self.fc20 = nn.Linear(32, 2)  
        self.fc21 = nn.Linear(32, 2)  
        self.fc22 = nn.Linear(32, 2)  
        self.fc23 = nn.Linear(32, 2)  
        self.fc24 = nn.Linear(32, 2)  
        self.fc25 = nn.Linear(32, 2)  
        self.fc26 = nn.Linear(32, 1)  


        self.n_latents = n_latents
        feature_scale = 2 
        #filters = [32, 64, 128, 256]
        filters = [64, 128, 256, 512]
        filters = [int(x / feature_scale) for x in filters]
        #filters = [16, 32, 64, 128] when feature_scale = 4 
        #filters = [32, 64, 128, 256] when feature_scale = 2 
        #The in_channels used to have filters[1]
        self.compatibility_score1 = AttentionBlock3D(in_channels=filters[2], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1,1, 1),
                                                     mode='concatenation_softmax')

    def forward(self, x):
        n_latents = self.n_latents
        #feature = self.features(x)
        conv1 = self.conv1(x)
        activation1 = self.activation1(conv1)
        conv2 = self.conv2(activation1)
        activation2 = self.activation2(conv2)
        conv3 = self.conv3(activation2)
        activation3 = self.activation3(conv3)
        conv4 = self.conv4(activation3)
        feature = self.activation4(conv4)

        #print(conv3.shape)
        #print(conv4.shape)
        g_conv1, att1 = self.compatibility_score1(conv3, conv4)
        '''
        print('att1  = ')
        print(att1)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        '''
        #print(feature.shape)         

        #TODO, change size 256 * 44 * 44 * 37
        #x = feature.view(-1, 18337792)
        x = feature.view(-1, 256 * 5 * 5 * 1)
        #x = feature.view(-1, 256 * 5 * 5)
        out = self.classifier(x)
        out0 = F.softmax(self.fc20(out), dim = 1)
        out1 = F.softmax(self.fc21(out), dim = 1)
        out2 = F.softmax(self.fc22(out), dim = 1)
        out3 = F.softmax(self.fc23(out), dim = 1)
        out4 = F.softmax(self.fc24(out), dim = 1)
        out5 = F.softmax(self.fc25(out), dim = 1)
        out6 = self.fc26(out)

        return [out0, out1, out2, out3, out4, out5, out6], att1 



class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)


