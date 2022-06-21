
from random import random
import numpy as np

import torch
from torch import nn

from torchvision.models import resnet18

from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack

def position_encoding(feature_size, d_model, device):

    pe = torch.zeros(feature_size, d_model, device=device)
    
    pos = torch.arange(0, feature_size, device=device).unsqueeze(1)
    _2i = torch.arange(0, d_model, step=2, device=device)

    pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    return pe[:feature_size, :]

class ResNet18(nn.Module):
    
    def __init__(
        self,
    ):

        super(ResNet18, self).__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=2, stride=2,),
            nn.Dropout2d(p=.3),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),

            nn.MaxPool2d(kernel_size=2, stride=2,),
            nn.Dropout2d(p=.3),

        )

    def forward(self, x):
        x = self.layers(x)
        return x

class AttentionBasedDecoder(nn.Module):

    def __init__(
        self,
        shared: nn.Embedding = None,
        lm_head: nn.Linear = None,
        d_model: int = None,
    ):

        super(AttentionBasedDecoder, self).__init__()

        self.rnn = nn.LSTMCell(d_model*2, d_model, bias=False,)

        self.w_e = nn.Linear(d_model, d_model, bias=False)
        self.w_h = nn.Linear(d_model, d_model, bias=False)
        self.w_f = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_s = nn.Linear(d_model, d_model, bias=False)

        self.g = nn.Linear(d_model*2, d_model, bias=False)

        self.shared = shared
        self.lm_head = lm_head

    def forward(self, f, decoder_input_ids):

        # features: (batch_size, feature_size, d_model)
        # decoder_input_ids: (batch_size, max_length)

        batch_size, feature_size, d_model = f.size()
        _, max_length = decoder_input_ids.size()
        device = f.get_device()

        s = torch.zeros_like(f)

        q = position_encoding(feature_size, d_model, device)

        decoder_inputs = self.shared(decoder_input_ids)

        h = torch.zeros(batch_size, d_model, device=device)
        c_prime = torch.zeros(batch_size, d_model, device=device)
        cell_state = torch.zeros(batch_size, d_model, device=device)

        predictions = None

        for t in range(max_length):

            h, cell_state = self.rnn(torch.cat((decoder_inputs[:, t], c_prime), -1), (h, cell_state)) # Eq.7

            e = self.w_e(torch.tanh(self.w_h(h).unsqueeze(1) + self.w_f(f) + self.w_q(q).unsqueeze(0) + self.w_s(s))) # Eq. 3

            alpha = torch.softmax(e, dim=-1) # Eq. 4

            s = s + alpha

            c = (alpha * f).sum(1) # Eq. 2
            c_prime = (alpha * (f + q)).sum(1) # Eq. 8
            
            # c: context vector
            # h: current hidden state
            prediction = self.g(torch.cat((c, h), -1)) # Eq. 1

            if predictions is None:
                predictions = prediction.unsqueeze(1)
            else:
                predictions = torch.cat((predictions, prediction.unsqueeze(1)), 1)

        return predictions

    def generate(self, f, bos_token_ids, max_length):

        # features: (batch_size, feature_size, d_model)
        # decoder_input_ids: (batch_size)

        batch_size, feature_size, d_model = f.size()
        device = f.get_device()

        s = torch.zeros_like(f)

        q = position_encoding(feature_size, d_model, device)

        decoder_inputs = self.shared(bos_token_ids)
        # decoder_inputs: (batch, d_model)

        h = torch.zeros(batch_size, d_model, device=device)
        c_prime = torch.zeros(batch_size, d_model, device=device)
        cell_state = torch.zeros(batch_size, d_model, device=device)

        predictions = None

        for t in range(max_length):

            e = self.w_e(torch.tanh(self.w_h(h).unsqueeze(1) + self.w_f(f) + self.w_q(q).unsqueeze(0) + self.w_s(s))) # Eq. 3

            alpha = torch.softmax(e, dim=-1) # Eq. 4

            s = s + alpha

            c = (alpha * f).sum(1) # Eq. 2
            c_prime = (alpha * (f + q)).sum(1) # Eq. 8

            h, cell_state = self.rnn(torch.cat((decoder_inputs, c_prime), -1), (h, cell_state)) # Eq.7
            
            # c: context vector
            # h: current hidden state
            prediction = self.g(torch.cat((c, h), -1)) # Eq. 1

            if predictions is None:
                predictions = prediction.unsqueeze(1)
            else:
                predictions = torch.cat((predictions, prediction.unsqueeze(1)), 1)

        return predictions

class MERecognitionModel(nn.Module):

    def __init__(
        self,
        vocab_size: int = None,
        d_model: int = None,
        num_layers: int = None,
    ):

        super(MERecognitionModel, self).__init__()

        self.shared = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.encoder = ResNet18()
        self.decoder = AttentionBasedDecoder(shared=self.shared, d_model=d_model)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

    def forward(self, x, decoder_input_ids):

        # x: (batch, n_channels, height, width)
        
        batch_size = x.size(0)

        features = self.encoder(x)

        features = features.view(batch_size, self.d_model, -1)
        features = features.reshape(batch_size, -1, self.d_model)

        decoder_output = self.decoder(
            features, decoder_input_ids
        )

        lm_logits = self.lm_head(decoder_output)

        return lm_logits

    def generate(self, x, decoder_input_ids):

        batch_size = x.size(0)

        features = self.encoder(x)

        features = features.view(batch_size, self.d_model, -1)
        features = features.reshape(batch_size, -1, self.d_model)

        decoder_output = self.decoder.generate(
            features, decoder_input_ids[:, 0], decoder_input_ids.size(1)
        )

        lm_logits = self.lm_head(decoder_output)

        return lm_logits