
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.image_encoder = models.resnet18()
        self.image_encoder.fc = nn.Identity()

        self.image_out = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256)
        )

        self.text_encoder = AutoModel.from_pretrained("dbmdz/distilbert-base-turkish-cased")
        self.target_token_idx = 0


        self.text_out = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 256)
        )


    def forward(self, image, text, mask):

        image_vec = self.image_encoder(image)

        image_vec = self.image_out(image_vec.view(-1,512))

        text_out = self.text_encoder(text, mask)
        last_hidden_states = text_out.last_hidden_state

        last_hidden_states = last_hidden_states[:,self.target_token_idx,:]

        text_vec = self.text_out(last_hidden_states.view(-1,768))

        return image_vec, text_vec
