import torch
import torch.nn as nn
from torch.nn import functional as F
import torchmetrics
import torchvision.models as models
from pytorch_lightning import LightningModule

class KelpClassifier(LightningModule):
    def __init__(self, backbone_name: str = "inception_v3", optimizer_name: str = "AdamW"):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.backbone_name = backbone_name
        backbone = getattr(models, backbone_name)(weights='DEFAULT')
        
        #implementing inception_v3
        self.model = backbone
        self.model.aux_logits = False
        self.model.fc = nn.Sequential(
        nn.Linear(self.model.fc.in_features, 2)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    