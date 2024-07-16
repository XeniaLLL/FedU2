import copy

import torch
from my.model.resnet import ResNet18
class SimsiamModel(torch.nn.Module):
    def __init__(self, use_deg=False):
        super(SimsiamModel, self).__init__()
        self.encoder = ResNet18()
        self.encoder.fc = torch.nn.Identity()
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 512)
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 512)
        )
        self.use_deg= use_deg
        self.deg_layer= torch.nn.Linear(512, 4) if use_deg else None

    def forward(self, x1, x2, x3=None):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        h1, h2 = self.projector(z1), self.projector(z2)
        p1, p2 = self.predictor(h1), self.predictor(h2)
        p1 = torch.nn.functional.normalize(p1, dim=1)
        p2 = torch.nn.functional.normalize(p2, dim=1)
        if (x3 is not None) and self.use_deg:
            deg_preds = self.deg_layer(self.projector(self.encoder(x3)))
            return p1, p2, h1, h2, z1, z2, deg_preds

        return p1, p2, h1, h2, z1, z2

class SimclrModel(torch.nn.Module):
    def __init__(self, use_deg=False):
        super(SimclrModel, self).__init__()
        self.encoder = ResNet18()
        self.encoder.fc = torch.nn.Identity()
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 512)
        )
        self.use_deg = use_deg
        self.deg_layer= torch.nn.Linear(512, 4) if use_deg else None

    def forward(self, x1, x2, x3=None):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        h1, h2 = self.projector(z1), self.projector(z2)
        h1 = torch.nn.functional.normalize(h1, dim=1)
        h2 = torch.nn.functional.normalize(h2, dim=1)
        if self.use_deg:
            deg_preds = self.deg_layer(self.projector(self.encoder(x3)))
            return  h1, h2, z1, z2, deg_preds
        return h1, h2, z1, z2

class ByolModel(torch.nn.Module):
    def __init__(self, use_deg=False):
        super(ByolModel, self).__init__()
        self.online_encoder = ResNet18()
        self.online_encoder.fc = torch.nn.Identity()
        self.online_projector = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 512)
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 512)
        )
        self.use_deg = use_deg
        self.deg_layer = torch.nn.Linear(512, 4) if use_deg else None

    def forward(self, x1, x2, x3=None):
        online_z1, online_z2 = self.online_encoder(x1), self.online_encoder(x2)
        online_h1, online_h2 = self.online_projector(online_z1), self.online_projector(online_z2)
        p1, p2 = self.predictor(online_h1), self.predictor(online_h2)
        p1 = torch.nn.functional.normalize(p1, dim=1)
        p2 = torch.nn.functional.normalize(p2, dim=1)
        with torch.no_grad():
            target_z1, target_z2 = self.target_encoder(x1), self.target_encoder(x2)
            target_h1, target_h2 = self.target_projector(target_z1), self.target_projector(target_z2)
            target_h1 = torch.nn.functional.normalize(target_h1, dim=1)
            target_h2 = torch.nn.functional.normalize(target_h2, dim=1)
        if self.use_deg:
            deg_preds= self.deg_layer(self.online_projector(self.online_encoder(x3)))
            return p1, p2, target_h1, target_h2, online_h1, online_h2, online_z1, online_z2, deg_preds
        return p1, p2, target_h1, target_h2, online_h1, online_h2, online_z1, online_z2