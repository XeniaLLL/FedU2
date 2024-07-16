import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from tensorloader import TensorLoader


@torch.no_grad()
def knn_evaluate(
    encoder: torch.nn.Module,
    train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
    test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    n_neighbors: int = 200,
    batch_size: int = 128,
) -> float:
    encoder.to(device)
    encoder.eval()
    train_z_list = []
    train_y_list = []
    test_z_list = []
    test_y_list = []
    train_loader = DataLoader(train_set, batch_size)
    for x, y in train_loader:
        x = x.to(device)
        z = encoder(x).cpu()
        z = torch.nn.functional.normalize(z, dim=1)
        train_z_list.append(z)
        train_y_list.append(y)
    test_loader = DataLoader(test_set, batch_size)
    for x, y in test_loader:
        x = x.to(device)
        z = encoder(x).cpu()
        z = torch.nn.functional.normalize(z, dim=1)
        test_z_list.append(z)
        test_y_list.append(y)
    encoder.cpu()
    train_z = torch.concat(train_z_list)
    train_y = torch.concat(train_y_list)
    test_z = torch.concat(test_z_list)
    test_y = torch.concat(test_y_list)
    pred = knn_fit_predict(train_z, train_y, test_z, n_neighbors, device, batch_size)
    acc = accuracy_score(test_y, pred)
    return float(acc)


def knn_fit_predict(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    test_z: torch.Tensor,
    n_neighbors: int,
    device: torch.device,
    batch_size: int = 128
) -> torch.Tensor:
    train_z = train_z.to(device).t().contiguous()
    train_y = train_y.to(device)
    classes = len(train_y.unique())
    pred_list = []
    for z in TensorLoader(test_z, batch_size=batch_size):
        pred = knn_predict(z.to(device), train_z, train_y, classes, n_neighbors, 0.1)
        pred = pred[:, 0]
        pred_list.append(pred.cpu())
    return torch.concat(pred_list)


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    feature_labels: torch.Tensor,
    classes: int,
    knn_k: int,
    knn_t: float
):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
