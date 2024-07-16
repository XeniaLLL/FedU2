import copy
import torch
from typing import Mapping, Union, Sequence
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

def calculate_ldawa_weights(
    local_models,
    global_model
):
    cosine_similarity = []
    global_para_vector = F.normalize(torch.cat([param.view(-1) for param in global_model.state_dict().values()]), p=2, dim=0)
    for model in local_models:
        model_para_vector = F.normalize(torch.cat([param.view(-1) for param in model.values()]), p=2, dim=0)
        cosine_similarity.append(torch.dot(global_para_vector, model_para_vector))
    weights = [sim / len(local_models) for sim in cosine_similarity]
    return weights

@torch.no_grad()
def layer_aggregate(
    local_models: Sequence[Mapping[str, Tensor]], 
    global_model: Module
) -> Mapping[str, Tensor]:
    cosine_similarity = []
    global_para_vector = F.normalize(torch.cat([param.view(-1) for param in global_model.state_dict().values()]), p=2, dim=0)
    for model in local_models:
        model_para_vector = F.normalize(torch.cat([param.view(-1) for param in model.values()]), p=2, dim=0)
        cosine_similarity.append(torch.dot(global_para_vector, model_para_vector))
    weights = [sim / len(local_models) for sim in cosine_similarity]

    print('cosine similaruty and weights:')
    for cid in range(len(local_models)):
        print(f'client{cid}, cosine similarity: {cosine_similarity[cid]}, weight: {weights[cid]}')

    aggregate_param = copy.deepcopy(global_model.state_dict())
    for name, param in aggregate_param.items():
        new_param = torch.zeros(param.shape)
        for params, weight in zip(local_models, weights):
            new_param += params[name] * weight
        aggregate_param[name] = new_param

    return aggregate_param