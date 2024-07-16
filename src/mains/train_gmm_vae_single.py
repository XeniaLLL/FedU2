import copy
import os.path

import torch
import wandb
from tensorloader import TensorLoader

from my.algo.gmm_vae_local import GMMVaeLocalClient, GMMVaeLocalServer
from my.data import UCRDataset
from my.model.gru import GRUEncoder, GRUDecoder
from . import paths
from .utils import Summary


@Summary.record
def main(
        *,
        hidden_size: int,
        latent_size: int,
        global_rounds: int,
        device,
        lr: float,
        batch_size: int,
        dataset_name: str = 'Two_Patterns',
        train_part: str = 'all',
        test_part: str = 'all',
        summary=Summary()
):
    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)
    root = os.path.join(paths.DATA_DIR, 'UCR_TS_Archive_2015')
    train_set = UCRDataset(root, dataset_name, train_part)
    test_set = UCRDataset(root, dataset_name, test_part)
    cluster_num = len(train_set.targets.unique())
    encoder = GRUEncoder(hidden_size=hidden_size)
    decoder = GRUDecoder(hidden_size=hidden_size, input_size=2)
    # note x -> hidden -> y & gambel_softmax
    y_block = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size)
    )  # careful 指定embedding的dim
    y_logits = torch.nn.Linear(hidden_size, cluster_num)
    # note y-> z prior
    prior_mean_mlp = torch.nn.Sequential(
        torch.nn.Linear(cluster_num, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, latent_size),
        torch.nn.Sigmoid(),
    )
    prior_var_mlp = copy.deepcopy(prior_mean_mlp)
    # note 保持生成p(z|w,x)的时候维度调整到一致
    h_top_mlp = torch.nn.Sequential(
        torch.nn.Linear(cluster_num, hidden_size),
        torch.nn.Dropout(0.2)
    )
    h_latent_mlp = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.Dropout(0.2)
    )
    mean_mlp = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, latent_size),
        torch.nn.Sigmoid(),

    )
    var_mlp = copy.deepcopy(mean_mlp)
    client = GMMVaeLocalClient(
        id=0,
        encoder=copy.deepcopy(encoder),
        mean_mlp=mean_mlp,
        var_mlp=var_mlp,
        prior_mean_mlp=prior_mean_mlp,
        prior_var_mlp=prior_var_mlp,
        y_block=y_block,
        y_logits=y_logits,
        h_top_mlp=h_top_mlp,
        h_latent_mlp=h_latent_mlp,
        decoder=copy.deepcopy(decoder),
        cluster_num=cluster_num,
        hidden_size=hidden_size,
        train_loader=TensorLoader(train_set.data, batch_size=batch_size, shuffle=True),
        test_loader=TensorLoader((test_set.data, test_set.targets), batch_size=batch_size),
        local_epochs=1,
        lr=lr,
        device=device,
    )
    server = GMMVaeLocalServer(
        global_rounds=global_rounds,
        device=device,
    )
    server.clients.append(client)
    server.fit()
