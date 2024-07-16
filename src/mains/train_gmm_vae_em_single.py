import copy
import os.path

import torch
from tensorloader import TensorLoader

from my.algo.gmm_vae_local import *
from my.data import UCRDataset
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
    device = torch.device(device)
    root = os.path.join(paths.DATA_DIR, 'UCR_TS_Archive_2015')
    train_set = UCRDataset(root, dataset_name, train_part)
    test_set = UCRDataset(root, dataset_name, test_part)
    cluster_num = len(train_set.targets.unique())
    gmvae_model= GMVAE(
            unsupervised_em_iters=5,
            fix_pi=False,
            hidden_size=hidden_size,
            component_size=cluster_num,
            latent_size=latent_size,
            train_mc_sample_size=10,
            test_mc_sample_size=10
    )
    client = GMMVaeEMLocalClient(
        id=0,
        model= gmvae_model,
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
    server.test()
