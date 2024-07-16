import sys
import os.path

import wandb
from fedbox.utils.training import set_seed

SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(SRC_DIR)

import mains

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
job_task_name= ""

def main():
    set_seed(66)
    wandb.init(
        # project='fedusl-test',
        # project='byol-backbone',
        project='YOUR PROJECT NAME',
        entity='YOUR GROUP NAME',
        name=job_task_name,
        # mode='disabled' # activate wandb
    )
    # mains.train_byol_single.main(
    #     dataset='cifar10',
    #     global_rounds=10,
    #     device="cuda:0",
    #     lr=0.01,
    #     batch_size=128,
    #     weight_decay=0.999,
    #     torchvision_root='./data',
    # )

    mains.train_fedU2.main(
        dataset='cifar10',
        split_file='./data/fed/cifar10_dirlabel,client=10,alpha=0.1.json',
        job_task_name=job_task_name,
        # divergence_threshold=0.6,
        sharpen_ratio=1,
        join_ratio=1.,
        client_method='simclr',
        mtl_method='EUA',
        whole_model=True,
        gmodel_lr=8,
        lr=0.048,
        lr_schedule_step2= 400,
        w_lr=0.01,
        local_epochs=5,
        weight_decay= 1e-4,
        # device='cpu',
        device='cuda:3',
        use_deg=True,
        torchvision_root='DATASET PATH', # 11
    )


if __name__ == '__main__':
    main()
