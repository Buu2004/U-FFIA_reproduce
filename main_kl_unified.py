import warnings

warnings.filterwarnings("ignore")
import argparse
import logging as log_config
import os
import time

import torch
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim import AdamW

from dataset.unified_dataset import get_dataloader
from models.Audio_front import Audio_Frontend
from models.Audio_model import AudioModel_pre_Cnn10
from models.model_zoo.models import Cnn10
from models.model_zoo.S3D import S3D
from models.UnifiedModel import Unified_Model
from models.Video_model import VideoModel_Pre_S3D
from tasks.unified_kl_task import trainer
from utils.warmupCosineScheduler import WarmupCosineScheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    parser.add_argument('--config', type=str, default='config/unified/exp1_av.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    workspace = config['Workspace']
    exp_name = config['Exp_name']
    modality = config['Modality']
    Training = config['Training']
    audio_features = config['Audio_features']

    modality_dropout = Training['modality_drop']
    audio_dropout = Training['audio_drop']
    batch_size = Training['Batch_size']
    max_epoch = Training['Max_epoch']
    learning_rate = Training['learning_rate']
    seed = Training['seed']
    classes_num = Training['classes_num']
    sample_rate = audio_features['sample_rate']

    ckpt_dir = os.path.join(workspace, exp_name, 'save_models')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(workspace, exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_config.basicConfig(
        level=log_config.INFO,
        format=' %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            log_config.FileHandler(
                os.path.join(log_dir, '%s-%d.log' % (exp_name, time.time()))
            ),
            log_config.StreamHandler(),
        ],
    )

    logger = log_config.getLogger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frontend = Audio_Frontend(**audio_features, training=True)
    frontend_full = Audio_Frontend(**audio_features, training=False)
    # frontend = Audio_Frontend(**audio_features)
    Pre_AudioModel = AudioModel_pre_Cnn10(frontend=frontend_full, backbone=Cnn10()).to(
        device
    )
    Pre_VideoModel = VideoModel_Pre_S3D(backbone=S3D(classes_num=4)).to(device)
    model = Unified_Model(frontend=frontend, classes_num=classes_num)
    model = model.to(device)

    train_loader = get_dataloader(
        split='train',
        batch_size=batch_size,
        seed=seed,
        epoch=0,
        sample_rate=sample_rate,
        num_workers=8,
        drop_last=True,
    )
    test_loader = get_dataloader(
        split='test',
        batch_size=batch_size,
        seed=seed,
        epoch=0,
        sample_rate=sample_rate,
        num_workers=8,
        drop_last=True,
    )
    val_loader = get_dataloader(
        split='val',
        batch_size=batch_size,
        seed=seed,
        epoch=0,
        sample_rate=sample_rate,
        num_workers=8,
        drop_last=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    logger.info(config)
    logger.info(model)
    logger.info(f"{modality} modality experiments running on {device}")
    logger.info(f"Training dataloader: {len(train_loader)* batch_size} samples")
    logger.info(f"Val dataloader: {len(val_loader)* batch_size} samples")
    logger.info(f"Test dataloader: {len(test_loader)* batch_size} samples")
    trainer(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        max_epoch,
        device,
        ckpt_dir,
        modality_dropout,
        audio_dropout,
        Pre_AudioModel,
        Pre_VideoModel,
    )
