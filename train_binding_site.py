import os

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from smartbind import BindingPL
from smartbind.preprocess import build_val_test_set
from smartbind.dataloader import RLDataLoader, RLDataset
from smartbind.utils import lightning_accelerator, resolve_device
from omegaconf import DictConfig, OmegaConf
import hydra
import logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None,
            config_path='conf',
            config_name='binding_site_training')
def binding_task_training(cfg: DictConfig) -> None:
    cfg.wandb_params.task = (
        f"{cfg.training_params.model_save_folder}"
        f"-fold{cfg.data_params.fold_num}s{cfg.training_params.seed}"
    )

    logger.info(f'Loading binding prediction training configuration... \n {OmegaConf.to_yaml(cfg)}')
    seed_everything(cfg.training_params.seed)
    current_path = os.path.dirname(os.path.abspath(__file__))
    current_fold_num = cfg.data_params.fold_num
    replace_res_type = ['R', 'Y', 'K', 'M', 'S', 'W', 'B', 'D', 'H', 'V', 'N', '-']

    logger.info(f'Constructing training for fold number {current_fold_num}...')
    val_rna_seq_list, _, val_rna_name_list, val_match_smol_name_list, val_binding_index_list, \
        val_match_smol_fp_list, train_rna_seq_list, _, train_rna_name_list, \
        train_match_smol_name_list, train_binding_index_list, train_match_smol_fp_list = (
                                        build_val_test_set(
                                            fold_num=current_fold_num-1,
                                            dict_path=os.path.join(current_path, cfg.data_params.training_data),
                                            return_rna_smol_map=False))

    logger.info(f'Splitting training and validation set, '
                f'# of training set: {len(train_rna_seq_list)}, '
                f'# of validation set: {len(val_rna_seq_list)}...')
    val_dataset = RLDataset(rna_sequences=val_rna_seq_list,
                            rna_sequences_names=val_rna_name_list,
                            match_smols=val_match_smol_fp_list,
                            match_smols_names=val_match_smol_name_list,
                            non_binding_index_list=val_binding_index_list,
                            is_val=True,
                            rna_smol_map=[])
    val_dataloader = RLDataLoader(dataset=val_dataset,
                                  batch_size=cfg.training_params.batch_size,
                                  num_workers=cfg.data_params.num_workers,
                                  if_shuffle=False)

    train_dataset = RLDataset(rna_sequences=train_rna_seq_list,
                              rna_sequences_names=train_rna_name_list,
                              match_smols=train_match_smol_fp_list,
                              match_smols_names=train_match_smol_name_list,
                              non_binding_index_list=train_binding_index_list,
                              replace_res_type=replace_res_type,
                              replace_ratio=cfg.data_params.data_aug_replace_ratio,
                              augmentation_factor=cfg.data_params.data_aug_factor,
                                                            is_val=False,
                                                            rna_smol_map=[])
    train_dataloader = RLDataLoader(dataset=train_dataset,
                                    batch_size=cfg.training_params.batch_size,
                                    num_workers=cfg.data_params.num_workers,
                                    if_shuffle=True)

    device = resolve_device(cfg.training_params.device)
    accelerator, devices = lightning_accelerator(device)
    logger.info(f'training the binding site model on {device} (accelerator={accelerator}, devices={devices})')

    """
    Training the binding module with RNA foundation model fine-tuned.
    """
    wandb.login(key=cfg.wandb_params.key)
    wandb.init(project=cfg.wandb_params.project,
               entity=cfg.wandb_params.entity,
               name=f'{cfg.wandb_params.task}_binding')
    wandb_logger = WandbLogger(project=cfg.wandb_params.project,
                               log_model=cfg.wandb_params.log_model,
                               offline=cfg.wandb_params.offline)
    wandb_logger.log_hyperparams(cfg)

    logger.info(f'Set early stopping for binding site prediction: patience={cfg.training_params.patience}')
    binding_early_stop_callback = EarlyStopping(monitor="avg_val_loss",
                                                min_delta=0.00,
                                                patience=cfg.training_params.patience,
                                                verbose=False,
                                                mode="min")

    binding_trainer = Trainer(accelerator=accelerator,
                              devices=devices,
                              max_epochs=cfg.training_params.max_epochs,
                              callbacks=[binding_early_stop_callback],
                              logger=wandb_logger,
                              sync_batchnorm=True,
                              detect_anomaly=True,
                              )

    # build model
    binding_model = BindingPL(device=device,
                              out_feature=cfg.model_params.out_feature,
                              root_path=current_path,
                              smol_binding_lr=cfg.training_params.smol_binding_lr,
                              smol_binding_weight_decay=cfg.training_params.smol_binding_weight_decay,
                              binding_site_lr=cfg.training_params.binding_site_lr,
                              binding_site_weight_decay=cfg.training_params.binding_site_weight_decay,
                              fold_num=cfg.data_params.fold_num,
                              gradient_clip_val=cfg.training_params.gradient_clip_val,
                              mlp_dropout=cfg.training_params.mlp_dropout,
                              binding_positive_weight=cfg.model_params.binding_positive_weight,
                              model_save_folder=cfg.training_params.model_save_folder,
                              seed=cfg.training_params.seed
                              ).to(device)

    binding_score_model_dir = cfg.model_params.binding_score_model_path
    if not os.path.isabs(binding_score_model_dir):
        binding_score_model_dir = os.path.join(current_path, binding_score_model_dir)
    binding_score_model_path = os.path.join(binding_score_model_dir, f'fold{cfg.data_params.fold_num}.pth')
    if not os.path.exists(binding_score_model_path):
        logger.warning(f'Contact model path {binding_score_model_path} does not exist, so the binding '
                       f'score module will start from the pretrained RNA-FM weights instead of from a '
                       f'trained SMARTBind binding score model. This is NOT the published Stage 2 setup; '
                       f'set model_params.binding_score_model_path to the folder holding fold*.pth '
                       f'(SMARTBind_weight after downloading the weights) if that was not intended.')
        binding_score_model_path = None
    binding_model.load_pretrained_model(model_path=binding_score_model_path,
                                        device=device,
                                        mode=cfg.model_params.training_mode)

    binding_trainer.fit(binding_model,
                        train_dataloaders=train_dataloader.dataloader,
                        val_dataloaders=val_dataloader.dataloader)
    wandb.finish()


if __name__ == '__main__':
    binding_task_training()
