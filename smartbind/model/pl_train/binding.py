import io
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import random
import torch
import wandb
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from sklearn.metrics import auc, confusion_matrix, matthews_corrcoef, roc_curve
from torch import nn
from torch.nn.init import kaiming_normal_

from smartbind.model.binding_head import BindingPredictor
from smartbind.model.RNAFM_pretrained import fm
from smartbind.model.weighted_bce import WeightedBCELoss
from smartbind.model.utils import freeze_model


class BindingPL(LightningModule):
    '''
    BindingPL is a PyTorch Lightning module for training the binding site prediction model.
    '''
    def __init__(self,
                 device: str = 'cuda',
                 out_feature: int = 256,
                 root_path: str = '.',
                 gradient_clip_val: float = 0.5,
                 mlp_dropout: float = 0.1,
                 smol_binding_lr: float = 0.01,
                 smol_binding_weight_decay: float = 0.1,
                 binding_site_lr: float = 0.005,
                 binding_site_weight_decay: float = 0.1,
                 binding_positive_weight: float = 0.6,
                 model_save_folder: str = 'saved_model',
                 fold_num: int = 1,
                 seed: int = 42,
                 vs_mode: bool = False,
                 ):
        '''
        :param device: 'cuda' or 'cpu'
        :param out_feature: joint embedding size for RNA and SMOL representations
        :param root_path:
        :param gradient_clip_val: gradient clip value
        :param mlp_dropout: dropout rate for MLP layers
        :param smol_binding_lr: learning rate for SMOL binding featurizer
        :param smol_binding_weight_decay: weight decay for SMOL binding featurizer
        :param binding_site_lr: learning rate for binding site featurizer
        :param binding_site_weight_decay: weight decay for binding site featurizer
        :param binding_positive_weight: weight for positive class in binding loss
        :param model_save_folder: folder name for saving the model
        :param fold_num: fold number for cross-validation
        :param seed: random seed
        :param vs_mode: validation mode, where default is False
        '''
        super(BindingPL, self).__init__()
        seed_everything(seed)
        self.str_device = device
        self.automatic_optimization = False
        self.gradient_clip_val = gradient_clip_val

        self.rna_fm_model, rna_fm_alphabet = fm.pretrained.rna_fm_t12(device=device)
        self.rna_fm_model = self.rna_fm_model.to(self.device)
        self.batch_converter = rna_fm_alphabet.get_batch_converter()

        smol_layers = [
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, out_feature, bias=True),
            nn.BatchNorm1d(out_feature),
            nn.LeakyReLU(inplace=True),
        ]
        self.smol_featurizer = nn.Sequential(*smol_layers).to(self.device)

        rna_layers = [
            nn.Linear(640, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, out_feature, bias=True),
            nn.BatchNorm1d(out_feature),
            nn.LeakyReLU(inplace=True),
        ]
        self.rna_featurizer = nn.Sequential(*rna_layers).to(self.device)
        for layer in self.rna_featurizer:
            if isinstance(layer, nn.Linear):
                kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # updated layers
        smol_binding_layers = [
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        ]
        self.smol_binding_featurizer = nn.Sequential(*smol_binding_layers).to(self.device)
        self.binding_site_featurizer = BindingPredictor(smol_dim=512, rna_dim=640, hidden_dim=256, dropout_rate=0.2,
                                                        attention_dropout_rate=0.4).to(self.str_device)
        self.smol_binding_lr = smol_binding_lr
        self.smol_binding_weight_decay = smol_binding_weight_decay
        self.binding_site_lr = binding_site_lr
        self.binding_site_weight_decay = binding_site_weight_decay
        self.binding_map_loss = WeightedBCELoss(positive_weight=binding_positive_weight)

        self.validation_step_outputs = []
        self.lowest_val_loss = np.inf
        self.best_model = None
        self.train_predictions = []
        self.train_labels = []
        self.lowest_val_binding_loss = np.inf

        if not vs_mode:
            self.model_path = (f'{root_path}/binding_training_result/fold_{str(fold_num)}/'
                               f'{model_save_folder}-{time.strftime("%d%m%Y-%H%M%S")}')
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            self.save_hyperparameters()

    def configure_optimizers(self):
        smol_binding_opt = torch.optim.AdamW(
            params=self.smol_binding_featurizer.parameters(),
            lr=self.smol_binding_lr,
            weight_decay=self.smol_binding_weight_decay)
        smol_binding_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=smol_binding_opt, T_0=8, eta_min=0.001, T_mult=1)

        binding_site_opt = torch.optim.AdamW(
            params=self.binding_site_featurizer.parameters(),
            lr=self.binding_site_lr,
            weight_decay=self.binding_site_weight_decay)
        binding_site_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=binding_site_opt, T_0=8, eta_min=0.001, T_mult=1)

        return [{"optimizer": smol_binding_opt, "lr_scheduler": smol_binding_lr_scheduler},
                {"optimizer": binding_site_opt, "lr_scheduler": binding_site_lr_scheduler},
                ]

    def training_step(self, batch, batch_idx):
        '''
        Training step for the binding site prediction model
        '''
        rna_sequences, match_smols, _, _, _, binding_index_list = batch
        match_smol_tensor = torch.tensor(match_smols).to(self.str_device)
        match_smol_tensor = match_smol_tensor.float()
        smol_embeddings = self.smol_binding_featurizer(match_smol_tensor)
        token_embeddings_list, truncated_positions_list = self._rna_processing(rna_sequences=rna_sequences,
                                                                               binding_lists=binding_index_list,
                                                                               mode='train')
        avg_binding_map_loss = self._binding_loss(token_embeddings_list=token_embeddings_list,
                                                  smol_embeddings=smol_embeddings,
                                                  binding_index_list=binding_index_list,
                                                  truncated_positions_list=truncated_positions_list)

        opt_smol_binding, opt_binding_map_rna = self.optimizers()
        opt_smol_binding.zero_grad()
        opt_binding_map_rna.zero_grad()
        self.manual_backward(avg_binding_map_loss)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.binding_site_featurizer.parameters(), self.gradient_clip_val)
            torch.nn.utils.clip_grad_norm_(self.smol_binding_featurizer.parameters(), self.gradient_clip_val)
        opt_smol_binding.step()
        opt_binding_map_rna.step()
        self.log("train/binding_loss", avg_binding_map_loss.item())
        return {'loss': avg_binding_map_loss}

    def on_train_epoch_end(self) -> None:
        """
        Called when the train epoch ends.
        """
        smol_binding_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        binding_site_lr = self.trainer.optimizers[1].param_groups[0]['lr']

        wandb.log({
            "lr/smol_binding_featurizer": smol_binding_lr,
            "lr/binding_site_featurizer": binding_site_lr,
        })

        all_predictions = [item for sublist in self.train_predictions for item in sublist]
        all_labels = [item for sublist in self.train_labels for item in sublist]
        all_predictions = torch.tensor(all_predictions)
        all_labels = torch.tensor(all_labels)
        accuracy_tensor = torch.sum(torch.round(all_predictions) == all_labels)
        accuracy_sum = accuracy_tensor.item()
        epoch_binding_map_accuracy = accuracy_sum / len(all_predictions.cpu().numpy())
        self.logger.experiment.log({
            "train/binding_accuracy": epoch_binding_map_accuracy})
        self.train_predictions.clear()
        self.train_labels.clear()

        # update the learning rate of the scheduler
        sch = self.lr_schedulers()
        if isinstance(sch, list):
            for scheduler in sch:
                scheduler.step()
        else:
            sch.step()

    def validation_step(self, batch, batch_idx):
        '''
        Validation step for the binding site prediction model
        '''
        rna_sequences, match_smols, _, rna_sequence_name, _, binding_index_list = batch
        token_embeddings_list = self._rna_processing(rna_sequences, mode='val')
        epoch_binding_loss = []
        binding_prediction_list = []
        binding_label_list = []
        rna_name_list = []
        for match_smol, token_embeddings, rna_names, binding_index in (
                zip(match_smols, token_embeddings_list, rna_sequence_name, binding_index_list)):
            match_smol_tensor = torch.tensor(match_smol).to(self.str_device)
            match_smol_tensor = match_smol_tensor.float()
            smol_embeddings = self.smol_binding_featurizer(match_smol_tensor.unsqueeze(0))

            binding_map_loss, prediction, label = (
                self._binding_position_forward_val(token_embeddings, smol_embeddings, binding_index))

            this_mean_binding_map_loss = binding_map_loss
            epoch_binding_loss.append(this_mean_binding_map_loss)
            binding_prediction_list.append(prediction)
            binding_label_list.append(label)
            rna_name_list.append(rna_names)

        avg_val_binding_map_loss = torch.mean(torch.stack(epoch_binding_loss))

        self.validation_step_outputs.append({
            'epoch_val_binding_map_loss': avg_val_binding_map_loss,
            'prediction': binding_prediction_list,
            'label': binding_label_list,
            'rna_name': rna_name_list
        })

        return {'val_loss': avg_val_binding_map_loss}

    def on_validation_epoch_end(self):
        '''
        Called when the validation epoch ends.
        '''
        all_binding_map_loss = [x["epoch_val_binding_map_loss"] for x in self.validation_step_outputs]
        avg_binding_map_loss = torch.mean(torch.stack(all_binding_map_loss))

        if_save = False
        if avg_binding_map_loss.item() < self.lowest_val_binding_loss:
            if_save = True
            self.lowest_val_binding_loss = avg_binding_map_loss
            self.best_model = self.state_dict()
            # save the model and the mean rank percentiles
            torch.save(self.best_model, self.model_path + '/best_model_val_binding_loss.pth')

        # calculate the binding map accuracy
        self._auc_plot2wandb(prediction_list=[x["prediction"] for x in self.validation_step_outputs],
                             label_list=[x["label"] for x in self.validation_step_outputs],
                             rna_name_list=[x["rna_name"] for x in self.validation_step_outputs],
                             save=if_save)
        self.validation_step_outputs.clear()

        self.log('avg_val_loss', avg_binding_map_loss.item())
        wandb.log({
            "validation/binding_loss": avg_binding_map_loss.item(),
        })

    def _binding_loss(self, token_embeddings_list, smol_embeddings, binding_index_list, truncated_positions_list):
        '''

        :param token_embeddings_list:
        :param smol_embeddings:
        :param binding_index_list:
        :param truncated_positions_list:
        :return:
        '''
        # calculate the binding map loss and record the related variables
        backward_binding_map_losses, binding_prediction_list, binding_label_list = (
            self._binding_position_forward_train(token_embeddings_list,
                                                 smol_embeddings,
                                                 binding_index_list,
                                                 truncated_positions_list))

        self.train_predictions.extend(binding_prediction_list)
        self.train_labels.extend(binding_label_list)
        avg_binding_map_loss = torch.mean(torch.stack(backward_binding_map_losses))
        return avg_binding_map_loss

    def _binding_position_forward_train(self,
                                        rna_sequence_embedding_list,
                                        match_smol_embedding_list,
                                        binding_index_list,
                                        truncated_positions_list):
        '''

        :param rna_sequence_embedding_list:
        :param match_smol_embedding_list:
        :param binding_index_list:
        :param truncated_positions_list:
        :return:
        '''
        self.binding_site_featurizer.to(self.str_device)
        binding_map_loss_list = []
        binding_prediction_list = []
        binding_label_list = []
        for rna_sequence_embedding, match_smol_embedding, binding_index, truncated_positions in zip(
                rna_sequence_embedding_list, match_smol_embedding_list, binding_index_list, truncated_positions_list):
            # get rna-sequence embedding by the truncated positions
            (truncated_start, truncated_end) = truncated_positions
            binding_index_truncated = binding_index[truncated_start:truncated_end]
            binding_map_prediction = self.binding_site_featurizer(smol=match_smol_embedding,
                                                                  rna_sequence=rna_sequence_embedding).squeeze()
            binding_index_tensor = torch.tensor(binding_index_truncated).to(self.str_device)
            binding_map_loss = (self.binding_map_loss(binding_map_prediction, binding_index_tensor.float()) /
                                len(binding_index_truncated))
            binding_map_loss_list.append(binding_map_loss)
            binding_prediction_list.append(binding_map_prediction)
            binding_label_list.append(binding_index_tensor)
        return binding_map_loss_list, binding_prediction_list, binding_label_list

    def _rna_processing(self, rna_sequences, binding_lists=None, mode='train'):
        '''

        :param rna_sequences:
        :param binding_lists:
        :param mode:
        :return:
        '''
        assert mode in ['train', 'val', 'inference'], "Mode must be 'train' or 'val' or 'inference'"
        if mode == 'train':
            rna_embedding_mtx, token_embeddings_mtx, truncated_positions_list = self._rna_encoder_train(rna_sequences,
                                                                                                        binding_lists)
            return token_embeddings_mtx, truncated_positions_list
        elif mode == 'inference':
            rna_embedding_list = []
            token_embeddings_list = []
            for rna_sequence in rna_sequences:
                rna_embedding, token_embeddings = self._rna_encoder_val(rna_sequence)
                rna_embedding_list.append(rna_embedding)
                token_embeddings_list.append(token_embeddings)

            rna_embeddings = torch.stack(rna_embedding_list).to(self.str_device)
            self.rna_featurizer.to(self.str_device)
            rna_tokenized = self.rna_featurizer(rna_embeddings)
            return torch.split(rna_tokenized, 1, dim=0), token_embeddings_list
        else:
            rna_embedding_list = []
            token_embeddings_list = []
            for rna_sequence in rna_sequences:
                rna_embedding, token_embeddings = self._rna_encoder_val(rna_sequence)
                rna_embedding_list.append(rna_embedding)
                token_embeddings_list.append(token_embeddings)

            return token_embeddings_list

    def _rna_encoder_train(self, rna_sequences, binding_lists, max_len=512):
        '''
        This rna encoder is used as the purpose of data augmentation.
        When the sequence length is smaller than 512, it will be padded to 512.
        When the sequence length is larger than 512, it will be truncated to 512, and the sequences have to have a
        binding in the truncated 512.
        :param rna_sequences:
        :param binding_lists:
        :param max_len:
        :return:
        '''
        self.rna_fm_model.to(self.str_device)
        data = []
        len_rna_seq_list = []
        truncated_positions_list = []
        for rna_sequence, binding_list in zip(rna_sequences, binding_lists):
            rna_sequence = rna_sequence[1]
            if len(rna_sequence) > max_len:
                binding_index = [i for i, x in enumerate(binding_list) if x == 1]
                if len(binding_index) == 0:
                    random_index = random.randint(0, len(rna_sequence) - 1)
                else:
                    random_index = random.choice(binding_index)

                while True:
                    start_index = random.randint(0, len(rna_sequence) - max_len)
                    end_index = start_index + max_len
                    if start_index <= random_index < end_index:
                        break

                rna_sequence = rna_sequence[start_index:end_index]
                data.append(('rna', rna_sequence))
                len_rna_seq_list.append(None)
                truncated_positions_list.append((start_index, end_index))
            else:
                rna_sequence_mask = rna_sequence + '<pad>' * (max_len - len(rna_sequence))
                data.append(('rna', rna_sequence_mask))
                len_rna_seq_list.append(len(rna_sequence))
                truncated_positions_list.append((0, len(rna_sequence)))

        _, _, batch_tokens = self.batch_converter(data, max_len=max_len)

        batch_tokens = batch_tokens.to(self.str_device)

        self.rna_fm_model.eval()
        with torch.no_grad():
            results = self.rna_fm_model(batch_tokens, repr_layers=[12])
            token_embeddings = results["representations"][12].squeeze(0)[:, 1:max_len + 1, :]

        token_embeddings_list = []
        for len_rna_seq, token_embedding in zip(len_rna_seq_list, token_embeddings):
            if len_rna_seq is None:
                token_embeddings_list.append(token_embedding)
            else:
                token_embeddings_list.append(token_embedding[:len_rna_seq])

        # mean pooling with token embeddings list
        sequence_embedding_list = [torch.mean(token_embedding, dim=0) for token_embedding in token_embeddings_list]
        sequence_embedding_mtx = torch.stack(sequence_embedding_list)
        return sequence_embedding_mtx, token_embeddings_list, truncated_positions_list

    def _rna_encoder_val(self, rna_sequences):
        '''
        Encode the RNA sequences into embedding
        :param rna_sequences:
        :return:
        '''
        self.rna_fm_model.to(self.str_device)

        if len(rna_sequences[1]) > 1020:
            chunks = [rna_sequences[1][i:i + 1020] for i in range(0, len(rna_sequences[1]), 1020)]
            embeddings = []
            for chunk_sequence in chunks:
                data = [('rna', chunk_sequence)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.str_device)

                self.rna_fm_model.eval()
                with torch.no_grad():
                    results = self.rna_fm_model(batch_tokens, repr_layers=[12])
                    embeddings.append(results["representations"][12].squeeze(0)[1:-1])
            token_embeddings = torch.cat(embeddings, dim=0)
        else:
            data = [rna_sequences]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.str_device)

            self.rna_fm_model.eval()
            with torch.no_grad():
                results = self.rna_fm_model(batch_tokens, repr_layers=[12])
                token_embeddings = results["representations"][12].squeeze(0)[1:-1]
        # mean pooling
        token_embedding = torch.mean(token_embeddings, dim=0)
        return token_embedding, token_embeddings

    def _binding_position_forward_val(self,
                                      rna_sequence_embedding,
                                      match_smol_embedding,
                                      binding_index):
        '''

        :param rna_sequence_embedding:
        :param match_smol_embedding:
        :param binding_index:
        :return:
        '''
        self.binding_site_featurizer.to(self.str_device)

        if len(rna_sequence_embedding) <= 512:
            binding_map_prediction = self.binding_site_featurizer(match_smol_embedding,
                                                                  rna_sequence_embedding).squeeze()
            binding_index_tensor = torch.tensor(binding_index).to(self.str_device)
            binding_map_loss = self.binding_map_loss(binding_map_prediction, binding_index_tensor.float()) / len(
                binding_index)
            return binding_map_loss, binding_map_prediction, binding_index_tensor
        else:  # if the length of the binding map is larger than 512, we will split it into chunks
            if len(rna_sequence_embedding) % 512 == 1:
                rna_sequence_embedding = rna_sequence_embedding[:-1]
                binding_index = binding_index[:-1]
            chunks = [rna_sequence_embedding[i:i + 512] for i in range(0, len(rna_sequence_embedding), 512)]
            binding_index_chunks = [binding_index[i:i + 512] for i in range(0, len(binding_index), 512)]
            binding_map_loss_list = []
            binding_prediction_list = []
            binding_label_list = []
            for rna_sequence_embedding_chunk, binding_index_chunk in zip(chunks, binding_index_chunks):
                binding_map_prediction = self.binding_site_featurizer(match_smol_embedding,
                                                                      rna_sequence_embedding_chunk).squeeze()
                binding_index_tensor = torch.tensor(binding_index_chunk).to(self.str_device)
                binding_map_loss = (self.binding_map_loss(binding_map_prediction, binding_index_tensor.float()) /
                                    len(binding_index_chunk))
                binding_map_loss_list.append(binding_map_loss)
                binding_prediction_list.append(binding_map_prediction)
                binding_label_list.append(binding_index_tensor)

            binding_map_prediction = torch.cat(binding_prediction_list, dim=0)
            binding_index_tensor = torch.cat(binding_label_list, dim=0)
            return torch.mean(torch.stack(binding_map_loss_list)), binding_map_prediction, binding_index_tensor

    def _draw_rank_box_plot(self, rank_list):
        '''

        :param rank_list:
        :return:
        '''
        rank_list = [item for sublist in rank_list for item in sublist]

        buf = io.BytesIO()
        plt.boxplot(rank_list)
        plt.axhline(y=sum(rank_list) / len(rank_list), color='r', linestyle='-')
        plt.scatter([1] * len(rank_list), rank_list, alpha=0.5)
        plt.text(1.1, sum(rank_list) / len(rank_list),
                 f'mean={round(sum(rank_list) / len(rank_list), 4)}')
        plt.text(1.1, sorted(rank_list)[len(rank_list) // 2],
                 f'median={round(sorted(rank_list)[len(rank_list) // 2], 4)}')

        plt.title(f'Rank percentile distribution with total {len(rank_list)} data points')
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)
        pil_img = Image.open(buf)
        wandb.log({"validation/rank_box_plot": [wandb.Image(pil_img)]})

    def _confusion_matrix_plot2wandb(self, predictions, labels, is_train=False):
        '''
        Plot confusion matrix and log to wandb
        :param predictions:
        :param labels:
        :param is_train:
        :return:
        '''
        buf = io.BytesIO()
        cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
        df_cm = pd.DataFrame(cm, index=[i for i in "01"], columns=[i for i in "01"])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.title(f'Confusion matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        pil_img = Image.open(buf)
        if is_train:
            wandb.log({"confusion_matrix": [wandb.Image(pil_img)]})
        else:
            wandb.log({"validation/confusion_matrix": [wandb.Image(pil_img)]})

    def load_pretrained_model(self, model_path, device='cuda', mode='inference'):
        '''
        :param model_path:
        :param device:
        :param mode: 'inference' or 'contact_train' or 'contact_continual_train' or 'binding_continual_train' or 'binding_train'
            inference: load the model for inference, set all the parameters to not require grad
            contact_train: train the SMARTBind contact model from scratch
            contact_continual_train: continual training the SMARTBind contact model
            binding_train: train the SMARTBind binding model from scratch, freeze the contact part
            binding_continual_train: continual training the SMARTBind binding model, freeze the contact part
        :return:
        '''
        assert mode in ['inference', 'contact_train', 'contact_continual_train', 'binding_train', 'binding_continual_train'], \
            ("Mode must be 'inference' or 'contact_train' or 'contact_continual_train' or 'binding_train' or 'binding_continual_train'")
        print(f'Loading pretrained model from {model_path} for {mode}...')
        if model_path is None:
            return
        if mode == 'contact_train':
            for name, param in self.named_parameters():
                param.requires_grad = True
            return
        
        # only load params that are in the model and match the size
        model_dict = self.state_dict()
        pretrained_dict = torch.load(model_path, map_location=torch.device(device=device))
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        not_loaded = [k for k in pretrained_dict if k not in model_dict]
        if len(not_loaded) > 0:
            print(f"Params not loaded: {not_loaded}")

        if mode == 'inference':
            for name, param in self.named_parameters():
                param.requires_grad = False
            self.eval()
            return

        if mode == 'contact_continual_train':
            for name, param in self.named_parameters():
                if 'smol_binding_featurizer' in name or 'binding_site_featurizer' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        if mode in ['binding_train', 'binding_continual_train']:
            freeze_model(self.rna_fm_model)
            freeze_model(self.rna_featurizer)
            freeze_model(self.smol_featurizer)

            if mode == "binding_train":
                for layer in self.smol_binding_featurizer:
                    if isinstance(layer, nn.Linear):
                        kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                        if layer.bias is not None:
                            layer.bias.data.fill_(0.01)
                for layer in self.binding_site_featurizer.modules():
                    if isinstance(layer, nn.Linear):
                        kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                        if layer.bias is not None:
                            layer.bias.data.fill_(0.01)

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name} will be trained.")

    def freeze_contact_models(self):
        freeze_model(self.rna_fm_model)
        freeze_model(self.rna_featurizer)
        freeze_model(self.smol_featurizer)

    def predict_binding(self, rna_sequence, ligand):
        '''

        :param rna_sequence: RNA sequence string
        :param ligand: in fp2 list
        :return:
        '''
        rna_sequence = rna_sequence.upper()
        ligand = torch.tensor(ligand).to(self.device)
        rna_sequence = [('rna', rna_sequence)]
        _, _, batch_tokens = self.batch_converter(rna_sequence)
        batch_tokens = batch_tokens.to(self.device)
        self.eval()
        with torch.no_grad():
            results = self.rna_fm_model(batch_tokens, repr_layers=[12])
            rna_sequence_embedding = results["representations"][12].squeeze(0)[1:-1]
        match_smol_embedding = self.smol_binding_featurizer(ligand.float().unsqueeze(0))
        binding_map_prediction = self.binding_site_featurizer(match_smol_embedding,
                                                              rna_sequence_embedding).squeeze()
        return binding_map_prediction

    def _auc_plot2wandb(self, prediction_list=None,
                        label_list=None, rna_name_list=None, save=False):
        """
        Plot AUC curve and log to wandb.
        Plot MCC curve and log to wandb.
        """
        if save:
            result_dict = {
                'prediction_list': [[tensor.cpu() for tensor in sublist] for sublist in prediction_list],
                'label_list': [[tensor.cpu() for tensor in sublist] for sublist in label_list],
                'rna_name_list': rna_name_list,
            }
            with open(self.model_path + '/input_result_dict.pkl', 'wb') as f:
                pickle.dump(result_dict, f)

        # calculate threshold of the predictions using Youden's J statistic
        yjs_labels = []
        yjs_predictions = []
        for i in range(len(prediction_list)):
            this_prediction_batch = prediction_list[i]
            this_label_batch = label_list[i]
            for j in range(len(this_label_batch)):
                this_prediction = this_prediction_batch[j]
                this_label = this_label_batch[j]
                this_prediction_tensors = [pred.clone().detach() for pred in this_prediction]
                this_prediction = torch.stack(this_prediction_tensors).clone().detach()
                this_label_tensors = [label.clone().detach() for label in this_label]
                this_label = torch.stack(this_label_tensors).clone().detach()
                yjs_labels.extend(this_label.cpu().numpy())
                yjs_predictions.extend(this_prediction.cpu().numpy())

        fpr, tpr, thresholds = roc_curve(yjs_labels, yjs_predictions)
        optimal_idx = np.argmax(tpr - fpr)
        yjs_threshold = thresholds[optimal_idx]

        # calculate the auc score, accuracy, and mcc score separately
        auc_score_list = []
        mcc_score_list = []
        nan_auc_score_list = []
        nan_mcc_score_list = []
        nan_this_label_list = []
        nan_this_prediction_list = []
        for i in range(len(prediction_list)):
            this_prediction_batch = prediction_list[i]
            this_label_batch = label_list[i]
            for j in range(len(this_label_batch)):
                this_prediction = this_prediction_batch[j]
                this_label = this_label_batch[j]
                this_prediction_tensors = [pred.clone().detach() for pred in this_prediction]
                this_prediction = torch.stack(this_prediction_tensors).clone().detach()
                this_label_tensors = [label.clone().detach() for label in this_label]
                this_label = torch.stack(this_label_tensors).clone().detach()
                # prediction by threshold
                this_prediction_threshold = this_prediction > yjs_threshold
                mcc = matthews_corrcoef(this_label.cpu().numpy(), this_prediction_threshold.cpu().numpy())

                fpr, tpr, thresholds = roc_curve(this_label.cpu().numpy(), this_prediction.cpu().numpy())
                auc_score = auc(fpr, tpr)
                if np.isnan(auc_score):
                    nan_auc_score_list.append(auc_score)
                    nan_mcc_score_list.append(mcc)
                    nan_this_label_list.append(this_label)
                    nan_this_prediction_list.append(this_prediction)
                    print(f'nan auc score: {auc_score}')
                    continue
                auc_score_list.append(auc_score)
                mcc_score_list.append(mcc)

        avg_mcc = sum(mcc_score_list) / len(mcc_score_list)
        avg_auc = sum(auc_score_list) / len(auc_score_list)
        # save the result_dict as pkl
        if save:
            result_dict = {
                'mcc_score_list': mcc_score_list,
                'auc_score_list': auc_score_list,
            }
            with open(self.model_path + '/separate_result_dict.pkl', 'wb') as f:
                pickle.dump(result_dict, f)

            nan_dict = {
                'nan_mcc_score_list': nan_mcc_score_list,
                'nan_auc_score_list': nan_auc_score_list,
                'nan_this_label_list': nan_this_label_list,
                'nan_this_prediction_list': nan_this_prediction_list,
            }
            with open(self.model_path + '/nan_result_dict.pkl', 'wb') as f:
                pickle.dump(nan_dict, f)

            wandb.log({"validation_loss/separate_avg_mcc": avg_mcc,
                       "validation_loss/separate_avg_auc": avg_auc,
                       "validation_loss/yjs_threshold": yjs_threshold})

        wandb.log({"validation/separate_avg_mcc": avg_mcc,
                   "validation/separate_avg_auc": avg_auc,
                   "validation/yjs_threshold": yjs_threshold})

    def inference_single_rna(self, rna_sequence):
        """
        Inference the model with a single RNA sequence
        :param rna_sequence:
        :return:
        """
        self.eval()
        rna_tokenized_list, token_embeddings_list = self._rna_processing([('s', rna_sequence)], mode='inference')
        rna_tokenized = rna_tokenized_list[0]
        return rna_tokenized

    def inference_list_smols(self, smol_fp2_list):
        """
        Inference the model with a list of small molecules
        :param smol_fp2_list:
        :return:
        """
        self.eval()
        smol_tokenized_list, _ = self._smol_processing(smol_fp2_list)
        return smol_tokenized_list

    def _smol_processing(self, smol_fp2):
        smol_fp2_tensor = torch.tensor(smol_fp2).float().to(self.str_device)

        smol_embeddings = self.smol_featurizer(smol_fp2_tensor)
        smol_tokenized_list = smol_embeddings
        return smol_tokenized_list, smol_embeddings