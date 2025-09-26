import io
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import wandb
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch import nn

from ..margin import MarginScheduledLossFunction, sigmoid_cosine_distance_test

from ..RNAFM_pretrained import fm


class ContactPL(LightningModule):
    def __init__(self,
                 device='cuda',
                 margin: float = 0.6,
                 out_feature: int = 256,
                 rna_mlp_lr: float = 0.05,
                 rna_mlp_weight_decay: float = 0.001,
                 smol_mlp_lr: float = 0.05,
                 smol_mlp_weight_decay: float = 0.001,
                 root_path: str = '.',
                 lr_rna_fm: float = 1e-5,
                 rna_fm_weight_decay: float = 0.001,
                 fold_num: int = 1,
                 gradient_clip_val: float = 0.5,
                 mlp_dropout: float = 0.1,
                 model_save_folder: str = 'saved_model',
                 seed: int = 42,
                 vs_mode: bool = False,
                 ):
        super(ContactPL, self).__init__()
        seed_everything(seed)

        self.automatic_optimization = False
        self.gradient_clip_val = gradient_clip_val

        self.rna_mlp_lr = rna_mlp_lr
        self.rna_mlp_weight_decay = rna_mlp_weight_decay
        self.smol_mlp_lr = smol_mlp_lr
        self.smol_mlp_weight_decay = smol_mlp_weight_decay
        self.str_device = device

        self.rna_fm_model, rna_fm_alphabet = fm.pretrained.rna_fm_t12(device=device)
        self.rna_fm_model = self.rna_fm_model.to(self.str_device)
        self.batch_converter = rna_fm_alphabet.get_batch_converter()

        self.finetune_rna_fm = False if vs_mode else True
        self.lr_rna_fm = lr_rna_fm
        self.rna_fm_weight_decay = rna_fm_weight_decay

        if not self.finetune_rna_fm:
            self._freeze_model(self.rna_fm_model)
            self.rna_fm_model.eval()
        else:
            self._freeze_model(self.rna_fm_model)
            for param in self.rna_fm_model.layers[-1].parameters():
                param.requires_grad = True

        smol_layers = [
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, out_feature, bias=True),
            nn.BatchNorm1d(out_feature),
            nn.LeakyReLU(inplace=True),
        ]
        self.smol_featurizer = nn.Sequential(*smol_layers).to(self.str_device)

        rna_layers = [
            nn.Linear(640, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, out_feature, bias=True),
            nn.BatchNorm1d(out_feature),
            nn.LeakyReLU(inplace=True),
        ]
        self.rna_featurizer = nn.Sequential(*rna_layers).to(self.str_device)

        self.contrastive_loss_fct = MarginScheduledLossFunction(
            M_0=margin,
            N_epoch=40,
            N_restart=8,
            update_fn='tanh_decay',
        )

        self.epoch_train_loss_list = []
        self.validation_step_outputs = []
        self.low_rank_pairs = []
        self.lowest_val_loss = np.inf

        if not vs_mode:
            self.model_path = (f'{root_path}/contact_training_result/fold_{str(fold_num)}/'
                               f'{model_save_folder}-{time.strftime("%d%m%Y-%H%M%S")}')
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        rna_sequences, match_smols, decoy_smols_list, rna_sequences_names, match_smols_name, _ = batch
        rna_tokenized_list, token_embeddings_list = self._rna_processing(rna_sequences)
        backward_losses = []

        for anchor_rna, match_smol, decoy_smols, token_embeddings \
                in zip(rna_tokenized_list, match_smols, decoy_smols_list, token_embeddings_list):
            smol_cts_losses = []
            rna_cts_losses = []
            # add match_smol to the first position of decoy_smols in order to batch normalize
            full_smols_list = [match_smol] + decoy_smols
            full_smol_tokenized_list, _ = self._smol_processing(full_smols_list)
            match_smol_tokenized = full_smol_tokenized_list[0]
            decoy_smol_tokenized_list = full_smol_tokenized_list[1:]

            decoy_rna_tokenized_list = [rna_tokenized for rna_tokenized in rna_tokenized_list
                                        if rna_tokenized is not anchor_rna]
            for decoy_rna in decoy_rna_tokenized_list:
                rna_cts_loss = self.contrastive_loss_fct(match_smol_tokenized.unsqueeze(dim=0),
                                                         anchor_rna.squeeze(1),
                                                         decoy_rna.squeeze(1))
                rna_cts_losses.append(rna_cts_loss)

            for decoy_smol_tokenized in decoy_smol_tokenized_list:
                smol_cts_loss = self.contrastive_loss_fct(anchor_rna.squeeze(1),
                                                          match_smol_tokenized.unsqueeze(dim=0),
                                                          decoy_smol_tokenized.unsqueeze(dim=0))
                smol_cts_losses.append(smol_cts_loss)

            this_avg_smol_cts_loss = torch.mean(torch.stack(smol_cts_losses))
            this_avg_rna_cts_loss = torch.mean(torch.stack(rna_cts_losses))
            this_joint_cts_loss = (this_avg_smol_cts_loss + this_avg_rna_cts_loss) / 2
            backward_losses.append(this_joint_cts_loss)
            self.epoch_train_loss_list.append(this_joint_cts_loss.item())

        avg_loss = torch.mean(torch.stack(backward_losses))

        # manually optimize the model
        opt_rna_mlp, opt_smol_mlp, opt_rna_fm = self.optimizers()

        opt_rna_mlp.zero_grad()
        opt_smol_mlp.zero_grad()
        self.manual_backward(avg_loss, retain_graph=True)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.rna_featurizer.parameters(), self.gradient_clip_val)
            torch.nn.utils.clip_grad_norm_(self.smol_featurizer.parameters(), self.gradient_clip_val)
        opt_rna_mlp.step()
        opt_smol_mlp.step()

        # Update the foundation models
        if self.finetune_rna_fm:
            multi_loss = avg_loss
            if self.finetune_rna_fm:
                opt_rna_fm.zero_grad()

            self.manual_backward(multi_loss)
            if self.gradient_clip_val > 0:
                if self.finetune_rna_fm:
                    torch.nn.utils.clip_grad_norm_(self.rna_fm_model.parameters(), self.gradient_clip_val)

            if self.finetune_rna_fm:
                opt_rna_fm.step()

    def on_train_epoch_end(self) -> None:
        """
        Called when the train epoch ends.
        """
        epoch_train_loss = sum(self.epoch_train_loss_list) / len(self.epoch_train_loss_list)

        self.epoch_train_loss_list.clear()

        self.log("avg_train_loss", epoch_train_loss)
        self.logger.experiment.log({
            "avg_train_loss": epoch_train_loss,
        })

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.logger.experiment.log({
            "rna/mlp_lr": current_lr,
        })
        current_lr = self.trainer.optimizers[1].param_groups[0]['lr']
        self.logger.experiment.log({
            "smol/mlp_lr": current_lr,
        })
        if self.finetune_rna_fm:
            current_lr = self.trainer.optimizers[2].param_groups[0]['lr']
            self.logger.experiment.log({
                "rna/rna_fm_lr": current_lr,
            })

        # update the learning rate of the scheduler
        sch = self.lr_schedulers()
        if isinstance(sch, list):
            for scheduler in sch:
                scheduler.step()
        else:
            sch.step()

    def validation_step(self, batch, batch_idx):
        rna_sequences, match_smols, decoy_smols_list, rna_sequences_names, match_smols_name, _ = batch
        rna_tokenized_list, token_embeddings_list = self._rna_processing(rna_sequences)
        epochs_val_smol_cts_loss = []
        epochs_val_rna_cts_loss = []
        rank_percentile_list = []
        positive_anchor_distance = []

        for anchor_rna, match_smol, decoy_smols, rna_sequence_name, match_smol_name, token_embeddings in \
                zip(rna_tokenized_list, match_smols, decoy_smols_list, rna_sequences_names, match_smols_name, token_embeddings_list):
            smol_cts_losses = []
            rna_cts_losses = []
            anchor_rna = anchor_rna.squeeze(1)
            match_smol_tokenized_list, _ = self._smol_processing([match_smol])
            match_smol_tokenized = match_smol_tokenized_list[0]
            decoy_smol_tokenized_list, _ = self._smol_processing(decoy_smols)
            decoy_distance_list = []
            match_smol_tokenized = match_smol_tokenized.unsqueeze(dim=0)
            for decoy_smol_tokenized in decoy_smol_tokenized_list:
                decoy_smol_tokenized = decoy_smol_tokenized.unsqueeze(dim=0)
                decoy_distance = sigmoid_cosine_distance_test(anchor_rna.squeeze(1),
                                                              decoy_smol_tokenized)
                decoy_distance_list.append(decoy_distance)

                this_smol_cts_loss = self.contrastive_loss_fct(anchor_rna.squeeze(1),
                                                               match_smol_tokenized,
                                                               decoy_smol_tokenized)
                smol_cts_losses.append(this_smol_cts_loss)

            decoy_rna_tokenized_list = [rna_tokenized for rna_tokenized in rna_tokenized_list
                                        if rna_tokenized is not anchor_rna]
            for decoy_rna in decoy_rna_tokenized_list:
                this_rna_cts_loss = self.contrastive_loss_fct(match_smol_tokenized,
                                                              anchor_rna.squeeze(1),
                                                              decoy_rna.squeeze(1))
                rna_cts_losses.append(this_rna_cts_loss)

            this_avg_smol_cts_loss = torch.mean(torch.stack(smol_cts_losses))
            epochs_val_smol_cts_loss.append(this_avg_smol_cts_loss)
            this_avg_rna_cts_loss = torch.mean(torch.stack(rna_cts_losses))
            epochs_val_rna_cts_loss.append(this_avg_rna_cts_loss)

            match_distance = sigmoid_cosine_distance_test(anchor_rna, match_smol_tokenized)
            positive_anchor_distance.append(match_distance)
            rank_percentile = self._rank_eval(match_distance, decoy_distance_list)
            rank_percentile_list.append(rank_percentile)

            if rank_percentile <= 0.6:
                self.low_rank_pairs.append(
                    [rna_sequence_name, match_smol_name, rank_percentile, self.current_epoch]
                )

        avg_val_smol_cts_loss = torch.mean(torch.stack(epochs_val_smol_cts_loss))
        avg_val_rna_cts_loss = torch.mean(torch.stack(epochs_val_rna_cts_loss))
        avg_val_loss = (avg_val_smol_cts_loss + avg_val_rna_cts_loss) / 2

        self.validation_step_outputs.append({'val_loss': avg_val_loss,
                                             'rank_percentile_list': rank_percentile_list,
                                             'positive_anchor_distance': positive_anchor_distance})
        return {'val_loss': avg_val_loss, 'rank_percentile_list': rank_percentile_list}

    def on_validation_epoch_end(self):
        all_loss = [x["val_loss"] for x in self.validation_step_outputs]
        avg_loss = torch.mean(torch.stack(all_loss))
        all_rank_percentiles = [x["rank_percentile_list"] for x in self.validation_step_outputs]
        # convert 2d list to 1d list
        all_rank_percentiles_1d = [item for sublist in all_rank_percentiles for item in sublist]
        # calculate the mean rank percentile
        mean_rank_percentiles = sum(all_rank_percentiles_1d) / len(all_rank_percentiles_1d)

        # log for early stopping monitor
        self.log('avg_val_loss', avg_loss.item())
        self.logger.experiment.log({
            "avg_val_loss": avg_loss.item(),
        })

        # log the low rank pairs into wandb as table
        wandb_table_data = self.low_rank_pairs.copy()
        columns = ["rna_sequence_name", "match_smol_name", "rank_percentile", "epoch"]
        wandb_table = wandb.Table(data=wandb_table_data, columns=columns)
        wandb.log({"low_rank_pairs": wandb_table})

        # save model weight if the validation loss is the lowest
        if avg_loss.item() < self.lowest_val_loss:
            self.logger.experiment.log({
                "validation/mean_rank_percentile": mean_rank_percentiles,
            })

            # record positive anchor distances distribution
            all_positive_anchor_distance = [x["positive_anchor_distance"] for x in self.validation_step_outputs]
            all_positive_anchor_distance = [item for sublist in all_positive_anchor_distance for item in sublist]
            all_positive_anchor_distance = [item for sublist in all_positive_anchor_distance for item in sublist]
            all_positive_anchor_distance = [x.cpu().numpy() for x in all_positive_anchor_distance]
            # as box plot
            fig = plt.figure()
            plt.boxplot(all_positive_anchor_distance)
            plt.axhline(y=sum(all_positive_anchor_distance) / len(all_positive_anchor_distance), color='r',
                        linestyle='-')
            plt.scatter([1] * len(all_positive_anchor_distance), all_positive_anchor_distance, alpha=0.5)
            plt.text(1.1, sum(all_positive_anchor_distance) / len(all_positive_anchor_distance),
                     f'mean={round(sum(all_positive_anchor_distance) / len(all_positive_anchor_distance), 4)}')
            plt.title(f"Positive anchor distances for {len(all_positive_anchor_distance)} data points")
            plt.xlabel("Epoch")
            plt.ylabel("Distance")
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            wandb.log({"positive_anchor_distances": wandb.Image(image)})
            buf.close()
            plt.close()

            # save all_positive_anchor_distance and all_rank_percentiles to local as pkl
            with open(self.model_path + '/positive_anchor_distances.pkl', 'wb') as f:
                pickle.dump(all_positive_anchor_distance, f)
            with open(self.model_path + '/rank_percentiles.pkl', 'wb') as f:
                pickle.dump(all_rank_percentiles, f)

            self._draw_rank_box_plot(all_rank_percentiles)
            self.lowest_val_loss = avg_loss.item()

            torch.save(self.state_dict(), self.model_path + '/best_model_loss.pth')
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        rna_featurizer_opt = torch.optim.AdamW(self.rna_featurizer.parameters(),
                                               lr=self.rna_mlp_lr,
                                               weight_decay=self.rna_mlp_weight_decay)
        rna_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                rna_featurizer_opt,
                                T_0=8,
                                eta_min=0.001,
                                T_mult=1)

        smol_featurizer_opt = torch.optim.AdamW(self.smol_featurizer.parameters(),
                                                lr=self.smol_mlp_lr,
                                                weight_decay=self.smol_mlp_weight_decay)
        smol_lr_scheduler = (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                smol_featurizer_opt,
                                T_0=8,
                                eta_min=0.001,
                                T_mult=1))

        opt_rna_fm = torch.optim.AdamW(self.rna_fm_model.parameters(),
                                       lr=self.lr_rna_fm,
                                       weight_decay=self.rna_fm_weight_decay)

        return [{"optimizer": rna_featurizer_opt, "lr_scheduler": rna_lr_scheduler},
                {"optimizer": smol_featurizer_opt, "lr_scheduler": smol_lr_scheduler},
                {"optimizer": opt_rna_fm}
                ]

    def _rna_processing(self, rna_sequences):
        """
        RNA processing with sequence data augmentation option
        Prepare for RNA decoy sample generation
        """
        rna_embedding_list = []
        token_embeddings_list = []
        for rna_sequence in rna_sequences:
            rna_embedding, token_embeddings = self._rna_encoder(rna_sequence)
            rna_embedding_list.append(rna_embedding)
            token_embeddings_list.append(token_embeddings)

        rna_embeddings = torch.stack(rna_embedding_list).to(self.str_device)
        self.rna_featurizer.to(self.str_device)
        rna_tokenized = self.rna_featurizer(rna_embeddings)
        return torch.split(rna_tokenized, 1, dim=0), token_embeddings_list

    def _smol_processing(self, smol_fp2):
        smol_fp2_tensor = torch.tensor(smol_fp2).float().to(self.str_device)

        smol_embeddings = self.smol_featurizer(smol_fp2_tensor)
        smol_tokenized_list = smol_embeddings
        return smol_tokenized_list, smol_embeddings

    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _draw_rank_box_plot(self, rank_list):
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

    def _auc_plot2wandb(self, predictions, labels, is_train=False):
        """
        Plot AUC curve and log to wandb
        """
        fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), predictions.cpu().numpy())
        auc_score = auc(fpr, tpr)
        buf = io.BytesIO()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve with AUC score {auc_score}')
        plt.legend(loc="lower right")
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        pil_img = Image.open(buf)
        if is_train:
            wandb.log({"roc_curve": [wandb.Image(pil_img)]})
        else:
            wandb.log({"validation/roc_curve": [wandb.Image(pil_img)]})

    def _confusion_matrix_plot2wandb(self, predictions, labels, is_train=False):
        """
        Plot confusion matrix and log to wandb
        """
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

    def _rank_eval(self, match_distance, decoy_distance_list):
        """
        Calculate the rank percentile of the match distance among the decoy distance list
        Larger distance means more similar
        """
        rank = 0
        for decoy_distance in decoy_distance_list:
            if match_distance.item() <= decoy_distance.item():
                rank += 1
        return (len(decoy_distance_list)+1 -rank) / (len(decoy_distance_list)+1)

    def _rna_encoder(self, rna_sequences):
        """
        Encode the RNA sequences into embedding
        """
        self.rna_fm_model.to(self.str_device)

        if len(rna_sequences[1]) > 1020:
            chunks = [rna_sequences[1][i:i + 1020] for i in range(0, len(rna_sequences[1]), 1020)]
            embeddings = []
            for chunk_sequence in chunks:
                data = [('rna', chunk_sequence)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.str_device)
                if self.finetune_rna_fm:
                    with torch.set_grad_enabled(self.finetune_rna_fm):
                        results = self.rna_fm_model(batch_tokens, repr_layers=[12])
                        embeddings.append(results["representations"][12].squeeze(0)[1:-1])
                else:
                    self.rna_fm_model.eval()
                    with torch.no_grad():
                        results = self.rna_fm_model(batch_tokens, repr_layers=[12])
                        embeddings.append(results["representations"][12].squeeze(0)[1:-1])
            token_embeddings = torch.cat(embeddings, dim=0)
        else:
            data = [rna_sequences]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.str_device)
            if self.finetune_rna_fm:
                with torch.set_grad_enabled(self.finetune_rna_fm):
                    results = self.rna_fm_model(batch_tokens, repr_layers=[12])
                    token_embeddings = results["representations"][12].squeeze(0)[1:-1]
            else:
                self.rna_fm_model.eval()
                with torch.no_grad():
                    results = self.rna_fm_model(batch_tokens, repr_layers=[12])
                    token_embeddings = results["representations"][12].squeeze(0)[1:-1]
        # mean pooling
        token_embedding = torch.mean(token_embeddings, dim=0)
        return token_embedding, token_embeddings

    def load_pretrained_model(self, model_path):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading {k} from pretrained model")
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        # print out the params that are not loaded
        not_loaded = [k for k in model_dict if k not in pretrained_dict]
        if len(not_loaded) > 0:
            print(f"Params not loaded: {not_loaded} from pretrained model")
        not_loaded = [k for k in pretrained_dict if k not in model_dict]
        if len(not_loaded) > 0:
            print(f"Params not loaded: {not_loaded} from current model")

    def inference(self, data_loader, save_path='./', fold_num=1):
        """
        Inference the model with a single RNA sequence and a single match small molecule from data loader

        Args:
            data_loader: a data loader for inference

        Returns:
            low_rank_pairs: dictionary of pairs and their rank percentile
            rank_dict: dictionary of pairs and their rank percentile
        """
        # set the model to evaluation mode
        self.eval()

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        rank_dict = {}
        all_cosine_similarity = {}  # include positive and negative samples
        all_embeddings = {}  # include positive and negative samples
        # iterate batches from data loader, without loss calculation
        for batch in data_loader:
            rna_sequences, match_smols, decoy_smols_list, rna_sequences_names, match_smols_name, contact_index_list = batch
            rna_tokenized_list, token_embeddings_list = self._rna_processing(rna_sequences)
            for anchor_rna, match_smol, decoy_smols, rna_sequence_name, match_smol_name, token_embeddings, contact_index, rna_sequence in \
                    zip(rna_tokenized_list, match_smols, decoy_smols_list, rna_sequences_names, match_smols_name, token_embeddings_list, contact_index_list, rna_sequences):
                if len(decoy_smols) == 0:
                    continue

                anchor_rna = anchor_rna.squeeze(1)
                match_smol_tokenized_list, _ = self._smol_processing([match_smol])
                match_smol_tokenized = match_smol_tokenized_list[0]

                decoy_smol_tokenized_list, _ = self._smol_processing(decoy_smols)
                decoy_distance_list = []
                match_smol_tokenized = match_smol_tokenized.unsqueeze(dim=0)
                match_distance = sigmoid_cosine_distance_test(anchor_rna, match_smol_tokenized)
                for decoy_smol_tokenized in decoy_smol_tokenized_list:
                    decoy_smol_tokenized = decoy_smol_tokenized.unsqueeze(dim=0)
                    decoy_distance = sigmoid_cosine_distance_test(anchor_rna,
                                                                  decoy_smol_tokenized)
                    decoy_distance_list.append(decoy_distance)

                rank_percentile = self._rank_eval(match_distance, decoy_distance_list)
                rank_dict[(f'{rna_sequence[0]}_{match_smol_name}', match_smol_name)] = rank_percentile

                all_cosine_similarity[f'{rna_sequence[0]}_{match_smol_name}'] = {
                    'positive_pair': match_distance.item(),
                    'negative_pair': []
                }
                all_embeddings[f'{rna_sequence[0]}_{match_smol_name}'] = {
                    'rna_anchor': anchor_rna.detach().numpy(),
                    'rna_anchor_name': rna_sequence_name,
                    'positive_smol': match_smol_tokenized.detach().numpy(),
                    'positive_name': match_smol_name,
                    'negative_smol': []
                }

                for decoy_smol_tokenized in decoy_smol_tokenized_list:
                    decoy_smol_tokenized = decoy_smol_tokenized.unsqueeze(dim=0)
                    decoy_distance = sigmoid_cosine_distance_test(anchor_rna.squeeze(1),
                                                                  decoy_smol_tokenized)
                    decoy_distance_list.append(decoy_distance)
                    all_cosine_similarity[f'{rna_sequence[0]}_{match_smol_name}']['negative_pair'].append(decoy_distance.item())
                    all_embeddings[f'{rna_sequence[0]}_{match_smol_name}']['negative_smol'].append(decoy_smol_tokenized.detach().numpy())

        # save the rank_dict to local as pkl
        with open(save_path + f'/rank_dict_fold{fold_num}.pkl', 'wb') as f:
            pickle.dump(rank_dict, f)
        with open(save_path + f'/cosine_similarity_fold{fold_num}.pkl', 'wb') as f:
            pickle.dump(all_cosine_similarity, f)
        with open(save_path + f'/embeddings_fold{fold_num}.pkl', 'wb') as f:
            pickle.dump(all_embeddings, f)
        return rank_dict, all_cosine_similarity, all_embeddings

    def inference_single_smol(self, smol_fp2):
        """
        Inference the model with a single small molecule
        :param smol_fp2:
        :return:
        """
        self.eval()
        smol_tokenized_list, _ = self._smol_processing([smol_fp2])
        smol_tokenized = smol_tokenized_list[0]
        return smol_tokenized

    def inference_list_smols(self, smol_fp2_list):
        """
        Inference the model with a list of small molecules
        :param smol_fp2_list:
        :return:
        """
        self.eval()
        smol_tokenized_list, _ = self._smol_processing(smol_fp2_list)
        return smol_tokenized_list

    def inference_single_rna(self, rna_sequence):
        """
        Inference the model with a single RNA sequence
        :param rna_sequence:
        :return:
        """
        self.eval()
        rna_tokenized_list, token_embeddings_list = self._rna_processing([('s', rna_sequence)])
        rna_tokenized = rna_tokenized_list[0]
        return rna_tokenized

    def inference_known_match(self, data_loader, save_path='./', save_name='rank_box_plot.png'):
        """
        Inference the model with a single RNA sequence and a single match small molecule from data loader

        Args:
            data_loader: a data loader for inference

        Returns:
            low_rank_pairs: dictionary of pairs and their rank percentile
            rank_dict: dictionary of pairs and their rank percentile
        """
        # set the model to evaluation mode
        self.eval()

        low_rank_pairs = {}
        # rank_dict = {}
        rank_list = []
        # iterate batches from data loader
        for batch in data_loader:
            rna_sequences, match_smols, decoy_smols_list, rna_sequences_names, match_smols_name, contact_index_list = batch
            rna_tokenized_list, token_embeddings_list = self._rna_processing(rna_sequences)

            for anchor_rna, match_smol, decoy_smols, rna_sequence_name, match_smol_name, token_embeddings, contact_index in \
                    zip(rna_tokenized_list, match_smols, decoy_smols_list, rna_sequences_names, match_smols_name, token_embeddings_list, contact_index_list):
                anchor_rna = anchor_rna.squeeze(1).to(self.str_device)
                match_smol_tokenized_list, _ = self._smol_processing([match_smol])
                match_smol_tokenized = match_smol_tokenized_list[0]

                decoy_smol_tokenized_list, _ = self._smol_processing(decoy_smols)
                decoy_distance_list = []
                for decoy_smol_tokenized in decoy_smol_tokenized_list:
                    decoy_distance = sigmoid_cosine_distance_test(anchor_rna, decoy_smol_tokenized)
                    decoy_distance_list.append(decoy_distance)

                match_distance = sigmoid_cosine_distance_test(anchor_rna, match_smol_tokenized)
                rank_percentile = self._rank_eval(match_distance, decoy_distance_list)
                # rank_dict[(rna_sequence_name, match_smol_name)] = rank_percentile
                rank_list.append(rank_percentile)
                if rank_percentile <= 0.6:
                    low_rank_pairs[(rna_sequence_name, match_smol_name)] = rank_percentile

        # draw and save box plot without calling functions and save to local
        # rank_list = list(rank_dict.values())
        buf = io.BytesIO()
        plt.boxplot(rank_list)
        plt.axhline(y=sum(rank_list) / len(rank_list), color='r', linestyle='-')
        plt.scatter([1] * len(rank_list), rank_list, alpha=0.5)
        plt.text(1.1, sum(rank_list) / len(rank_list),
                    f'mean={round(sum(rank_list) / len(rank_list), 4)}')
        plt.text(1.1, sorted(rank_list)[len(rank_list) // 2],
                    f'median={round(sorted(rank_list)[len(rank_list) // 2], 4)}')

        plt.title(f'Rank percentile distribution with total {len(rank_list)} data points')
        # save to path
        plt.savefig(buf, format='png')
        buf.seek(0)
        pil_img = Image.open(buf)
        pil_img.save(save_path + save_name)
        buf.close()
        plt.close()

        print(f'Mean rank percentile: {sum(rank_list) / len(rank_list)}')
        print(f'Median rank percentile: {sorted(rank_list)[len(rank_list) // 2]}')
        # save rank_list
        with open(save_path + save_name.split('.')[0] + '.pkl', 'wb') as f:
            pickle.dump(rank_list, f)

        return low_rank_pairs, rank_list