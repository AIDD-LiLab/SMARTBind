from torch import nn
import torch


class CrossAttention(nn.Module):
    def __init__(self,
                 smol_dim: int = 512,
                 rna_dim: int = 640,
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.2,
                 attention_dropout_rate: float = 0.3):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.rna_dim = rna_dim

        self.smol_projection = nn.Linear(smol_dim, rna_dim, bias=True)
        self.projection = nn.Linear(rna_dim+smol_dim, smol_dim, bias=True)
        self.batch_norm_projection = nn.BatchNorm1d(smol_dim)
        self.projection_leaky_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.self_attention = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(smol_dim, num_heads=8, dropout=attention_dropout_rate),
                nn.LayerNorm(smol_dim)
            )
        ])

        self.mlp = nn.Sequential(
            nn.Linear(smol_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid()
        )

    @staticmethod
    def positional_encoding(seq_length, dim):
        pos = torch.arange(seq_length).unsqueeze(1)
        i = torch.arange(dim // 2).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * i) / dim)
        angle_rads = pos * angle_rates

        sines = torch.sin(angle_rads)
        cosines = torch.cos(angle_rads)

        pos_encoding = torch.cat([sines, cosines], dim=-1)
        pos_encoding = pos_encoding[:seq_length, :]
        return pos_encoding

    def forward(self, smol, rna_sequence):
        seq_length = rna_sequence.size(0)
        pos_encoding = self.positional_encoding(seq_length, self.rna_dim)
        rna_sequence_pe = rna_sequence + pos_encoding.to(rna_sequence.device)

        smol_repeated = smol.repeat(rna_sequence.size(0), 1, 1)  # L, 512
        smol_repeated = smol_repeated.squeeze()

        if len(smol_repeated.size()) == 1:
            smol_repeated = smol_repeated.unsqueeze(0)
        rna_smol_concat = torch.cat([rna_sequence_pe, smol_repeated], dim=-1)  # (L, 640) + (L, 512) -> (L, 1152)

        rna_smol_concat_hidden = self.projection_leaky_relu(self.batch_norm_projection(self.projection(rna_smol_concat)))

        for layer in self.self_attention:
            attention_output, _ = layer[0](rna_smol_concat_hidden, rna_smol_concat_hidden, rna_smol_concat_hidden)
            rna_smol_concat_hidden = layer[1](attention_output)

        smol_repeated = smol.repeat(rna_sequence.size(0), 1, 1).squeeze()
        rna_smol_concat_residual = rna_smol_concat_hidden + smol_repeated

        output = self.mlp(rna_smol_concat_residual)
        return output


class BindingPredictor(nn.Module):
    def __init__(self, smol_dim, rna_dim, hidden_dim, dropout_rate=0.2, attention_dropout_rate=0.3):
        super(BindingPredictor, self).__init__()
        self.cross_attention = CrossAttention(smol_dim, rna_dim, hidden_dim, dropout_rate, attention_dropout_rate)

    def forward(self, smol, rna_sequence):
        contact_output = self.cross_attention(smol, rna_sequence)
        return contact_output
