import math
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from deepfold.models.esm_model import MLPLayer



# attention  module
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项 Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作."""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values),self.attention_weights


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状."""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力."""
    def __init__(self,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 num_heads,
                 dropout,
                 bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None, output_attentions=True):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output,weight = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        weight_concat = transpose_output(weight, self.num_heads)
        outputs = (output_concat, weight_concat) if output_attentions else (output_concat,)
            
 
        return outputs

class MLPLayer(nn.Module):
    def __init__(self, input_size=1280, num_labels=10000, dropout_ratio=0.1):
        super().__init__()

        self.hidden_size = input_size * 2
        self.num_labels = num_labels
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.norm = nn.BatchNorm1d(self.hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_ratio)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, embeddings):
        out = self.fc1(embeddings)
        out = out.transpose(-1,-2)
        out = self.norm(out)
        out = out.transpose(-1,-2)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits

class LabelWiseAttentionModel(nn.Module):
    def __init__(self, aa_dim=1280, latent_dim=512, dropout_rate=0.0,n_head=8,nb_classes=5874,fintune=False):
        super().__init__()
        # backbone
        self._model, _ = esm.pretrained.load_model_and_alphabet(
            'esm1b_t33_650M_UR50S')

        self.nb_classes = nb_classes
        # aa_embedding transform
        self.aa_transform = nn.Linear(aa_dim, latent_dim)
        # go embedder
        self.go_embedder = nn.Embedding(nb_classes,latent_dim)
        # label-wise attention
        self.attention = MultiHeadAttention(latent_dim,
                                            latent_dim,
                                            latent_dim,
                                            latent_dim,
                                            num_heads=n_head,
                                            dropout=dropout_rate)
        self.all_gos = nn.Parameter(torch.arange(self.nb_classes),requires_grad=False)
        # classifier
        self.fc = MLPLayer(latent_dim, 1)
        self.fintune = fintune
        if not self.fintune:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self._model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, lengths=None, labels=None, output_attentions=True):
        # backbone
        model_outputs = self._model(
            input_ids,
            repr_layers=[33],
        )
        x = model_outputs['representations'][33]
        x = x[:, 1:]
        # x [B,L,C]
        # AA_embedding transform
        x = self.aa_transform(x)
        # go embedding
        go_embedding = self.go_embedder(self.all_gos)
        # label-wise attention
        y_embedding = go_embedding.repeat((x.shape[0], 1, 1))
        label_attn_embedding, weight = self.attention(y_embedding, x, x, lengths)
        # output layer
        logits = self.fc(label_attn_embedding)
        logits = logits.squeeze(-1)
        outputs = (logits, )

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.nb_classes),
                            labels.view(-1, self.nb_classes))

            outputs = (loss, logits)
        if output_attentions:
            outputs = (loss, logits, weight)
        return outputs



if __name__ == '__main__':
# data = torch.randint(low=1,high=20,size=(4,1023))
# tmp = torch.ones((4,1))*32
# data = torch.concat((tmp,data),dim=1)
# data = data.int()
# data.shape

# len_list = [100,150,999,1023]
# labels = torch.zeros((4,5874))
# labels = labels.to('cuda:0')

# backbone, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# model = LabelWiseAttentionModel(backbone)
# data = data.to('cuda:0')
# model = model.to('cuda:0')
# out = model(data,labels=labels)
    import esm
    from torch.utils.data import DataLoader
    from deepfold.data.esm_dataset import EsmDataset

    # backbone, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = LabelWiseAttentionModel()
    data_path = '../../data'
    train_dataset = EsmDataset(data_path=data_path,
                                   file_name='train_data.pkl')
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=4,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )
    for batch in train_loader:
        data = batch['input_ids']
        valid_len = batch['lengths']
        labels = batch['labels']
        data, valid_len, labels = data.cuda(), valid_len.cuda(), labels.cuda()
        model = LabelWiseAttentionModel(1280,512,0.0,8,5874)
        model = model.cuda()
        outs = model(data,valid_len,labels,True)
        print(f'loss:{outs[0]}')
        print(f'logits:{outs[1].shape}')
        print(f'attention_weights:{outs[2].shape}')
        break

