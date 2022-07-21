import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from .gnn_model import GCN, Embedder


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        seq_feature = self.Linear(x)

        return seq_feature


class ProtGCNModel(nn.Module):
    """ ProtGCNModel: Kyudam Choi, Yurim Lee, Cheongwon Kim, Minsung Yoon \
        An Effective GCN-based Hierarchical Multi-label classification for Protein Function Prediction \
        arxiv, http://arxiv.org/abs/2112.02810
    """
    def __init__(self,
                 nodes: torch.Tensor,
                 adjmat: torch.Tensor,
                 seq_dim: int = 1024,
                 node_feats: int = 512,
                 hidden_dim: int = 512):
        super().__init__()
        assert nodes.shape[0] == adjmat.shape[0]
        self.nodesMat = nodes
        self.adjMat = adjmat
        self.num_nodes = nodes.shape[0]
        self.seq_mlp = MLP(seq_dim, hidden_dim)
        self.graph_embedder = Embedder(self.num_nodes, node_feats)
        # self.gcn = CustomGCN(node_feats, hidden_dim, dropout)
        self.gcn = GCN(node_feats, hidden_dim)
        self.num_labels = self.num_nodes
        self.fc = nn.Linear(self.num_nodes,self.num_nodes)

    def forward(self, embeddings, labels):
        seq_out = self.seq_mlp(embeddings)
        node_embd = self.graph_embedder(self.nodesMat)
        graph_out = self.gcn(node_embd, self.adjMat)
        graph_out = graph_out.transpose(-2, -1)

        logits = self.fc(torch.matmul(seq_out, graph_out))
        outputs = (logits, )
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))

            outputs = (loss, ) + outputs

        return outputs


class ProtPubMedBert(nn.Module):
    """ProtPubMedBert."""
    def __init__(self,
                 embeds: torch.Tensor,
                 seq_dim: int = 1280,
                 hidden_dim: int = 512):

        super(ProtPubMedBert, self).__init__()
        self.embeds = embeds
        emb_dim = self.embeds.shape[1]
        self.fc_seq = nn.Linear(seq_dim, hidden_dim)
        self.fc_embed = nn.Linear(emb_dim, hidden_dim)
        self.num_labels = self.embeds.shape[0]

    def forward(self, embeddings, labels):
        seq_out = self.fc_seq(embeddings)
        embed_out = self.fc_embed(self.embeds)
        embed_out = embed_out.transpose(-2, -1)

        logits = torch.matmul(seq_out, embed_out)
        outputs = (logits, )
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))

            outputs = (loss, ) + outputs

        return outputs


if __name__ == '__main__':
    embeds = torch.ones((100, 2000))
    model = ProtPubMedBert(embeds=embeds)
    print(model)
