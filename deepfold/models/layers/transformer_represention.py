import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=hidden_size,
                               out_channels=256,
                               kernel_size=1,
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=256,
                               out_channels=1,
                               kernel_size=2,
                               padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.permute(0, 2, 1)
        cnn_embeddings = self.conv1(hidden_states)
        cnn_embeddings = F.relu(cnn_embeddings)
        cnn_embeddings = self.conv1(cnn_embeddings)
        return cnn_embeddings


class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size, seq_len=1024, dropout_rate=0.) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.seq_len = seq_len

        self.q = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.k = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.att = nn.Linear(self.seq_len, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, all_hidden_states, item_seq=None):
        q = self.q(all_hidden_states)
        k = self.k(all_hidden_states)
        k = torch.transpose(k, 1, 2)
        q = torch.matmul(q, k)
        q = torch.softmax(q, dim=-1)
        q = self.att(q)
        q = torch.softmax(q, dim=1)
        q = torch.sum(q * all_hidden_states, 1)
        return q


class AttentionPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(loc=0.0,
                                scale=0.1,
                                size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([
            all_hidden_states[layer_i][:, 0].squeeze()
            for layer_i in range(1, self.num_hidden_layers + 1)
        ],
                                    dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers,
                                           self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


class AttentionPooling2(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.):
        super(AttentionPooling2, self).__init__()

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, all_hidden_states, item_seq=None):
        att_net = self.fc1(all_hidden_states)
        att_net = self.tanh(att_net)
        att_net = self.dropout(att_net)
        att_net = self.fc2(att_net)
        if item_seq is not None:  # padding sequence length
            pass
        att_net = self.softmax(att_net)

        att_net = torch.sum(all_hidden_states * att_net,
                            1)  # batch_size*seq_len*1

        return att_net


class WKPooling(nn.Module):
    def __init__(self, layer_start: int = 4, context_window_size: int = 2):
        super(WKPooling, self).__init__()
        self.layer_start = layer_start
        self.context_window_size = context_window_size

    def forward(self, all_hidden_states, attention_mask):
        ft_all_layers = all_hidden_states
        org_device = ft_all_layers.device
        all_layer_embedding = ft_all_layers.transpose(1, 0)
        all_layer_embedding = all_layer_embedding[:, self.
                                                  layer_start:, :, :]  # Start from 4th layers output

        # torch.qr is slow on GPU (see https://github.com/pytorch/pytorch/issues/22573). So compute it on CPU until issue is fixed
        all_layer_embedding = all_layer_embedding.cpu()

        attention_mask = attention_mask.cpu().numpy()
        unmask_num = np.array([sum(mask) for mask in attention_mask
                               ]) - 1  # Not considering the last item
        embedding = []

        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[
                sent_index, :, :unmask_num[sent_index], :]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                # 'Unified Word Representation'
                token_embedding = self.unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            # features.update({'sentence_embedding': features['cls_token_embeddings']})

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(sentence_feature,
                                                     one_sentence_embedding)
            embedding.append(sentence_embedding)

        output_vector = torch.stack(embedding).to(org_device)
        return output_vector

    def unify_token(self, token_feature):
        # Unify Token Representation
        window_size = self.context_window_size

        alpha_alignment = torch.zeros(token_feature.size()[0],
                                      device=token_feature.device)
        alpha_novelty = torch.zeros(token_feature.size()[0],
                                    device=token_feature.device)

        for k in range(token_feature.size()[0]):
            left_window = token_feature[k - window_size:k, :]
            right_window = token_feature[k + 1:k + window_size + 1, :]
            window_matrix = torch.cat(
                [left_window, right_window, token_feature[k, :][None, :]])
            Q, R = torch.qr(window_matrix.T)

            r = R[:, -1]
            alpha_alignment[k] = torch.mean(
                self.norm_vector(R[:-1, :-1], dim=0), dim=1).matmul(
                    R[:-1, -1]) / torch.norm(r[:-1])
            alpha_alignment[k] = 1 / (alpha_alignment[k] *
                                      window_matrix.size()[0] * 2)
            alpha_novelty[k] = torch.abs(r[-1]) / torch.norm(r)

        # Sum Norm
        alpha_alignment = alpha_alignment / torch.sum(
            alpha_alignment)  # Normalization Choice
        alpha_novelty = alpha_novelty / torch.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment
        alpha = alpha / torch.sum(alpha)  # Normalize

        out_embedding = torch.mv(token_feature.t(), alpha)
        return out_embedding

    def norm_vector(self, vec, p=2, dim=0):
        # Implements the normalize() function from sklearn
        vec_norm = torch.norm(vec, p=p, dim=dim)
        return vec.div(vec_norm.expand_as(vec))

    def unify_sentence(self, sentence_feature, one_sentence_embedding):
        # Unify Sentence By Token Importance
        sent_len = one_sentence_embedding.size()[0]

        var_token = torch.zeros(sent_len, device=one_sentence_embedding.device)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:, token_index, :]
            sim_map = self.cosine_similarity_torch(token_feature)
            var_token[token_index] = torch.var(sim_map.diagonal(-1))

        var_token = var_token / torch.sum(var_token)
        sentence_embedding = torch.mv(one_sentence_embedding.t(), var_token)

        return sentence_embedding

    def cosine_similarity_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_layers
        self.layer_weights = layer_weights if layer_weights is not None else nn.Parameter(
            torch.tensor([1] * (num_layers + 1 - layer_start),
                         dtype=torch.float))

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(
            dim=0) / self.layer_weights.sum()
        return weighted_average


class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size,
                            self.hiddendim_lstm,
                            batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([
            all_hidden_states[layer_i][:, 0].squeeze()
            for layer_i in range(1, self.num_hidden_layers + 1)
        ],
                                    dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers,
                                           self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out
