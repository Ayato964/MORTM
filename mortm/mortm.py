import torch
from torch import Tensor
import torch.nn as nn
from .PositionalEncoding import PositionalEncoding

from .progress import LearningProgress


class MORTM(nn.Module):
    token_dict = {
        0: "600_633",
        1: "10_139",
        2: "300_429",
        3: "500_600",
        4: "600_605"
    }

    def __init__(self, vocab_size, progress: LearningProgress, trans_layer=6, num_heads=8, d_model=512,
                 dim_feedforward=1024, dropout=0.1,
                 position_length=2048):
        super(MORTM, self).__init__()

        self.progress = progress
        self.trans_layer = trans_layer
        self.num_heads = num_heads
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        #位置エンコーディングを作成
        self.positional: PositionalEncoding = PositionalEncoding(self.d_model, dropout, position_length).to(
            self.progress.get_device())
        #Transformerの設定
        self.transformer: nn.Transformer = nn.Transformer(d_model=self.d_model, nhead=num_heads,  #各種パラメーターの設計
                                                          num_encoder_layers=self.trans_layer,
                                                          num_decoder_layers=self.trans_layer,
                                                          dropout=self.dropout, dim_feedforward=dim_feedforward,
                                                          ).to(self.progress.get_device())
        print(f"Input Vocab Size:{vocab_size}")
        self.Wout: nn.Linear = nn.Linear(self.d_model, vocab_size).to(self.progress.get_device())

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, self.d_model).to(self.progress.get_device())
        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(self.progress.get_device())

    def forward(self, inputs_seq, tgt_seq, input_mask, tgt_mask, input_padding_mask, tgt_padding_mask):

        inputs_em: Tensor = self.embedding(inputs_seq)
        inputs_em = inputs_em.permute(1, 0, 2)
        inputs_pos: Tensor = self.positional(inputs_em)

        if tgt_seq is None:
            tgt_pos = inputs_pos
        else:
            tgt_em: Tensor = self.embedding(tgt_seq)
            tgt_em = tgt_em.permute(1, 0, 2)
            tgt_pos: Tensor = self.positional(tgt_em)

        #print(inputs_pos.shape, tgt_pos.shape)
        out: Tensor = self.transformer(inputs_pos, tgt_pos, input_mask, tgt_mask,
                                       src_key_padding_mask=input_padding_mask, tgt_key_padding_mask=tgt_padding_mask)

        out.permute(1, 0, 2)

        score:Tensor = self.Wout(out)
        return score.to(self.progress.get_device())

    def generate_by_length(self, input_seq, max_length, p=0.9, temperature=0.1):
        self.eval()
        output = torch.tensor([input_seq], dtype=torch.long).unsqueeze(1).to(self.progress.get_device())
        for _ in range(max_length):
            with torch.no_grad():
                output = self._next_note_token(output)

        return output

    def _next_note_token(self, output, p=0.9, temperature=0.1):
        isEnd = False
        token_count = 0
        while not isEnd:
            mask = self.transformer.generate_square_subsequent_mask(output.shape[1]).to(self.progress.get_device())
            outputs = self(output, output, mask, mask, None, None)
            logits = outputs[:, -1, :]

            str_token_duration = self.token_dict[token_count]
            parts = str_token_duration.split("_")
            token_duration = [int(part) for part in parts]
            if token_count != 4:
                token = self._next_token(logits[:, token_duration[0]:token_duration[1]])
                token += token_duration[0]
            else:
                token = self._next_token(logits)
                isEnd = True
            token_count += 1

            output = torch.cat((output.flatten(), token.unsqueeze(0))).unsqueeze(0).to(self.progress.get_device())

        return output

    def _next_token(self, end_logit_score: Tensor, p=0.9, temperature=0.1):
        end_logit_score = end_logit_score / temperature

        sorted_logits, sorted_indices = torch.sort(end_logit_score, descending=True)
        cumulative_probs = torch.cumsum(self.softmax(sorted_logits), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        end_logit_score[:, indices_to_remove] = -float('Inf')

        score = self.softmax(end_logit_score)[-1]
        dis = torch.distributions.categorical.Categorical(probs=score)
        next_token = dis.sample()

        #print(next_token)
        return next_token

    def top_p_sampling(self, input_ids, tokenizer, p=0.9, max_length=20, temperature=0.2):
        self.eval()
        output = torch.tensor([input_ids], dtype=torch.long).unsqueeze(1).to(self.progress.get_device())
        for _ in range(max_length):
            with torch.no_grad():
                mask = self.transformer.generate_square_subsequent_mask(output.shape[1]).to(self.progress.get_device())
                outputs = self(output, output, mask, mask, None, None)
                logits = outputs[:, -1, :]

                #print(logits[-1, 10:138])

                logits = logits / temperature

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(self.softmax(sorted_logits), dim=-1)
                sorted_indices_to_remove = cumulative_probs > p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

                probabilities = self.softmax(logits)[-1]
                dis = torch.distributions.categorical.Categorical(probs=probabilities)
                next_token = dis.sample()
                # バッチサイズを一致させるために次元を調整
                next_token = next_token.unsqueeze(0)
                output = torch.cat((output.flatten(), next_token)).unsqueeze(0).to(self.progress.get_device())
                #print(output)
        return output.tolist()


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, **kwargs):
        return memory
