import torch
from torch import Tensor
import torch.nn as nn

from .PositionalEncoding import PositionalEncoding
from .RelativePositionalRepresentations import CustomTransformerEncoder
from .rpr import TransformerEncoderLayerRPR, TransformerEncoderRPR
from .progress import LearningProgress


class MORTM(nn.Module):
    token_dict = {
        'SHIFT': [1, 6],
        'START': [7, 38],
        'PITCH': [39, 166],
        'DURATION': [167, 266]
    }

    def __init__(self, vocab_size, progress: LearningProgress, trans_layer=6, num_heads=8, d_model=512,
                 dim_feedforward=1024, dropout=0.1,
                 position_length=2048, use_rpr=True):
        super(MORTM, self).__init__()

        self.progress = progress
        self.trans_layer = trans_layer
        self.num_heads = num_heads
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        #位置エンコーディングを作成
        #self.positional: LearnablePositionalEncoding = LearnablePositionalEncoding(self.d_model, progress, dropout, position_length).to(self.progress.get_device())
        self.positional: PositionalEncoding = PositionalEncoding(self.d_model, progress, dropout, position_length).to(self.progress.get_device())
        #Transformerの設定
        if not use_rpr:
            self.transformer: nn.Transformer = nn.Transformer(d_model=self.d_model, nhead=num_heads,  #各種パラメーターの設計
                                                              num_encoder_layers=self.trans_layer,
                                                              num_decoder_layers=0,
                                                              dropout=self.dropout, dim_feedforward=dim_feedforward,
                                                              custom_decoder=DummyDecoder()
                                                              ).to(self.progress.get_device())
        else:
            encoder_norm = nn.LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.num_heads, self.dim_feedforward, self.dropout, er_len=position_length)
            encoder = TransformerEncoderRPR(encoder_layer, self.trans_layer, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.num_heads, num_encoder_layers=self.trans_layer,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.dim_feedforward, custom_decoder=DummyDecoder(), custom_encoder=encoder
            ).to(device=progress.get_device())

            print("Use RPR Transformer")
        print(f"Input Vocab Size:{vocab_size}")
        self.Wout: nn.Linear = nn.Linear(self.d_model, vocab_size).to(self.progress.get_device())

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, self.d_model).to(self.progress.get_device())
        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(self.progress.get_device())

    def forward(self, inputs_seq, input_padding_mask=None):
        mask = self.transformer.generate_square_subsequent_mask(inputs_seq.shape[1]).to(self.progress.get_device())

        inputs_em: Tensor = self.embedding(inputs_seq)
        inputs_em = inputs_em.permute(1, 0, 2)

        inputs_pos: Tensor = self.positional(inputs_em)

        #print(inputs_pos.shape, tgt_pos.shape)
        out: Tensor = self.transformer(inputs_pos, inputs_pos, src_mask=mask,
                                       src_key_padding_mask=input_padding_mask, tgt_key_padding_mask=input_padding_mask)

        out.permute(1, 0, 2)

        score:Tensor = self.Wout(out)
        return score.to(self.progress.get_device())

    def generate_by_length(self, input_sequence, top_k=3, max_length=100):
        self.eval()
        if not isinstance(input_sequence, torch.Tensor):
            input_sequence = torch.tensor(input_sequence, dtype=torch.long, device=self.progress.get_device())


            for _ in range(max_length):
              #print(f"I{i} input_tensor")

                # モデルに入力して次のトークンのスコアを取得 (3次元で返ってくる)
                with torch.no_grad():
                    input_sequence = self.generate_note(input_sequence, top_k)
        return input_sequence

    def generate_note(self, input_sequence, top_k):
        if len(input_sequence) != 0:
            fill_count = len(input_sequence) % 4
        else:
            fill_count = 0
        for i in range(fill_count, 4):
            input_tensor = input_sequence.unsqueeze(0)  # (1, sequence_length)

            score = self(input_tensor)
            logits = score[:, -1, :]
            logits = logits[-1, :]
            if i == 0:
                token_id = self.sampling(logits, top_k=1, temperature=0.5)
            elif i == 1:
                token_id = self.sampling(logits, top_k=2, temperature=0.5)
            elif i == 2:
                token_id = self.sampling(logits, top_k=10, temperature=1.2)
            else:
                token_id = self.sampling(logits, top_k=2, temperature=0.5)

            input_sequence = torch.cat((input_sequence, torch.tensor([token_id], device=self.progress.get_device()))).to(self.progress.get_device())

        return input_sequence







    def sampling(self, logits: Tensor, top_k: int, temperature: float)-> int:
        logits_t = logits / temperature
        probs = self.softmax(logits_t)
        # トップKの確率でトークンをフィルタリング
        sorted_probs, sorted_indices = torch.topk(probs, top_k)

        # 再度正規化
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # トークンをサンプリング
        sampled_index = torch.multinomial(sorted_probs, 1).item()  # サンプリングされたインデックスを取得

        # ソートされたインデックスから元のインデックスに変換
        next_token = sorted_indices[sampled_index].item()
        return next_token


    def top_k_sampling_with_temperature_sequence(self, input_sequence, temperature=1.0, top_k=3, max_length=100):
        self.eval()
        """
        MORTMモデルにトップKサンプリングと温度シーケンスを実装する関数

        Args:
            input_sequence: 1次元の入力シーケンス (List[int] or torch.Tensor)。
            temperature: 温度パラメータ。デフォルトは1.0。
            top_k: サンプリングする上位K個のトークンの数。デフォルトは3。
            max_length: 生成するシーケンスの最大長さ。デフォルトは100。

        Returns:
            1次元の生成されたシーケンス (torch.Tensor)。
        """

        # 入力シーケンスをtorch.tensorに変換
        if not isinstance(input_sequence, torch.Tensor):
            input_sequence = torch.tensor(input_sequence, dtype=torch.long, device=self.progress.get_device())

        # 入力シーケンスの長さを取得
        input_length = input_sequence.size(0)

        # 生成されたシーケンスを格納するリスト
        generated_sequence = input_sequence.tolist()

        # 生成をループ
        for i in range(max_length):
            # モデルに渡すための入力の準備 (2次元に変換)
            input_tensor = input_sequence.unsqueeze(0)  # (1, sequence_length)
            #print(f"I{i} input_tensor")

            # モデルに入力して次のトークンのスコアを取得 (3次元で返ってくる)
            with torch.no_grad():
                mask = self.transformer.generate_square_subsequent_mask(input_tensor.shape[1]).to(
                    self.progress.get_device())
                scores = self(input_tensor)  # (1, sequence_length, vocab_size)

            # 最新のトークンのスコアを取得 (最後のトークンに対するスコア)
            logits = scores[:, -1, :]  # (1, vocab_size)

            # 温度の適用
            logits = logits / temperature
            logits = logits[-1, :]

            # ソフトマックスを適用して確率を取得
            probs = self.softmax(logits)  # (vocab_size)

            # トップKの確率でトークンをフィルタリング
            sorted_probs, sorted_indices = torch.topk(probs, top_k)
            #print(sorted_probs, sorted_indices)

            # 再度正規化
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # トークンをサンプリング
            sampled_index = torch.multinomial(sorted_probs, 1).item()  # サンプリングされたインデックスを取得

            # ソートされたインデックスから元のインデックスに変換
            next_token = sorted_indices[sampled_index].item()

#            print(next_token)

            # シーケンスにトークンを追加
            generated_sequence.append(next_token)

            # 次のステップの入力として準備
            input_sequence = torch.tensor(generated_sequence, dtype=torch.long, device=self.progress.get_device())

        return input_sequence

    def top_p_sampling(self, input_ids, tokenizer, p=0.8, max_length=20, temperature=0.2):
        self.eval()
        output = torch.tensor([input_ids], dtype=torch.long).to(self.progress.get_device())
        for _ in range(max_length):
            with torch.no_grad():
                mask = self.transformer.generate_square_subsequent_mask(output.shape[1]).to(self.progress.get_device())
                outputs = self( output, output, None, None)

                #print(outputs)

                logits = outputs[:, -1, :]

                #print(logits)


                logits = logits / temperature



                sorted_logits, sorted_indices = torch.sort(logits, descending=True)



                #cumulative_probs = torch.cumsum(self.softmax(sorted_logits), dim=-1)
                cumulative_probs = self.softmax(sorted_logits)

                print(cumulative_probs.argmax())

                sorted_indices_to_remove = cumulative_probs > p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

                probabilities = self.softmax(logits)[-1]

                print(probabilities)

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
