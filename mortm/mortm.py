import torch
from torch import Tensor
import torch.nn as nn

from .PositionalEncoding import PositionalEncoding
from .RelativePositionalRepresentations import CustomTransformerEncoder
from .rpr import TransformerEncoderLayerRPR, TransformerEncoderRPR
from .progress import LearningProgress
from torch.distributions import Categorical

class MORTM(nn.Module):
    token_dict = {
        'SHIFT': [1, 6],
        'START': [7, 38],
        'PITCH': [39, 166],
        'DURATION': [167, 266]
    }

    def __init__(self, vocab_size, progress: LearningProgress, trans_layer=6, num_heads=8, d_model=512,
                 dim_feedforward=1024, dropout=0.1,
                 position_length=2048, use_rpr=True, use_decoder=False):
        super(MORTM, self).__init__()

        if use_decoder:
            print("Use decoder mode!")

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
                                                              num_decoder_layers= self.trans_layer if use_decoder else 0, #デコーダーは使うときと使わない時がある。
                                                              dropout=self.dropout, dim_feedforward=dim_feedforward,
                                                              custom_decoder=None if use_decoder else DummyDecoder()
                                                              ).to(self.progress.get_device())
        else:
            encoder_norm = nn.LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.num_heads, self.dim_feedforward, self.dropout, er_len=position_length)
            encoder = TransformerEncoderRPR(encoder_layer, self.trans_layer, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.num_heads, num_encoder_layers=self.trans_layer,
                num_decoder_layers=self.trans_layer if use_decoder else 0,
                dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.dim_feedforward,
                custom_decoder=None if use_decoder else DummyDecoder(),
                custom_encoder=encoder
            ).to(device=progress.get_device())

            print("Use RPR Transformer")
        print(f"Input Vocab Size:{vocab_size}")
        self.Wout: nn.Linear = nn.Linear(self.d_model, vocab_size).to(self.progress.get_device())

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, self.d_model).to(self.progress.get_device())
        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(self.progress.get_device())

    def forward(self, inputs_seq, tgt_seq=None, src_mask=None, tgt_mask=None, input_padding_mask=None, tgt_padding_mask=None):
        if tgt_seq is None or src_mask is None:
            mask = self.transformer.generate_square_subsequent_mask(inputs_seq.shape[1]).to(self.progress.get_device())
        elif src_mask is not None:
            mask = src_mask
        else:
            mask = None

        inputs_em: Tensor = self.embedding(inputs_seq)
        inputs_em = inputs_em.permute(1, 0, 2)

        inputs_pos: Tensor = self.positional(inputs_em)

        if tgt_seq is not None:
            tgt_e = self.embedding(tgt_seq)
            tgt_e = tgt_e.permute(1, 0, 2)
            tgt_p = self.positional(tgt_e)
        else:
            tgt_p = inputs_pos

        out: Tensor = self.transformer(inputs_pos, tgt_p, src_mask=mask, tgt_mask=tgt_mask,
                                       src_key_padding_mask=input_padding_mask, tgt_key_padding_mask=tgt_padding_mask)

        out.permute(1, 0, 2)

        score:Tensor = self.Wout(out)
        return score.to(self.progress.get_device())

    def top_k_sampling_length_encoder(self, input_sequence, temperature=1.0, top_k=3, max_length=100):
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
        log_prob_list = []
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
            print(f"\r Generating...{i / max_length * 100}%", end="")
            # モデルに入力して次のトークンのスコアを取得 (3次元で返ってくる)
            with torch.no_grad():
                mask = self.transformer.generate_square_subsequent_mask(input_tensor.shape[1]).to(
                    self.progress.get_device())
                #print(input_tensor.shape)
                scores = self(input_tensor)  # (1, sequence_length, vocab_size)
                #print(f"SCORE: {scores.shape}")

            # 最新のトークンのスコアを取得 (最後のトークンに対するスコア)
            logits = scores[:, -1, :]  # (1, vocab_size)

            # 温度の適用
            logits = logits / temperature
            logits = logits[-1, :]

            # ソフトマックスを適用して確率を取得
            probs = self.softmax(logits)  # (vocab_size)

            # トップKの確率でトークンをフィルタリング
            topk_probs, topk_indices = torch.topk(probs, top_k)
            #print(sorted_probs, sorted_indices)

            # 再度正規化
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            # トークンをサンプリング
            distribution = Categorical(topk_probs)
            sampled_index = distribution.sample()

            # ソートされたインデックスから元のインデックスに変換
            next_token = topk_indices[sampled_index].item()

            log_probs = torch.log(topk_probs[sampled_index])
            log_prob_list.append(log_probs)
#            print(next_token)

            # シーケンスにトークンを追加
            generated_sequence.append(next_token)

            # 次のステップの入力として準備
            input_sequence = torch.tensor(generated_sequence, dtype=torch.long, device=self.progress.get_device())

        return input_sequence, torch.tensor(log_prob_list, device=self.progress.get_device())

    def argmax_sampling_encoder(self, input_sequence, max_length=500):
        #self.eval()
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
        log_prob_list = []
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

            print(f"\r Generating...{i / max_length * 100}%", end="")
            # モデルに入力して次のトークンのスコアを取得 (3次元で返ってくる)
            mask = self.transformer.generate_square_subsequent_mask(input_tensor.shape[1]).to(
                self.progress.get_device())

            scores = self(input_tensor)  # (1, sequence_length, vocab_size)

            # 最新のトークンのスコアを取得 (最後のトークンに対するスコア)
            logits = scores[:, -1, :]  # (1, vocab_size)
            logits = logits[-1, :]

            # ソフトマックスを適用して確率を取得
            probs: Tensor = self.softmax(logits)  # (vocab_size)
            d = Categorical(probs)
            next_token = d.sample()

            log_probs = d.log_prob(next_token)
            log_prob_list.append(log_probs)
            # シーケンスにトークンを追加
            generated_sequence.append(next_token)

            # 次のステップの入力として準備
            input_sequence = torch.tensor(generated_sequence, dtype=torch.long, device=self.progress.get_device())

        return input_sequence, torch.stack(log_prob_list).to(self.progress.get_device())


    def top_k_sampling_length_decoder(self, input_sequence, temperature=1.0, top_k=3, max_length=100):
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
        generated_sequence = [11]

        # 生成をループ
        for i in range(max_length):
            # モデルに渡すための入力の準備 (2次元に変換)
            tgt = torch.tensor(generated_sequence, device=self.progress.get_device()).unsqueeze(0)  # (1, sequence_length)
            src = input_sequence.unsqueeze(0)
            #print(f"I{i} input_tensor")

            # モデルに入力して次のトークンのスコアを取得 (3次元で返ってくる)
            with torch.no_grad():
                mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(
                    self.progress.get_device())
                print(src.shape)
                scores = self(src, tgt_seq=tgt)  # (1, sequence_length, vocab_size)
                print(f"SCORE: {scores.shape}")

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
#            input_sequence = torch.tensor(generated_sequence, dtype=torch.long, device=self.progress.get_device())
        input_sequence = torch.cat((input_sequence, torch.tensor(generated_sequence, device=self.progress.get_device())))
        return input_sequence

    def top_p_sampling_length(self, input_seq, p=0.8, max_length=20, temperature=0.2):
        self.eval()
        if not isinstance(input_seq, torch.Tensor):
            input_seq = torch.tensor(input_seq, dtype=torch.long, device=self.progress.get_device())

        generated = input_seq.tolist()
        for _ in range(max_length):
            logits= self(input_seq)
            token = self.top_p_sampling(logits, p=p, temperature=temperature)
            generated.append(token)

        input_seq = torch.tensor(generated, dtype=torch.long, device=self.progress.get_device())
        return input_seq

    def top_p_sampling(self, logits, p=0.9, temperature=1.0)-> int:

        logits = logits / temperature
        # logitsをソフトマックスで確率分布に変換
        probs = self.softmax(logits, dim=-1)
        # 確率の降順に並べ替え、そのインデックスを取得
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # 累積確率を計算
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 累積確率がpを超えるインデックスを取得
        cutoff_index = torch.where(cumulative_probs > p)[0][0]

        # 上位pに入らないトークンの確率を0にする
        sorted_probs[cutoff_index + 1:] = 0

        # 確率を再正規化
        sorted_probs /= torch.sum(sorted_probs)

        # トークンをサンプリング
        sampled_index = torch.multinomial(sorted_probs, 1)

        # インデックスを元の順序に戻す
        return sorted_indices[sampled_index].item()


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, **kwargs):
        return memory
