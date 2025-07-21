from .modules.layers import *
from .modules.config import MORTMArgs

from flash_attn.bert_padding import unpad_input, pad_input



class Critic(nn.Module):
    def __init__(self, args: MORTMArgs, progress):
        super().__init__()
        # Actorと同様のEmbeddingとDecoderを持つ
        self.embedding = nn.Embedding(args.vocab_size, args.d_model, padding_idx=0)
        self.decoder = MORTMDecoder(args, progress=progress)

        # 価値を出力するためのヘッド
        self.value_head = nn.Linear(args.d_model, 1)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        各トークン位置での「価値」を計算して返す。

        Returns:
            torch.Tensor: shape (batch_size, seq_len, 1) の価値テンソル
        """
        # ユーザーのMORTMのforward前半部分を参考
        x = self.embedding(input_ids).to(dtype=torch.bfloat16)

        cu_seqlens, max_s = None, None
        indices = None
        batch_size, seq_len = x.shape[0], x.shape[1]

        if attention_mask is not None:
            # flash_attn用のunpad処理
            x, indices, cu_seqlens, max_s, _ = unpad_input(x, attention_mask)

        # デコーダーを通して隠れ状態を取得
        hidden_states = self.decoder(tgt=x, tgt_is_causal=True, cu_seqlens=cu_seqlens, max_seqlen=max_s)

        if attention_mask is not None:
            # pad処理で元のシーケンス長に戻す
            hidden_states = pad_input(hidden_states, indices, batch_size, seq_len)

        # 価値ヘッドを通してスカラー値に変換
        values = self.value_head(hidden_states)

        return values

class BERTM(nn.Module):

    def __init__(self, args: MORTMArgs, progress):
        super(BERTM, self).__init__()
        self.args = args # argsを保存しておくと便利
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.decoder = MORTMDecoder(args=args,
                                    progress=progress)
        self.attn_pool = Pool(args)
        self.hidden = nn.Linear(args.d_model, args.d_model // 2)
        self.Wout = nn.Linear(args.d_model // 2, 1) # linear層の出力次元に合わせる

    def forward(self, x: Tensor, padding_mask=None):
        x: Tensor = self.embedding(x).to(dtype=torch.bfloat16)

        if padding_mask is not None:
            x, indices, cu_seqlens, max_s, used_seqlens = unpad_input(x, padding_mask)
        else:
            indices = cu_seqlens = max_s = used_seqlens = None

        out = self.decoder(tgt=x, tgt_is_causal=False, cu_seqlens=cu_seqlens, max_seqlen=max_s)

        out = self.attn_pool(out, cu_seqlens if cu_seqlens is not None else torch.tensor([0, len(x)], dtype=torch.int32, device=x.device))  # バッチサイズをcu_seqlensに設定

        out = self.hidden(out)
        hid = F.relu(out)
        score = self.Wout(hid)

        return score