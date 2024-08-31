import random
import time

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import constants
from .PositionalEncoding import PositionalEncoding
from messager import Messenger
IS_DEBUG = False


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def send_prediction_end_time(message,loader_len, begin_time, end_time,
                             vocab_size: int, num_epochs: int, trans_layer, num_heads, d_model,
                             dim_feedforward, dropout, position_length):
    t = end_time - begin_time
    end_time_progress = (t * loader_len * num_epochs) / 3600
    message.send_mail("終了見込みについて",
                      f"現在学習が進行しています。\n"
                      f"今回設定したパラメータに基づいて終了時刻を計算しました。\n"
                      f"ボキャブラリーサイズ:{vocab_size}\n"
                      f"エポック回数:{num_epochs}\n"
                      f"Transformerのレイヤー層:{trans_layer}\n"
                      f"Modelの次元数:{d_model}\n"
                      f"シーケンスの長さ:{dim_feedforward}\n"
                      f"ドロップアウト:{dropout}\n"
                      f"\n\n シーケンスの1回目の処理が終了しました。かかった時間は{t:.1f}秒でした。\n"
                      f"終了見込み時間は{end_time_progress:.2f}時間です"
                      )
# デバイスを取得
def set_train_data(directory, datasets):
    if not IS_DEBUG:
        print("Generating TrainData.....")
        t_data = MORTEM_DataSets()
        for dataset in datasets:
            print(f"Load [{directory + dataset}]")
            np_load_data = np.load(directory + dataset)
            train_data = None
            for i in range(len(np_load_data)):

                np_data = np.expand_dims(np_load_data[f'arr_{i}'], axis=0)[0]
                print(np_data.shape)

                if train_data is None:
                    train_data = np_data
                else:
                    train_data = np.concatenate((train_data, np_data), axis=0)
            print(f"最初の５音:{train_data[-1][0:25]}")
            t_data.add_data(train_data)
        print(f"Token size: {sum(len(sub) for sub in t_data.musics_seq)} ")
        print("----------------------------------")
        #t_data.split_seq_data()
        #t_data.set_padding()
        t_data.set_train_data()

        return t_data
    else:
        np_load_data = np.load(directory + datasets[0])
        print(np_load_data[f'arr_{3}'])
    return None


def get_padding_mask(input_ids):
    pad_id = None
    for inputs in input_ids:
        pad = []
        for token in inputs:
            if token == 0:
                pad.append(0)
            else:
                pad.append(1)
        if pad_id is None:
            pad_id = [pad]
        else:
            pad_id = pad_id + [pad]
    padding_mask = torch.tensor(pad_id, dtype=torch.float).to(device)
    return padding_mask


def train(ayato_dataset, message: Messenger, vocab_size: int, num_epochs: int, weight: Tensor,  trans_layer=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
          position_length=2048):
    loader = DataLoader(ayato_dataset, batch_size=16, shuffle=True, pin_memory=False)
    print("Creating Model....")
    model = MORTEM(vocab_size=vocab_size, trans_layer=trans_layer, num_heads=num_heads,
                   d_model=d_model, dim_feedforward=dim_feedforward,
                   dropout=dropout, position_length=position_length).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weight.to(device))  # 損失関数を定義
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)  # オプティマイザを定義

    print("Start training...")

    loss_val = None
    mail_bool = True
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1} start....")
        print(f"batch size :{len(loader)}")
        count = 0
        epoch_loss = 0.0
        model.train()
        for input_ids, targets in loader: # seqにはbatch_size分の楽曲が入っている
            print(f"learning sequence {count + 1}")
            begin_time = time.time()
            input_ids.to(device)

            optimizer.zero_grad()
            #inputs_mask = model.mortem.generate_square_subsequent_mask(input_ids.shape[1]).to(device)
            targets_mask = model.transformer.generate_square_subsequent_mask(targets.shape[1]).to(device)
            padding_mask_in: Tensor = get_padding_mask(input_ids)
            padding_mask_tgt: Tensor = get_padding_mask(targets)

            output = model(input_ids, targets, None, targets_mask, padding_mask_in, padding_mask_tgt)

            outputs = output.view(-1, output.size(-1))
            targets = targets.view(-1).long()

            loss = criterion(outputs, targets)  # 損失を計算
            loss.backward()  # 逆伝播
            optimizer.step()  # オプティマイザを更新
            epoch_loss = loss.item()
            count += 1
            end_time = time.time()

            if mail_bool:
                send_prediction_end_time(message, len(loader), begin_time, end_time, vocab_size, num_epochs, trans_layer, num_heads, d_model, dim_feedforward, dropout, position_length)
                mail_bool = False
            if (count + 1) % 100 == 0:
                message.send_mail("機械学習の途中経過について", f"Epoch {epoch + 1}/{num_epochs}の"
                                                                f"learning sequence {count + 1}結果は、\n {epoch_loss:.4f}でした。")
            print(epoch_loss)


        print(f"Epoch [{epoch + 1}/{num_epochs}],  Loss: {epoch_loss:.4f}")
        message.send_mail("機械学習の途中経過について", f"Epoch {epoch + 1}/{num_epochs}の結果は、{epoch_loss:.4f}でした。")
        loss_val = epoch_loss
    return model, loss_val


device = get_device()


class MORTEM(nn.Module):
    token_dict = {
        0: "600_633",
        1: "10_139",
        2: "300_429",
        3: "500_600",
        4: "600_605"
    }

    def __init__(self, vocab_size, trans_layer=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
                 position_length=2048):
        super(MORTEM, self).__init__()

        self.trans_layer = trans_layer
        self.num_heads = num_heads
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        #位置エンコーディングを作成
        self.positional: PositionalEncoding = PositionalEncoding(self.d_model, dropout, position_length).to(device)
        #Transformerの設定
        self.transformer: nn.Transformer = nn.Transformer(d_model=self.d_model, nhead=num_heads,  #各種パラメーターの設計
                                                          num_encoder_layers=self.trans_layer,
                                                          num_decoder_layers=self.trans_layer,
                                                          dropout=self.dropout, dim_feedforward=dim_feedforward,
                                                          ).to(device)
        print(f"Input Vocab Size:{vocab_size}")
        self.Wout = nn.Linear(self.d_model, vocab_size).to(device)

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, self.d_model).to(device)
        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(device)

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

        score = self.Wout(out)
        return score

    def generate_by_length(self, input_seq, max_length, p=0.9, temperature=0.1):
        self.eval()
        output = torch.tensor([input_seq], dtype=torch.long).unsqueeze(1).to(device)
        for _ in range(max_length):
            with torch.no_grad():
                output = self._next_note_token(output)

        return output

    def _next_note_token(self, output, p=0.9, temperature=0.1):
        isEnd = False
        token_count = 0
        while not isEnd:
            mask = self.transformer.generate_square_subsequent_mask(output.shape[1]).to(device)
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

            output = torch.cat((output.flatten(), token.unsqueeze(0))).unsqueeze(0).to(device)

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
        output = torch.tensor([input_ids], dtype=torch.long).unsqueeze(1).to(device)
        for _ in range(max_length):
            with torch.no_grad():
                mask = self.transformer.generate_square_subsequent_mask(output.shape[1]).to(device)
                outputs = self(output, output, mask, mask, None, None)
                logits = outputs[:, -1, :]

                print(logits[-1, 10:138])

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
                output = torch.cat((output.flatten(), next_token)).unsqueeze(0).to(device)
                #print(output)
        return output.tolist()



class MORTEM_DataSets(Dataset):
    def __init__(self):
        self.musics_seq = None
        self.tgt_seq = None

    def __len__(self):
        return len(self.musics_seq)

    def __getitem__(self, item):
        return torch.tensor(self.musics_seq[item], dtype=torch.int).to(device), torch.tensor(self.tgt_seq[item], dtype=torch.int).to(device)

    def add_data(self, music_seq: np.ndarray):
        if self.musics_seq is None:
            if len(music_seq.shape) == 1:
                self.musics_seq = [music_seq.tolist()]
            else:
                self.musics_seq = music_seq.tolist()
        else:
            if len(music_seq.shape) == 1:
                music_seq: list = [music_seq.tolist()]
                self.musics_seq = self.musics_seq + music_seq
            else:
                self.musics_seq = self.musics_seq + music_seq.tolist()

        print(self._get_shape(self.musics_seq))
        pass

    def split_seq_data(self):
        new_music_seq = [[]]
        for i in range(len(self.musics_seq)):
            result = []
            current_sublist = []
            for value in self.musics_seq[i]:
                current_sublist.append(value)
                if value == 2:
                    result.append(current_sublist)
                    current_sublist = []
            new_music_seq = new_music_seq + result

        self.musics_seq = new_music_seq

        pass

    def set_train_data(self):
        print(f"Set train data....{self._get_shape(self.musics_seq)}")
        for music_seq in self.musics_seq:
            tgt_data = []
            for i in range(len(music_seq) - 1):
                tgt_data.append(music_seq[i + 1])
            if self.tgt_seq is None:
                self.tgt_seq = [tgt_data]
            else:
                self.tgt_seq = self.tgt_seq + [tgt_data]


        print(f"clear! train shape is:{self._get_shape(self.tgt_seq)}")

    def _get_shape(self, lst):
        if isinstance(lst, list):
            return [len(lst)] + self._get_shape(lst[0]) if lst else []
        return []

    def set_padding(self):
#        self._padding(self.tgt_seq)
        self._padding(self.musics_seq)

        pass

    def get_max_length(self, target):
        max_lengths = []
        for t in target:
            max_lengths.append(len(t))
        max_length = max(max_lengths)
        return max_length

    def _padding(self, target: list):
        max_lengths = []
        for t in target:
            max_lengths.append(len(t))
        max_length = max(max_lengths)
        print(f"Max length is {max_length}")
        for t in target:
            if len(t) < max_length:
                for _ in range(max_length - len(t)):
                    t.append(0)
        pass


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, **kwargs):
        return memory