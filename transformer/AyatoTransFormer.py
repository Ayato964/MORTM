import random
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


# デバイスを取得
def set_train_data(directory, datasets):
    if not IS_DEBUG:
        print("Generating TrainData.....")
        t_data = AyatoDataSet()
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

            t_data.add_data(train_data)
        print(f"Token size: {sum(len(sub) for sub in t_data.musics_seq)} ")
        print("----------------------------------")
        print("Generate Padding...")
        t_data.split_seq_data()
        t_data.set_padding()
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


def train(ayato_dataset, message: Messenger, vocab_size: int, num_epochs: int, trans_layer=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
          position_length=2048):
    loader = DataLoader(ayato_dataset, batch_size=5, shuffle=False, pin_memory=False)
    print("Creating Model....")
    model = AyatoModel(vocab_size=vocab_size, trans_layer=trans_layer, num_heads=num_heads,
                       d_model=d_model, dim_feedforward=dim_feedforward,
                       dropout=dropout, position_length=position_length).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 損失関数を定義
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # オプティマイザを定義

    print("Start training...")
    message.send_mail("学習開始", f"研究室のPCから学習の開始をお知らせいたします。")
    message.send_mail("見込み終了時間", f"本コードの終了見込み時間は一週間です。\n 完成までしばらくお待ちください。")
    loss_val = None
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1} start....")
        print(f"batch size :{len(loader)}")
        count = 0
        epoch_loss = 0.0

        model.train()
        for input_ids, targets in loader: # seqにはbatch_size分の楽曲が入っている
            print(f"learning sequence {count + 1}")
            input_ids.to(device)
            input_ids.to(device)

            optimizer.zero_grad()
            #inputs_mask = model.transformer.generate_square_subsequent_mask(input_ids.shape[1]).to(device)
            targets_mask = model.transformer.generate_square_subsequent_mask(targets.shape[1]).to(device)
            padding_mask_in: Tensor = get_padding_mask(input_ids)
            padding_mask_tgt: Tensor = get_padding_mask(targets)

            output = model(input_ids, targets, None, targets_mask, padding_mask_in, padding_mask_tgt)

            outputs = output.view(-1, output.size(-1))
            targets = targets.view(-1).long()

            loss = criterion(outputs, targets)  # 損失を計算
            loss.backward()  # 逆伝播
            optimizer.step()  # オプティマイザを更新
            epoch_loss = loss
            count += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}],  Loss: {epoch_loss / count:.4f}")
        message.send_mail("機械学習の途中経過について", f"Epoch {epoch + 1}/{num_epochs}の結果は、{epoch_loss / count:.4f}でした。")
        loss_val = epoch_loss / count
    return model, loss_val


device = get_device()


class AyatoModel(nn.Module):
    def __init__(self, vocab_size, trans_layer=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
                 position_length=2048):
        super(AyatoModel, self).__init__()

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

    def generate(self, input_seq, max_length=50):
        self.eval()
        generated_seq = torch.tensor([input_seq], dtype=torch.long).unsqueeze(1).to(device)
        #print(f"first{generated_seq.shape}")

        with torch.set_grad_enabled(False):
            for i in range(max_length):
                rand = random.randint(0, 100)
                mask = self.transformer.generate_square_subsequent_mask(generated_seq.shape[1]).to(device)

                score: torch.Tensor = self.softmax(self(generated_seq, generated_seq, mask, mask, None, None))
                print(f"Score is {score.shape}")
                next_token_score: torch.Tensor = score[-1, :, :]
                next_token_score.flatten()
                if rand <= 99:
                    dis = torch.distributions.categorical.Categorical(probs=next_token_score)
                    output_node = dis.sample()
                else:
                    output_node = torch.argmax(next_token_score, dim=-1)
                    print(output_node.shape)
                generated_seq = torch.cat((generated_seq.flatten(), output_node)).unsqueeze(0).to(device)

        #print(f"Finally:{generated_seq}")
        return generated_seq.tolist()  # 生成されたシーケンスを返す

    def top_p_sampling(self, input_ids, tokenizer, p=0.9, max_length=20, temperature=0.5):
        output = torch.tensor([input_ids], dtype=torch.long).unsqueeze(1).to(device)
        for _ in range(max_length):
            with torch.no_grad():
                mask = self.transformer.generate_square_subsequent_mask(output.shape[1]).to(device)
                outputs = self(output, output, mask, mask, None, None)
                logits = outputs[:, -1, :]

                # 温度サンプリング
                logits = logits / temperature

                # トップpサンプリング
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
                print(output)
        return output.tolist()


class AyatoDataSet(Dataset):
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