"""E3 探索的ロバストネス: PMA(Attention pool)分類ヘッドの学習。

head アーム = 事前学習表現の **frozen-feature linear probe**。
「凍結した潜在表現に、Pool+headで読み出せるだけの(線形)分離性があるか」を測る。
★backbone は完全凍結。Pool(PMA)+分類器のみ学習(full-FTは破滅的忘却で前提を壊すため禁止)。
出力は `{arm}_{task}_pma/` に分離。

train_sft.py と同型: MORTMTrainSet を継承した MORTMHeadTrainSet を定義し、
  1. モデルを ClassificationMORTM(PMA head) に差し替え(backbone は strict=False でロード)
  2. backbone を requires_grad=False で凍結し、pma/classifier のみ学習
  3. データを (music_block → 単一クラスラベル) に差し替え(pre_processing/epoch_fc)
  4. 損失を CrossEntropyLoss(ignore_index無し, クラス0も有効ラベル) に変更
学習は train_custom で起動 (run_train.py と同型)。
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from mortm.train.train import (MORTMTrainSet, train_custom, collate_fn_with_tgt,
                               _get_padding_mask, _extract_paths_and_dataset_ids_from_pack)
from mortm.train.config import AbstractTrainSet, TrainArgs
from mortm.models.mortm import ClassificationMORTM
from mortm.models.modules.config import MORTMArgs
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.utils.messager import _DefaultMessenger
from mortm.models.modules.progress import _DefaultLearningProgress


def build_class_sets(tok):
    keys = sorted(i for n, i in tok.tokens.items() if str(n).startswith("k_"))
    dens = sorted(i for n, i in tok.tokens.items()
                  if ("DENSE" in str(n).upper() or "DENSITY" in str(n).upper()
                      or str(n).upper().startswith("<NOTE_DENSE")))
    genres = sorted(i for n, i in tok.tokens.items() if str(n).startswith("<GENRE_"))
    return np.array(keys), np.array(dens), np.array(genres)


def extract_samples(seq_list, task, trigs, maps, te_id):
    """1系列から (music_block, class_idx) を抽出。
    ★music_block = 先頭<EOS>を含む音楽ブロック + 末尾<TE> を付与:
      <EOS> <CONST_M> <INST_xxx> ... <TAG_END> <TE>
    (基盤/SFTが見る自己完結系列と同じ形。<META>/<SYSTEM>/答えは入力しない)
    trigs=(key_trig,dense_trig,genre_trig,meta_trig), maps=(key_map,dense_map,genre_map)。
    """
    key_trig, dense_trig, genre_trig, meta_trig = trigs
    key_map, dense_map, genre_map = maps
    out = []
    if task == "key":
        trig, cmap = key_trig, key_map
    elif task == "dense":
        trig, cmap = dense_trig, dense_map
    else:
        trig, cmap = genre_trig, genre_map

    if trig in seq_list:
        idx = seq_list.index(trig)
        mb = seq_list[:idx] + [te_id]          # 先頭<EOS>込み + 末尾<TE>
        if task == "dense":
            # <DENCE> <INST_*> <dense> : +2 で密度トークン
            if idx + 2 < len(seq_list) and seq_list[idx + 2] in cmap:
                out.append((mb, cmap[seq_list[idx + 2]]))
        else:
            if idx + 1 < len(seq_list) and seq_list[idx + 1] in cmap:
                out.append((mb, cmap[seq_list[idx + 1]]))
    elif meta_trig in seq_list:
        idx = seq_list.index(meta_trig)
        mb = seq_list[:idx] + [te_id]          # <EOS>...<TAG_END> + <TE>
        for tid in seq_list[idx + 1:]:
            if tid in cmap:
                out.append((mb, cmap[tid]))
                break
    return out


class _HeadInnerDataset(Dataset):
    """外側 npz バッチ(pack)から (music_block, label) を展開した内側データセット。"""
    def __init__(self, paths, task, trigs, maps, te_id, min_len, max_len):
        self.samples = []
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                with np.load(p, allow_pickle=True) as data:
                    i = 1
                    while f"array{i}" in data.files:
                        seq = data[f"array{i}"]
                        i += 1
                        if seq.ndim == 0:
                            continue
                        seq_list = seq.astype(np.int64).tolist()
                        for mb, lab in extract_samples(seq_list, task, trigs, maps, te_id):
                            if min_len < len(mb) < max_len:
                                self.samples.append((mb, lab))
            except Exception:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mb, lab = self.samples[idx]
        return torch.tensor(mb, dtype=torch.long), torch.tensor(lab, dtype=torch.long)


class MORTMHeadTrainSet(MORTMTrainSet):
    """MORTMTrainSet を PMA分類ヘッド用に拡張。ClassificationMORTM(PMA) + CrossEntropyLoss。
    backbone完全凍結・Pool(pma)+classifierのみ学習(frozen-feature linear probe)。"""

    def __init__(self, args: MORTMArgs, t_args: TrainArgs, tokenizer: Tokenizer, task: str,
                 calc_val_loss_tokens, progress, log_scale=False, project_name="",
                 config=None, load_directory=None, model_name=None):
        self.tokenizer = tokenizer
        self.task = task
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(f"cuda:{self.local_rank}")

        key_tokens, dense_tokens, genre_tokens = build_class_sets(tokenizer)
        self.trigs = (tokenizer.get("<KEY>"), tokenizer.get("<DENCE>"),
                      tokenizer.get("<GENRE>"), tokenizer.get("<META>"))
        self.te_id = tokenizer.get("<TE>")
        self.maps = ({int(t): i for i, t in enumerate(key_tokens)},
                     {int(t): i for i, t in enumerate(dense_tokens)},
                     {int(t): i for i, t in enumerate(genre_tokens)})
        num_classes = {"key": len(key_tokens), "dense": len(dense_tokens),
                       "genre": len(genre_tokens)}[task]

        # --- PMA head 付きモデルを構築。backbone を strict=False でロード(pma/classifier は新規) ---
        self.model = ClassificationMORTM(args, num_classes, progress).to(device)
        if load_directory is not None:
            sd = torch.load(load_directory, map_location=device)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            non_head_missing = [k for k in missing if not (k.startswith("pma") or k.startswith("classifier"))]
            if self.local_rank == 0:
                print(f"[HEAD-PMA] base loaded: missing={len(missing)}(head以外={len(non_head_missing)}), "
                      f"unexpected={len(unexpected)} num_classes={num_classes}")
                if non_head_missing:
                    print(f"[HEAD-PMA][警告] head以外のmissing: {non_head_missing[:5]}")

        # --- ★linearプローブ: 事前学習表現を完全凍結し、Pool(pma)+分類器のみ学習する。 ---
        # full-FT は backbone を書き換え(破滅的忘却)、「凍結表現に線形分離性があるか」という
        # head アームの前提を壊す。ここでは backbone を requires_grad=False で凍結し、
        # pma / classifier だけを学習可能にする(frozen-feature probe)。
        n_train = 0
        for name, p in self.model.named_parameters():
            trainable = name.startswith("pma") or name.startswith("classifier")
            p.requires_grad = trainable
            if trainable:
                n_train += p.numel()

        # 学習対象は forward で全て使われる(pma/classifier) → find_unused_parameters=False。
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=False)
        total_param, self.active_params = self.model.module.get_param()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        adam = torch.optim.AdamW(trainable_params, lr=t_args.lr_param, weight_decay=0.01)

        if self.local_rank == 0:
            print(f"[HEAD-PMA] task={task} frozen-backbone probe: 学習可能={n_train/1e6:.3f}M "
                  f"(Pool+head) / 全体={total_param/1e6:.1f}M lr={t_args.lr_param}")
            if log_scale and config is not None:
                import wandb
                with open(config) as f:
                    data = json.load(f)
                data.update({"model_params": self.active_params, "total_params": total_param,
                             "head_task": task, "head_type": "PMA"})
                wandb.init(project=project_name, name=model_name, config=data, reinit=True)

        # --- 損失: CrossEntropyLoss(ignore_index無し。クラス0も有効ラベル) ---
        AbstractTrainSet.__init__(
            self,
            criterion=nn.CrossEntropyLoss().to(device),
            optimizer=adam, t_args=t_args, m_args=args,
            calc_val_loss_tokens=calc_val_loss_tokens,
        )
        self.all_tokens = torch.tensor(0, device=device, dtype=torch.long)
        self.optimizer_steps = 0

    def pre_processing(self, pack, progress):
        paths, _ = _extract_paths_and_dataset_ids_from_pack(pack)
        return _HeadInnerDataset(paths, self.task, self.trigs, self.maps, self.te_id,
                                 self.args.min_length, self.args.position_length)

    def epoch_fc(self, model, pack, progress):
        """pack=(src[B,S], tgt[B])。ClassificationMORTM で logits[B,C] を出し CrossEntropy。"""
        device = torch.device(f"cuda:{self.local_rank}")
        src, tgt = pack
        src = src.to(device)
        tgt = tgt.to(device).long()
        padding_mask = _get_padding_mask(src, progress, device)
        if model.training:
            self.all_tokens += torch.sum(padding_mask == 1)
        logits: Tensor = model(x=src, padding_mask=padding_mask, is_causal=True)  # [B, C]
        return logits.to(device=device, dtype=torch.float32), tgt


def run_head(task, model_config, train_config, base_checkpoint, root_directory, save_directory,
             version, eval_list_json=None, project_name="MORTM_E3_HEAD_PMA", log_scale=True, seed=None):
    """run_train.py と同型。PMAヘッドトレーナーを構築して train_custom で起動する。"""
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if seed is not None:
        from mortm.utils.repro import set_seed
        set_seed(int(seed), deterministic=False)

    tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))
    progress = _DefaultLearningProgress()
    args = MORTMArgs(json_directory=model_config)
    t_args = TrainArgs(json_directory=train_config)

    trainer = MORTMHeadTrainSet(
        args, t_args, tokenizer, task, t_args.val_total_tokens, progress,
        log_scale=log_scale, project_name=project_name,
        config=model_config, load_directory=base_checkpoint, model_name=version,
    )
    train_custom(
        trainer, t_args, root_directory, save_directory, version,
        message=_DefaultMessenger(), eval_list_json=eval_list_json, progress=progress,
        coll_fn=collate_fn_with_tgt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True,
                        choices=["A1", "A2", "A1-40B-80M", "A1-40B-160M"])
    parser.add_argument("--task", type=str, required=True, choices=["key", "dense", "genre"])
    parser.add_argument("--backbone_cfg", type=str, default="configs/models/mortm/foundation/80M.json")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--train_config", type=str, default="configs/train/mortm/head/head_pma.json")
    parser.add_argument("--train_json", type=str, default="/home/takaaki-nagoshi/data/sft/analysis/train_50M.json")
    parser.add_argument("--eval_json", type=str, default="/home/takaaki-nagoshi/data/sft/analysis/eval.json")
    parser.add_argument("--save_dir", type=str, default="out/models/paper/E3")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    suffix = f"_s{args.seed}" if args.seed is not None else ""
    save_dir = os.path.join(args.save_dir, f"{args.arm}_{args.task}_pma{suffix}")
    version = f"E3-{args.arm}-head-pma-{args.task}{suffix}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting PMA-head training arm={args.arm} task={args.task} seed={args.seed}...")
    run_head(
        task=args.task,
        model_config=args.backbone_cfg,
        train_config=args.train_config,
        base_checkpoint=args.backbone_ckpt,
        root_directory=(args.train_json,),
        save_directory=save_dir,
        version=version,
        eval_list_json=(args.eval_json,),
        seed=args.seed,
    )
