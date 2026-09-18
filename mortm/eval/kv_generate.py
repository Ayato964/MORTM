"""トリガー無し(P形式)KVキャッシュ生成 — zero-SFT基盤評価用(設計書§6.1 P形式)。
モデルの is_save_cache API をそのまま使う(モデル無改変)。<MGEN>/<CGEN> を仮定しない。
model.top_sampling_measure_kv_cache は出力を<MGEN>/<CGEN>で分割するためP形式で落ちる。
本関数は decode ループ(=同じ KV キャッシュ機構)を回し、生の生成トークン(prompt後)を返すだけ。

使い方: gen = kv_generate(model, prompt_ids, tok, dev, max_measures=4, p=0.9, temperature=1.0)
"""
import torch


def _reset_kv_cache(model):
    """各生成の前に全アテンションの KV キャッシュを初期化(前回窓の残留を防ぐ)。"""
    for m in model.modules():
        if hasattr(m, "kv_cache"):
            m.kv_cache = None
        if hasattr(m, "cache_seqlens"):
            m.cache_seqlens = None


@torch.no_grad()
def kv_generate(model, prompt, tok, dev, max_measures, p=0.9, temperature=1.0, max_new=1200, seed=None):
    """prompt(list/1D)からCONST等を top-p 生成。<TE> か max_measures 小節 か max_new で停止。
    prompt 後に生成したトークン列(list[int])を返す。モデルは無改変・is_save_cache=True でKVキャッシュ利用。"""
    if seed is not None:
        torch.manual_seed(seed)
    TE = tok.get("<TE>"); TAG = tok.get("<TAG_END>"); SME = tok.get("<SME>")
    src = torch.tensor(prompt, dtype=torch.long, device=dev).unsqueeze(0)  # [1, L]
    _reset_kv_cache(model)
    # --- prefill(padding_mask経路: 出力[1,L,V]) ---
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model.forward(src, padding_mask=(src != 0), is_causal=True, is_save_cache=True)
    nxt = model.top_p_sampling(logits[:, -1, :], p=p, temperature=temperature)
    nid = int(nxt.flatten()[0])
    gen = []; sme = 0
    for _ in range(max_new):
        if nid in (TE, TAG):
            break
        if nid == SME:
            sme += 1
            if sme > max_measures:
                break
        gen.append(nid)
        # --- decode(padding_mask=None経路: 入力は1D[1], 出力[1,V]) ---
        step_in = torch.tensor([nid], dtype=torch.long, device=dev)  # 1D
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(step_in, padding_mask=None, is_causal=True, is_save_cache=True)  # [1,V]
        nxt = model.top_p_sampling(logits[-1:], p=p, temperature=temperature)
        nid = int(nxt.flatten()[0])
    return gen


@torch.no_grad()
def kv_generate_batch(model, prompts, tok, dev, max_measures_list, p=0.9, temperature=1.0, max_new=1200, seed=None):
    """prompts (List[List[int]]) から一括でKVキャッシュ生成を行う。バッチ対応版。"""
    if seed is not None:
        torch.manual_seed(seed)
    TE = tok.get("<TE>"); TAG = tok.get("<TAG_END>"); SME = tok.get("<SME>")
    
    batch_size = len(prompts)
    if batch_size == 0:
        return []
        
    # 左パディングでアライメントする
    max_len = max(len(pr) for pr in prompts)
    padded_prompts = []
    for pr in prompts:
        pad_len = max_len - len(pr)
        padded_prompts.append([0] * pad_len + pr)
        
    src = torch.tensor(padded_prompts, dtype=torch.long, device=dev)  # [B, max_len]
    
    _reset_kv_cache(model)
    
    # 1. Prefill
    prompt_padding_mask = (src != 0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model.forward(src, padding_mask=prompt_padding_mask, is_causal=True, is_save_cache=True)
        
    # 各バッチの最後のロジットを取り出す [B, V]
    last_logits = logits[:, -1, :]
    
    nxt = model.top_p_sampling(last_logits, p=p, temperature=temperature)  # [B]
    
    finished = [False] * batch_size
    generated = [[] for _ in range(batch_size)]
    sme_counts = [0] * batch_size
    
    nid = nxt.tolist()
    
    for step in range(max_new):
        all_finished = True
        step_toks = []
        for i in range(batch_size):
            if finished[i]:
                step_toks.append(0)
                continue
            all_finished = False
            curr_id = nid[i]
            
            if curr_id in (TE, TAG):
                finished[i] = True
                step_toks.append(0)
                continue
            if curr_id == SME:
                sme_counts[i] += 1
                if sme_counts[i] > max_measures_list[i]:
                    finished[i] = True
                    step_toks.append(0)
                    continue
            
            generated[i].append(curr_id)
            step_toks.append(curr_id)
            
        if all_finished:
            break
            
        # 2. Decode Step
        step_in = torch.tensor(step_toks, dtype=torch.long, device=dev)  # [B] (1D tensor)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(step_in, padding_mask=None, is_causal=True, is_save_cache=True)  # [B, V] or [B, 1, V]
            
        nxt = model.top_p_sampling(logits.squeeze(1) if logits.dim() == 3 else logits, p=p, temperature=temperature)  # [B]
        nid = nxt.tolist()
        
    return generated

