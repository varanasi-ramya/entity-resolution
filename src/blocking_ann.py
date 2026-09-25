"""ANN candidate blocking via sentence-transformers + per-country FAISS."""
import os, time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


def _get_model():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer(MODEL_NAME, device=dev)


def _encode(texts, cache_path, bs=1024):
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if len(emb) == len(texts):
            return emb
    model = _get_model()
    emb = model.encode(
        texts, batch_size=bs, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype('float32')
    tmp = cache_path + '.tmp.npy'
    np.save(tmp, emb)
    os.replace(tmp, cache_path)
    return emb


def _load_texts(clean_parquet):
    df = pd.read_parquet(clean_parquet,
                         columns=['entity_id','name_norm','addr_norm','country'])
    txt = (df['name_norm'].fillna('') + ' ' +
           df['addr_norm'].fillna('')).tolist()
    return df, txt


def _make_index(dim, use_gpu):
    import faiss
    idx = faiss.IndexFlatIP(dim)
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
        except Exception as e:
            print(f"  [warn] GPU FAISS unavailable ({e}), falling back to CPU")
    return idx


def _ann_one_pair(split, out_dir, source_name, tag, k=20):
    s1_clean = f'{out_dir}/{split}_source1_clean.parquet'
    sX_clean = f'{out_dir}/{split}_{source_name}_clean.parquet'
    out_pq   = f'{out_dir}/ann_{split}_s1_{tag}.parquet'
    if os.path.exists(out_pq):
        n = pd.read_parquet(out_pq).shape[0]
        print(f"  {os.path.basename(out_pq):40s} (skip, {n:,} rows)")
        return

    import torch
    use_gpu = torch.cuda.is_available()

    t0 = time.time()
    df1, txt1 = _load_texts(s1_clean)
    df2, txt2 = _load_texts(sX_clean)
    emb1 = _encode(txt1, f'{out_dir}/{split}_source1_emb.npy')
    emb2 = _encode(txt2, f'{out_dir}/{split}_{source_name}_emb.npy')

    s1_ids  = df1['entity_id'].values
    s2_ids  = df2['entity_id'].values
    s1_ctry = df1['country'].values
    s2_ctry = df2['country'].values

    pairs = []
    for c in np.unique(s2_ctry):
        m1 = np.where(s1_ctry == c)[0]
        m2 = np.where(s2_ctry == c)[0]
        if len(m1) == 0 or len(m2) == 0:
            continue
        sub2 = emb2[m2]
        idx = _make_index(sub2.shape[1], use_gpu)
        idx.add(sub2)
        D, I = idx.search(emb1[m1], k=min(k, len(m2)))
        for row, irow in enumerate(I):
            s1_id = s1_ids[m1[row]]
            for j in irow:
                if j < 0:
                    continue
                pairs.append((s1_id, s2_ids[m2[j]], tag))

    out = pd.DataFrame(pairs, columns=['s1_id','cand_id','cand_source'])
    tmp = out_pq + '.tmp'
    out.to_parquet(tmp, index=False, compression='zstd')
    os.replace(tmp, out_pq)
    print(f"  {os.path.basename(out_pq):40s} {len(out):>10,} pairs  "
          f"{time.time()-t0:6.1f}s  (gpu={use_gpu})")


def build_ann_candidates(out_dir, split='train', k=20):
    _ann_one_pair(split, out_dir, 'source2', 's2', k=k)
    _ann_one_pair(split, out_dir, 'source3', 's3', k=k)
