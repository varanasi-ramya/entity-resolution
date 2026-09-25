"""Union symbolic + ANN candidates."""
import os
import pandas as pd


def union_candidates(out_dir, split='train'):
    out_pq = f'{out_dir}/candidates_{split}.parquet'
    if os.path.exists(out_pq):
        n = pd.read_parquet(out_pq).shape[0]
        print(f"  candidates_{split}.parquet (skip, {n:,} pairs)")
        return
    parts = []
    for tag, tbl in [
        (f'sym_{split}_s1_s2', 's2'),
        (f'sym_{split}_s1_s3', 's3'),
        (f'ann_{split}_s1_s2', 's2'),
        (f'ann_{split}_s1_s3', 's3'),
    ]:
        p = f'{out_dir}/{tag}.parquet'
        if not os.path.exists(p):
            print(f"  [warn] missing {p}")
            continue
        d = pd.read_parquet(p)
        d['cand_source'] = tbl
        parts.append(d[['s1_id','cand_id','cand_source']])
    u = pd.concat(parts, ignore_index=True).drop_duplicates(['s1_id','cand_id'])
    tmp = out_pq + '.tmp'
    u.to_parquet(tmp, index=False, compression='zstd')
    os.replace(tmp, out_pq)
    print(f"  candidates_{split}.parquet              {len(u):>10,} pairs")
