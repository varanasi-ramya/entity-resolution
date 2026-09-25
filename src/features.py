"""Pairwise feature engineering on candidate pairs."""
import os
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import jellyfish

# city_match removed — there was no city column being extracted; the old
# line compared state to state a second time (copy-paste bug), adding a
# redundant, noisy column with no real signal.
FEATURES = [
    'lev_ratio', 'token_sort', 'jaro_winkler',
    'addr_token_sort',
    'state_match', 'street_num_match',
    'cand_addr_missing', 'name_len_diff', 'name_token_overlap',
]


def _token_overlap(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_features(out_dir, split='train'):
    out_pq = f'{out_dir}/features_{split}.parquet'
    if os.path.exists(out_pq):
        n = pd.read_parquet(out_pq).shape[0]
        print(f"  features_{split}.parquet (skip, {n:,} rows)")
        return

    s1 = pd.read_parquet(f'{out_dir}/{split}_source1_clean.parquet')
    s2 = pd.read_parquet(f'{out_dir}/{split}_source2_clean.parquet')
    s3 = pd.read_parquet(f'{out_dir}/{split}_source3_clean.parquet')
    s2['src_tag'] = 's2'; s3['src_tag'] = 's3'
    others = pd.concat([s2, s3], ignore_index=True)

    keep = ['entity_id','name_norm','addr_norm','state','street_num','src_tag']
    s1 = s1[[c for c in keep if c in s1.columns]]
    others = others[[c for c in keep if c in others.columns]]

    cand = pd.read_parquet(f'{out_dir}/candidates_{split}.parquet')
    m = cand.merge(
        s1.rename(columns={c: f'a_{c}' for c in s1.columns if c != 'entity_id'}),
        left_on='s1_id', right_on='entity_id', how='left',
    ).drop(columns=['entity_id']).merge(
        others.rename(columns={c: f'b_{c}' for c in others.columns
                               if c != 'entity_id'}),
        left_on='cand_id', right_on='entity_id', how='left',
    ).drop(columns=['entity_id'])

    a_name = m['a_name_norm'].fillna('').astype(str).values
    b_name = m['b_name_norm'].fillna('').astype(str).values
    a_addr = m['a_addr_norm'].fillna('').astype(str).values
    b_addr = m['b_addr_norm'].fillna('').astype(str).values

    feats = pd.DataFrame({
        'lev_ratio':       [fuzz.ratio(a,b)/100.0     for a,b in zip(a_name,b_name)],
        'token_sort':      [fuzz.token_sort_ratio(a,b)/100.0 for a,b in zip(a_name,b_name)],
        'jaro_winkler':    [jellyfish.jaro_winkler_similarity(a,b)
                            for a,b in zip(a_name,b_name)],
        'addr_token_sort': [fuzz.token_sort_ratio(a,b)/100.0
                            for a,b in zip(a_addr,b_addr)],
        'state_match':     (m['a_state'].fillna('_') == m['b_state'].fillna('~')).astype('int8'),
        'street_num_match':(m['a_street_num'].fillna('_') == m['b_street_num'].fillna('~')).astype('int8'),
        'cand_addr_missing': m['b_addr_norm'].isna().astype('int8'),
        'name_len_diff':   (m['a_name_norm'].str.len().fillna(0)
                            - m['b_name_norm'].str.len().fillna(0)).abs().astype('int16'),
        'name_token_overlap': [_token_overlap(a,b) for a,b in zip(a_name,b_name)],
    })

    out = pd.concat([
        m[['s1_id','cand_id','cand_source']].reset_index(drop=True),
        feats.reset_index(drop=True),
    ], axis=1)
    tmp = out_pq + '.tmp'
    out.to_parquet(tmp, index=False, compression='zstd')
    os.replace(tmp, out_pq)
    print(f"  features_{split}.parquet                {len(out):>10,} rows")
