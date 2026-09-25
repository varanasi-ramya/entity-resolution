
"""Apply trained model, produce submission.tsv."""
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb

from .features import FEATURES


def predict(out_dir):
    model_path = f'{out_dir}/lgbm_pairwise.txt'
    thr_path   = f'{out_dir}/threshold.json'
    feat_path  = f'{out_dir}/features_test.parquet'
    sub_path   = f'{out_dir}/submission.tsv'

    model = lgb.Booster(model_file=model_path)
    thr = json.load(open(thr_path))['threshold']

    feat = pd.read_parquet(feat_path)
    feat['proba'] = model.predict(feat[FEATURES])
    mask = feat['proba'].values >= thr
    print(f"  test pairs={len(feat):,} kept={mask.sum():,} "
          f"({100*mask.mean():.1f}%)  thr={thr:.2f}")

    g = (feat[mask]
         .astype({'s1_id': str, 'cand_id': str})
         .groupby('s1_id')['cand_id']
         .apply(lambda s: ','.join(sorted(set(s))))
         .rename('matched_entity_ids')
         .reset_index()
         .rename(columns={'s1_id': 'source1_entity_id'}))

    all_s1 = pd.read_parquet(f'{out_dir}/test_source1_clean.parquet',
                             columns=['entity_id'])
    all_s1 = all_s1.rename(columns={'entity_id': 'source1_entity_id'})
    all_s1['source1_entity_id'] = all_s1['source1_entity_id'].astype(str)

    sub = all_s1.merge(g, on='source1_entity_id', how='left')
    sub['matched_entity_ids'] = sub['matched_entity_ids'].fillna('')
    sub.to_csv(sub_path, sep='\t', index=False)
    n_pred = (sub['matched_entity_ids'] != '').sum()
    print(f"  wrote {sub_path}  rows={len(sub):,} "
          f"with_matches={n_pred:,} ({100*n_pred/len(sub):.1f}%)")
