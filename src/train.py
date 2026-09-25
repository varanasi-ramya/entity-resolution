
"""Train LightGBM pairwise classifier; tune threshold for set-F1."""
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .features import FEATURES


def set_f1(pred_sets, true_sets):
    f1s = []
    for s1_id, true in true_sets.items():
        pred = pred_sets.get(s1_id, set())
        if not pred and not true:
            f1s.append(1.0); continue
        tp = len(pred & true)
        prec = tp / len(pred) if pred else 0.0
        rec  = tp / len(true) if true else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2*prec*rec/(prec+rec))
    return float(np.mean(f1s)) if f1s else 0.0


def _parse_gt_set(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return set()
    s = str(s).strip()
    return {x.strip() for x in s.split(',') if x.strip()} if s else set()


def train(out_dir, gt_parquet, seed=42):
    model_path = f'{out_dir}/lgbm_pairwise.txt'
    thr_path   = f'{out_dir}/threshold.json'
    if os.path.exists(model_path) and os.path.exists(thr_path):
        info = json.load(open(thr_path))
        print(f"  lgbm_pairwise.txt + threshold.json (skip) "
              f"val_f1={info['val_f1']:.4f} thr={info['threshold']:.2f}")
        return

    feat = pd.read_parquet(f'{out_dir}/features_train.parquet')
    gt   = pd.read_parquet(gt_parquet)
    gt_sets = {str(r['source1_entity_id']): _parse_gt_set(r['matched_entity_ids'])
               for _, r in gt.iterrows()}

    feat['label'] = [
        1 if str(c) in gt_sets.get(str(s), set()) else 0
        for s, c in zip(feat['s1_id'], feat['cand_id'])
    ]
    pos = int(feat['label'].sum())
    print(f"  train rows={len(feat):,} pos={pos:,} ({100*pos/len(feat):.2f}%)")

    s1_ids = feat['s1_id'].astype(str).unique()
    tr_ids, va_ids = train_test_split(s1_ids, test_size=0.15, random_state=seed)
    tr = feat[feat['s1_id'].astype(str).isin(tr_ids)].copy()
    va = feat[feat['s1_id'].astype(str).isin(va_ids)].copy()

    X_tr, y_tr = tr[FEATURES], tr['label']
    X_va, y_va = va[FEATURES], va['label']

    spw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    params = {
        'objective': 'binary', 'metric': 'auc',
        'learning_rate': 0.05, 'num_leaves': 63,
        'min_data_in_leaf': 50, 'feature_fraction': 0.9,
        'bagging_fraction': 0.9, 'bagging_freq': 5,
        'scale_pos_weight': float(spw), 'verbose': -1,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=1000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    auc = roc_auc_score(y_va, model.predict(X_va))
    print(f"  LGBM trained | val AUC={auc:.4f}")
    model.save_model(model_path)

    va = va.copy()
    va['proba'] = model.predict(X_va)
    val_s1 = va['s1_id'].astype(str).values
    val_c  = va['cand_id'].astype(str).values
    val_true = {sid: gt_sets.get(sid, set()) for sid in va_ids}

    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.30, 0.91, 0.02):
        pred_sets = {}
        m = va['proba'].values >= t
        for sid, cid in zip(val_s1[m], val_c[m]):
            pred_sets.setdefault(sid, set()).add(cid)
        f1 = set_f1(pred_sets, val_true)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    json.dump({'threshold': best_t, 'val_f1': best_f1, 'val_auc': float(auc)},
              open(thr_path, 'w'))
    print(f"  threshold={best_t:.2f} | val set-F1={best_f1:.4f}")
