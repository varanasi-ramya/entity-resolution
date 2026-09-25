"""Symbolic candidate blocking via DuckDB (same-state prefix + soundex)."""
import os, time
import duckdb
import jellyfish


def _register_udf(con):
    con.create_function(
        'soundex_py',
        lambda x: jellyfish.soundex(x) if x else None,
        ['VARCHAR'], 'VARCHAR',
    )


def _candidates_one_pair(con, s1_clean, other_clean, out_parquet, other_tag):
    if os.path.exists(out_parquet):
        n = con.execute(f"SELECT COUNT(*) FROM '{out_parquet}'").fetchone()[0]
        print(f"  {os.path.basename(out_parquet):40s} (skip, {n:,} rows)")
        return
    t0 = time.time()
    tmp = out_parquet + '.tmp'
    con.execute(f"""
        COPY (
            WITH a AS (
                SELECT entity_id, name_norm, country, state
                FROM '{s1_clean}'
                WHERE name_norm IS NOT NULL AND country IS NOT NULL
            ),
            b AS (
                SELECT entity_id, name_norm, country, state
                FROM '{other_clean}'
                WHERE name_norm IS NOT NULL AND country IS NOT NULL
            ),
            p1 AS (
                SELECT a.entity_id AS s1_id, b.entity_id AS cand_id
                FROM a JOIN b
                  ON a.country = b.country
                 AND a.state = b.state
                 AND SUBSTR(a.name_norm, 1, 4) = SUBSTR(b.name_norm, 1, 4)
            ),
            p2 AS (
                SELECT a.entity_id AS s1_id, b.entity_id AS cand_id
                FROM a JOIN b
                  ON a.country = b.country
                 AND soundex_py(a.name_norm) = soundex_py(b.name_norm)
            ),
            u AS (
                SELECT s1_id, cand_id FROM p1
                UNION
                SELECT s1_id, cand_id FROM p2
            )
            SELECT s1_id, cand_id, '{other_tag}' AS cand_source FROM u
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    os.replace(tmp, out_parquet)
    n = con.execute(f"SELECT COUNT(*) FROM '{out_parquet}'").fetchone()[0]
    print(f"  {os.path.basename(out_parquet):40s} {n:>10,} pairs  "
          f"{time.time()-t0:6.1f}s")


def build_symbolic_candidates(out_dir, split='train'):
    con = duckdb.connect()
    _register_udf(con)
    s1 = f'{out_dir}/{split}_source1_clean.parquet'
    s2 = f'{out_dir}/{split}_source2_clean.parquet'
    s3 = f'{out_dir}/{split}_source3_clean.parquet'
    _candidates_one_pair(con, s1, s2, f'{out_dir}/sym_{split}_s1_s2.parquet', 's2')
    _candidates_one_pair(con, s1, s3, f'{out_dir}/sym_{split}_s1_s3.parquet', 's3')
