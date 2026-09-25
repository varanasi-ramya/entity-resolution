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


def _configure(con, out_dir):
    tmp_dir = f'{out_dir}/duckdb_tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    con.execute("SET memory_limit='8GB'")
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("PRAGMA threads=4")


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
                SELECT entity_id, name_norm, country, state,
                       soundex_py(name_norm) AS sdx
                FROM '{s1_clean}'
                WHERE name_norm IS NOT NULL AND country IS NOT NULL
            ),
            b AS (
                SELECT entity_id, name_norm, country, state,
                       soundex_py(name_norm) AS sdx
                FROM '{other_clean}'
                WHERE name_norm IS NOT NULL AND country IS NOT NULL
            ),
            p1_raw AS (
                SELECT a.entity_id AS s1_id, b.entity_id AS cand_id,
                       row_number() OVER (
                           PARTITION BY a.entity_id ORDER BY b.entity_id
                       ) AS rn
                FROM a JOIN b
                  ON a.country = b.country
                 AND a.state = b.state
                 AND SUBSTR(a.name_norm, 1, 4) = SUBSTR(b.name_norm, 1, 4)
            ),
            p1 AS (
                SELECT s1_id, cand_id FROM p1_raw WHERE rn <= 50
            ),
            p2_raw AS (
                SELECT a.entity_id AS s1_id, b.entity_id AS cand_id,
                       row_number() OVER (
                           PARTITION BY a.entity_id ORDER BY b.entity_id
                       ) AS rn
                FROM a JOIN b
                  ON a.country = b.country
                 AND a.state = b.state
                 AND a.sdx = b.sdx
            ),
            p2 AS (
                SELECT s1_id, cand_id FROM p2_raw WHERE rn <= 50
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
    _configure(con, out_dir)
    s1 = f'{out_dir}/{split}_source1_clean.parquet'
    s2 = f'{out_dir}/{split}_source2_clean.parquet'
    s3 = f'{out_dir}/{split}_source3_clean.parquet'
    _candidates_one_pair(con, s1, s2, f'{out_dir}/sym_{split}_s1_s2.parquet', 's2')
    _candidates_one_pair(con, s1, s3, f'{out_dir}/sym_{split}_s1_s3.parquet', 's3')
