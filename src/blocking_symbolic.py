"""Symbolic candidate blocking via DuckDB (same-state prefix + soundex).

Processes one (country, state) chunk at a time to bound peak memory —
a whole-dataset join was materializing before any cap could be applied.
Oversized blocks (a name-prefix or soundex code shared by too many
records) are dropped before joining, not filtered after, since filtering
after still requires computing the full join first.
Each state chunk is itself checkpointed, so an interrupted run resumes
without redoing completed states.
"""
import os, time, shutil
import duckdb
import jellyfish

BLOCK_SIZE_CAP  = 2000   # drop a block if its candidate-side count exceeds this
CAP_PER_ENTITY  = 50     # max candidates kept per S1 entity, per key type


def _register_udf(con):
    con.create_function(
        'soundex_py',
        lambda x: jellyfish.soundex(x) if x else None,
        ['VARCHAR'], 'VARCHAR',
    )


def _configure(con, out_dir):
    tmp_dir = f'{out_dir}/duckdb_tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    con.execute("SET memory_limit='6GB'")       # lower cap, more headroom under 12.7GB
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("PRAGMA threads=2")              # fewer parallel workers = lower peak RAM


def _states_present(con, s1_clean, other_clean):
    q = f"""
        SELECT DISTINCT country, state FROM (
            SELECT country, state FROM '{s1_clean}' WHERE state IS NOT NULL
            UNION
            SELECT country, state FROM '{other_clean}' WHERE state IS NOT NULL
        )
        ORDER BY country, state
    """
    return con.execute(q).fetchall()


def _process_chunk(con, s1_clean, other_clean, country, state, chunk_path, other_tag):
    tmp = chunk_path + '.tmp'
    con.execute(f"""
        COPY (
            WITH a AS (
                SELECT entity_id, name_norm, soundex_py(name_norm) AS sdx
                FROM '{s1_clean}'
                WHERE name_norm IS NOT NULL AND country = ? AND state = ?
            ),
            b AS (
                SELECT entity_id, name_norm, soundex_py(name_norm) AS sdx
                FROM '{other_clean}'
                WHERE name_norm IS NOT NULL AND country = ? AND state = ?
            ),
            b_pfx_ok AS (
                SELECT SUBSTR(name_norm,1,4) AS pfx
                FROM b GROUP BY pfx HAVING COUNT(*) <= {BLOCK_SIZE_CAP}
            ),
            b_sdx_ok AS (
                SELECT sdx FROM b GROUP BY sdx HAVING COUNT(*) <= {BLOCK_SIZE_CAP}
            ),
            p1 AS (
                SELECT a.entity_id AS s1_id, b.entity_id AS cand_id
                FROM a JOIN b ON SUBSTR(a.name_norm,1,4) = SUBSTR(b.name_norm,1,4)
                WHERE SUBSTR(a.name_norm,1,4) IN (SELECT pfx FROM b_pfx_ok)
            ),
            p2 AS (
                SELECT a.entity_id AS s1_id, b.entity_id AS cand_id
                FROM a JOIN b ON a.sdx = b.sdx
                WHERE a.sdx IN (SELECT sdx FROM b_sdx_ok)
            ),
            u AS (
                SELECT s1_id, cand_id FROM p1
                UNION
                SELECT s1_id, cand_id FROM p2
            ),
            capped AS (
                SELECT s1_id, cand_id,
                       row_number() OVER (PARTITION BY s1_id ORDER BY cand_id) AS rn
                FROM u
            )
            SELECT s1_id, cand_id, '{other_tag}' AS cand_source
            FROM capped WHERE rn <= {CAP_PER_ENTITY}
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """, [country, state, country, state])
    os.replace(tmp, chunk_path)


def _candidates_one_pair(con, s1_clean, other_clean, out_parquet, other_tag):
    if os.path.exists(out_parquet):
        n = con.execute(f"SELECT COUNT(*) FROM '{out_parquet}'").fetchone()[0]
        print(f"  {os.path.basename(out_parquet):40s} (skip, {n:,} rows)")
        return

    t0 = time.time()
    chunk_dir = out_parquet + '.chunks'
    os.makedirs(chunk_dir, exist_ok=True)

    states = _states_present(con, s1_clean, other_clean)
    print(f"    {len(states)} country/state chunks")

    for country, state in states:
        safe = f"{country}_{state}".replace('/', '-')
        chunk_path = f'{chunk_dir}/{safe}.parquet'
        if os.path.exists(chunk_path):
            continue
        _process_chunk(con, s1_clean, other_clean, country, state, chunk_path, other_tag)

    tmp_final = out_parquet + '.tmp'
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{chunk_dir}/*.parquet'))
        TO '{tmp_final}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    os.replace(tmp_final, out_parquet)
    shutil.rmtree(chunk_dir)

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
