
"""
Normalization pipeline for Amazon ML entity resolution.

Usage:
    from src.normalize import normalize_sources
    normalize_sources(TRAIN_DIR, TEST_DIR, OUT_DIR)
"""
import re, os, time, unicodedata
import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ---- State lookup tables ------------------------------------------------
US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC',
}
IN_STATES = {
    'AP','AR','AS','BR','CG','GA','GJ','HR','HP','JH','KA','KL','MP','MH','MN',
    'ML','MZ','NL','OD','PB','RJ','SK','TN','TG','TR','UP','UT','WB',
    'AN','CH','DN','DD','DL','JK','LA','LD','PY',
}

# ---- Regexes ------------------------------------------------------------
_WS    = re.compile(r'\s+')
_PUNCT = re.compile(r'[^\w\s]')
_STREET_NUM = re.compile(r'^\s*(\d+[a-z]?)', re.IGNORECASE)
_ZIP_US = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
_ZIP_IN = re.compile(r'\b(\d{6})\b')

SUFFIX_MAP = [
    (re.compile(r'\bprivate\s+limited\b'), 'pvt ltd'),
    (re.compile(r'\bpvt\.?\s*ltd\.?\b'),   'pvt ltd'),
    (re.compile(r'\bpublic\s+limited\b'),  'pvt ltd'),
    (re.compile(r'\blimited\b'),           'ltd'),
    (re.compile(r'\bincorporated\b'),      'inc'),
    (re.compile(r'\bcorp(oration)?\b'),    'corp'),
    (re.compile(r'\bco(mpany)?\b'),        'co'),
]

_SCRIPT_RANGES = [
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
]

def _detect_script(text):
    for ch in text:
        cp = ord(ch)
        for lo, hi, scheme in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                return scheme
    return None

def to_latin(text):
    if not text:
        return text
    scheme = _detect_script(text)
    if scheme is None:
        return text
    try:
        return transliterate(text, scheme, sanscript.ITRANS)
    except Exception:
        return text

def normalize_text(text):
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if text == '' or text.lower() in ('null', 'nan', 'none', 'na', 'n/a'):
        return None

    text = to_latin(text)
    text = unicodedata.normalize('NFKC', text).lower()
    text = _PUNCT.sub(' ', text)
    text = _WS.sub(' ', text).strip()

    for pat, rep in SUFFIX_MAP:
        text = pat.sub(rep, text)

    return text if text else None

def extract_street_num(addr_norm):
    if not addr_norm:
        return None
    m = _STREET_NUM.match(addr_norm)
    return m.group(1) if m else None

def extract_zip(addr_norm, country):
    if not addr_norm or not country:
        return None
    c = country.upper()
    if c in ('US', 'USA', 'UNITED STATES'):
        m = _ZIP_US.search(addr_norm)
        return m.group(1) if m else None
    if c in ('IN', 'INDIA'):
        m = _ZIP_IN.search(addr_norm)
        return m.group(1) if m else None
    return None

def extract_state(addr_raw, country):
    if not addr_raw or not country:
        return None
    c = country.upper()
    table = US_STATES if c in ('US', 'USA', 'UNITED STATES') else \
            IN_STATES if c in ('IN', 'INDIA') else set()
    if not table:
        return None
    tokens = re.findall(r'\b([A-Z]{2})\b', addr_raw.upper())
    for t in tokens:
        if t in table:
            return t
    return None

def normalize_source(in_parquet, out_parquet, chunksize=500_000):
    if os.path.exists(out_parquet):
        print(f"  {os.path.basename(out_parquet):40s} (already done)")
        return
    import pyarrow as pa, pyarrow.parquet as pq

    pf = pq.ParquetFile(in_parquet)
    n_total = pf.metadata.num_rows
    pass
    writer = None
    t0 = time.time()
    done = 0

    for batch in pf.iter_batches(batch_size=chunksize):
        df = batch.to_pandas()
        df['name_norm'] = df['business_name'].map(normalize_text)
        df['addr_norm'] = df['business_address'].map(normalize_text)
        df['country']   = df['country'].astype(str).str.upper().str.strip()

        df['street_num'] = df['addr_norm'].map(extract_street_num)
        df['state']      = [
            extract_state(a, c)
            for a, c in zip(df['business_address'], df['country'])
        ]
        df['zip']        = [
            extract_zip(a, c) for a, c in zip(df['addr_norm'], df['country'])
        ]

        out_df = df[[
            'entity_id', 'name_norm', 'addr_norm',
            'country', 'state', 'street_num', 'zip',
        ]]
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table.schema,
                                      compression='zstd')
        writer.write_table(table)
        done += len(df)
        pass

    if writer is not None:
        writer.close()
    print(f"  {os.path.basename(out_parquet):40s} {n_total:>10,} rows  {time.time()-t0:6.1f}s")


def normalize_sources(train_dir, test_dir, out_dir):
    jobs = [
        (f'{train_dir}/train_source1.tsv', f'{out_dir}/train_source1_clean.parquet'),
        (f'{train_dir}/train_source2.tsv', f'{out_dir}/train_source2_clean.parquet'),
        (f'{train_dir}/train_source3.tsv', f'{out_dir}/train_source3_clean.parquet'),
        (f'{test_dir}/test_source1.tsv',   f'{out_dir}/test_source1_clean.parquet'),
        (f'{test_dir}/test_source2.tsv',   f'{out_dir}/test_source2_clean.parquet'),
        (f'{test_dir}/test_source3.tsv',   f'{out_dir}/test_source3_clean.parquet'),
    ]
    for in_tsv, out_pq in jobs:
        base = os.path.basename(in_tsv).replace('.tsv', '.parquet')
        in_parquet = f'{out_dir}/{base}'
        normalize_source(in_parquet, out_pq)
