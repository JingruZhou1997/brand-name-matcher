"""
Brand Name Matcher - Streamlit App
====================================
Adintel: returns ALL matches above threshold (multi-row per brand)
Pathmatics & Media Radar: return single best match per brand
"""

import streamlit as st
import pandas as pd
import re
import io
import os
import gc
from rapidfuzz import fuzz, process
from datetime import datetime

st.set_page_config(page_title="Brand Name Matcher", page_icon="🔍", layout="wide")

TOP_N_MATCHES = 3
MATCH_THRESHOLD = 95
SUGGEST_THRESHOLD = 90
ADINTEL_REF = "adintel_brands.csv.gz"
PATHMATICS_REF = "pathmatics_brands.csv.gz"
MEDIARADAR_REF = "mediaradar_brands.csv.gz"

# ── Common abbreviations ──
ABBREVIATIONS = {
    "management": "mgmt", "mgmt": "management",
    "international": "intl", "intl": "international",
    "services": "svcs", "svcs": "services",
    "service": "svc", "svc": "service",
    "corporation": "corp", "corp": "corporation",
    "company": "co", "co": "company",
    "incorporated": "inc", "inc": "incorporated",
    "limited": "ltd", "ltd": "limited",
    "department": "dept", "dept": "department",
    "association": "assn", "assn": "association",
    "national": "natl", "natl": "national",
    "american": "amer", "amer": "american",
    "financial": "fin", "fin": "financial",
    "insurance": "ins", "ins": "insurance",
    "manufacturing": "mfg", "mfg": "manufacturing",
    "technology": "tech", "tech": "technology",
    "technologies": "tech",
    "products": "pdts", "pdts": "products",
    "product": "pdt", "pdt": "product",
    "laboratory": "lab", "lab": "laboratory",
    "laboratories": "labs", "labs": "laboratories",
    "pharmaceutical": "pharm", "pharm": "pharmaceutical",
    "pharmaceuticals": "pharma", "pharma": "pharmaceuticals",
    "communications": "comm", "comm": "communications",
    "development": "dev", "dev": "development",
    "engineering": "engr", "engr": "engineering",
    "equipment": "equip", "equip": "equipment",
    "distribution": "dist", "dist": "distribution",
    "enterprises": "ent", "ent": "enterprises",
    "properties": "prop", "prop": "properties",
    "solutions": "sol", "sol": "solutions",
    "systems": "sys", "sys": "systems",
    "medical": "med", "med": "medical",
    "healthcare": "hlthcr", "hlthcr": "healthcare",
    "health": "hlth", "hlth": "health",
    "advertising": "adv", "adv": "advertising",
    "marketing": "mktg", "mktg": "marketing",
    "doctor": "dr", "dr": "doctor",
    "university": "univ", "univ": "university",
    "education": "edu", "edu": "education",
    "restaurant": "rest", "rest": "restaurant",
    "supply": "sup", "sup": "supply",
    "group": "grp", "grp": "group",
    "partners": "ptnrs", "ptnrs": "partners",
    "holdings": "hldgs", "hldgs": "holdings",
    "industries": "ind", "ind": "industries",
    "network": "ntwk", "ntwk": "network",
    "digital": "dgtl", "dgtl": "digital",
    "construction": "const", "const": "construction",
    "consulting": "consult", "consult": "consulting",
    "automotive": "auto", "auto": "automotive",
    "electric": "elec", "elec": "electric",
    "electronics": "elec", "electrical": "elec",
    "foundation": "fdn", "fdn": "foundation",
    "brothers": "bros", "bros": "brothers",
    "center": "ctr", "ctr": "center",
}

def expand_abbreviations(text):
    words = text.split()
    changed = False
    new_words = []
    for w in words:
        if w in ABBREVIATIONS:
            new_words.append(ABBREVIATIONS[w])
            changed = True
        else:
            new_words.append(w)
    return " ".join(new_words) if changed else None

# ── Normalization ──
STRIP_SUFFIXES = (
    r'\b(inc|llc|llp|ltd|limited|lp|co|corp|corporate|corporation|company|'
    r'group|international|intl|holdings|enterprises|usa|global|worldwide|'
    r'brands|solutions|services|technologies|systems|partners|associates|'
    r'management|consulting|design|labs|laboratory|laboratories|'
    r'pty|plc|gmbh|ag|sa|nv|bv|srl|spa|'
    r'foundation|institute|association|society|organization|org|'
    r'stores|store|shop|retail|online|digital|media|network|studio|studios)\b'
)
STRIP_PATTERNS = [
    r"'s\b", STRIP_SUFFIXES, r'\b(the|a|an|of|for|and|by)\b',
    r'[.,\'\"!?;:()\-/&+@#]', r'\.\s*com\b', r'\.\s*net\b', r'\.\s*org\b', r'\.\s*io\b',
]

def normalize(name):
    if not isinstance(name, str): return ""
    s = name.strip()
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    s = s.lower()
    s = re.sub(r'[^\x00-\x7f]', ' ', s)
    s = re.sub(r'(\d)([a-z])', r'\1 \2', s)
    s = re.sub(r'([a-z])(\d)', r'\1 \2', s)
    for p in STRIP_PATTERNS: s = re.sub(p, ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def get_query_variants(query):
    variants = set()
    n = normalize(query)
    if n: variants.add(n)
    ns = n.replace(" ", "") if n else ""
    if ns: variants.add(ns)
    if n:
        expanded = expand_abbreviations(n)
        if expanded: variants.add(expanded)
    s = query.strip().lower()
    s = re.sub(r'[^\x00-\x7f]', ' ', s)
    for p in STRIP_PATTERNS: s = re.sub(p, ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if s and s != n: variants.add(s)
    sn = s.replace(" ", "") if s else ""
    if sn: variants.add(sn)
    return list(variants)

def extract_name_parts(brand):
    parts = [brand.strip()]
    before = re.split(r'\s*[\(\[]', brand)[0].strip()
    if before and before != brand.strip(): parts.append(before)
    for p in re.findall(r'[\(\[]([^\)\]]+)[\)\]]', brand):
        if p.strip() and len(p.strip()) > 2: parts.append(p.strip())
    if '/' in brand:
        for c in brand.split('/'):
            if c.strip() and c.strip() != brand.strip() and len(c.strip()) > 2: parts.append(c.strip())
    return parts

def composite_score(q, c):
    base = fuzz.token_sort_ratio(q,c)*0.30 + fuzz.token_set_ratio(q,c)*0.30 + fuzz.partial_ratio(q,c)*0.20 + fuzz.ratio(q,c)*0.20
    qt, ct = set(q.split()), set(c.split())
    if qt and ct:
        ov = len(qt & ct); mx = max(len(qt),len(ct)); mn = min(len(qt),len(ct)); ovr = ov/mx
        if ovr >= 0.5 and ov >= 2: base = min(100, base+5)
        if qt <= ct and len(qt) >= 2: base = min(100, base+8)
        if len(ct)==1 and len(qt)>=2 and ov<=1: base *= 0.70
        if mx >= 2*mn and ovr < 0.4: base *= 0.85
        if not (qt <= ct) and not (ct <= qt): base = min(base, 94)
    return base


class FastMultiColMatcher:
    """
    Indexes multiple columns into one deduped search list.
    Supports both single-best and find-all modes.
    """
    def __init__(self, df, search_cols, display_cols, cat_col):
        self.display_cols = display_cols
        self.cat_col = cat_col
        self.n_pairs = len(df)

        # Store column values as lists
        self.col_data = {}
        for col in set(search_cols + display_cols + [cat_col]):
            if col in df.columns:
                self.col_data[col] = df[col].astype(str).tolist()

        # Build deduped index: norm_string -> first row index
        norm_to_row = {}
        for col in search_cols:
            if col not in self.col_data:
                continue
            vals = self.col_data[col]
            for i, v in enumerate(vals):
                n = normalize(v)
                if n and n not in norm_to_row:
                    norm_to_row[n] = i
                ns = n.replace(" ", "") if n else ""
                if ns and ns not in norm_to_row:
                    norm_to_row[ns] = i

        self.norm_strings = list(norm_to_row.keys())
        self.norm_to_row_idx = list(norm_to_row.values())
        self.exact_lookup = norm_to_row

        # For find_all: build reverse index from Brand Core norm -> all row indices
        # This lets us find all variants of a matched brand
        self.search_cols = search_cols

    def find_best(self, query):
        """Return single best match."""
        parts = extract_name_parts(query)
        best_score = 0.0
        best_row_idx = None

        for part in parts:
            variants = get_query_variants(part)
            if not variants: continue

            for v in variants:
                if v in self.exact_lookup:
                    return self._make_result(self.exact_lookup[v], 100.0)

            for v in variants:
                seen = set()
                cands = []
                for scorer in [fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.partial_ratio]:
                    results = process.extract(v, self.norm_strings, scorer=scorer,
                                              limit=TOP_N_MATCHES * 3, score_cutoff=60)
                    if results:
                        for ns, _, idx in results:
                            if idx not in seen:
                                seen.add(idx)
                                cands.append((ns, idx))

                for ns, idx in cands:
                    sc = composite_score(v, self.norm_strings[idx])
                    if sc > best_score:
                        best_score = sc
                        best_row_idx = self.norm_to_row_idx[idx]

            if best_score >= 100: break

        if best_row_idx is not None:
            return self._make_result(best_row_idx, best_score)
        return {col: "" for col in self.display_cols} | {"category": "", "score": 0.0}

    def find_all(self, query, threshold=SUGGEST_THRESHOLD):
        """
        Return ALL rows where any search column matches the query above threshold.

        Strategy:
        1. Find the best match (to get the matched normalized string)
        2. Use that match to find all rows containing similar brand names
        3. Score each row independently and return all above threshold
        """
        parts = extract_name_parts(query)
        query_norms = set()
        for part in parts:
            for v in get_query_variants(part):
                if v: query_norms.add(v)

        if not query_norms:
            return []

        # Collect all candidate row indices with their scores
        row_scores = {}  # row_idx -> best score

        for qn in query_norms:
            # Exact matches
            if qn in self.exact_lookup:
                row_scores[self.exact_lookup[qn]] = 100.0

            # Fuzzy matches — get more candidates for find_all
            seen = set()
            for scorer in [fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.partial_ratio]:
                results = process.extract(qn, self.norm_strings, scorer=scorer,
                                          limit=50, score_cutoff=55)
                if results:
                    for ns, _, idx in results:
                        if idx not in seen:
                            seen.add(idx)
                            sc = composite_score(qn, self.norm_strings[idx])
                            ri = self.norm_to_row_idx[idx]
                            if ri not in row_scores or sc > row_scores[ri]:
                                row_scores[ri] = sc

        # Now expand: for each matched row, find ALL rows with the same Brand Core
        # This catches "LEGOLAND FLORIDA", "LEGOLAND CALIFORNIA" etc.
        if "Brand Core" in self.col_data:
            brand_cores_to_find = set()
            for ri, sc in row_scores.items():
                if sc >= threshold:
                    core = self.col_data["Brand Core"][ri].strip().upper()
                    if core:
                        brand_cores_to_find.add(core)

            # Find all rows with matching Brand Core
            if brand_cores_to_find:
                all_cores = self.col_data["Brand Core"]
                for i, core_val in enumerate(all_cores):
                    if core_val.strip().upper() in brand_cores_to_find:
                        if i not in row_scores:
                            row_scores[i] = 100.0  # Same Brand Core family = Match

        # Filter by threshold and build results
        matches = []
        for ri, sc in sorted(row_scores.items(), key=lambda x: -x[1]):
            if sc >= threshold:
                matches.append(self._make_result(ri, sc))

        return matches

    def _make_result(self, row_idx, score):
        result = {"score": score}
        for col in self.display_cols:
            result[col] = self.col_data[col][row_idx] if col in self.col_data else ""
        result["category"] = self.col_data[self.cat_col][row_idx] if self.cat_col in self.col_data else ""
        return result


def find_file(filename):
    for d in [os.path.dirname(os.path.abspath(__file__)), os.getcwd(),
              "/mount/src/brand-name-matcher", "."]:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner="Loading and indexing reference data...")
def load_and_build():
    matchers = {}

    ap = find_file(ADINTEL_REF)
    if ap:
        try:
            df = pd.read_csv(ap, compression="gzip").fillna("")
            matchers["ad"] = FastMultiColMatcher(
                df,
                search_cols=["Subsidiary", "Brand Core", "Brand Extension"],
                display_cols=["Subsidiary", "Brand Core", "Brand Extension"],
                cat_col="category"
            )
            del df; gc.collect()
        except Exception as e:
            print(f"Error loading Adintel: {e}")

    pp = find_file(PATHMATICS_REF)
    if pp:
        try:
            df = pd.read_csv(pp, compression="gzip").fillna("")
            matchers["pa"] = FastMultiColMatcher(
                df,
                search_cols=["Advertiser", "Brand Leaf"],
                display_cols=["Advertiser", "Brand Leaf"],
                cat_col="category"
            )
            del df; gc.collect()
        except Exception as e:
            print(f"Error loading Pathmatics: {e}")

    mp = find_file(MEDIARADAR_REF)
    if mp:
        try:
            df = pd.read_csv(mp, compression="gzip").fillna("")
            matchers["mr"] = FastMultiColMatcher(
                df,
                search_cols=["Parent", "Product Line"],
                display_cols=["Parent", "Product Line"],
                cat_col="category"
            )
            del df; gc.collect()
        except Exception as e:
            print(f"Error loading Media Radar: {e}")

    return matchers


def status_label(score):
    if score >= MATCH_THRESHOLD: return "✅ Match"
    elif score >= SUGGEST_THRESHOLD: return "⚠️ Needs Review"
    else: return "❌ Needs Review"


def main():
    st.title("🔍 Brand Name Matcher")
    st.markdown("Match competitive brand names against **Adintel**, **Pathmatics**, and **Media Radar** databases.")
    st.markdown("*Adintel returns **all** matching brand variants. Pathmatics & Media Radar return the single best match.*")
    st.markdown("---")

    matchers = load_and_build()

    if not matchers:
        st.error("No reference files found. Run prepare_reference_data.py first.")
        st.stop()

    parts = []
    if "ad" in matchers:
        m = matchers["ad"]
        parts.append(f"**{m.n_pairs:,}** Adintel ({len(m.norm_strings):,} unique)")
    if "pa" in matchers:
        m = matchers["pa"]
        parts.append(f"**{m.n_pairs:,}** Pathmatics ({len(m.norm_strings):,} unique)")
    if "mr" in matchers:
        m = matchers["mr"]
        parts.append(f"**{m.n_pairs:,}** Media Radar ({len(m.norm_strings):,} unique)")
    st.success("✅ Loaded: " + " | ".join(parts))

    missing = [n for k, n in [("ad","Adintel"),("pa","Pathmatics"),("mr","Media Radar")] if k not in matchers]
    if missing:
        st.warning(f"⚠️ Missing: {', '.join(missing)} — those columns will show '—'")

    st.subheader("📋 Enter Brand Names")
    st.markdown("Paste brand names below — **one per line**:")
    brand_input = st.text_area("Brand names", height=300,
        placeholder="Legoland\nBlue Cross Blue Shield\nCovetrus (SmartPak Equine)\n...",
        label_visibility="collapsed")

    col1, col2 = st.columns([1, 4])
    with col1: run_btn = st.button("🚀 Match Brands", type="primary", use_container_width=True)

    NO_PA = {"Advertiser": "", "Brand Leaf": "", "category": "", "score": 0.0}
    NO_MR = {"Parent": "", "Product Line": "", "category": "", "score": 0.0}

    if run_btn and brand_input.strip():
        brands = [b.strip() for b in brand_input.strip().split('\n') if b.strip()]
        if not brands: st.warning("No brand names found."); st.stop()

        results = []
        progress = st.progress(0, text="Matching brands...")

        for i, brand in enumerate(brands):
            # Adintel: find ALL matches
            if "ad" in matchers:
                ad_all = matchers["ad"].find_all(brand)
            else:
                ad_all = []

            # Pathmatics & Media Radar: single best match
            pa_res = matchers["pa"].find_best(brand) if "pa" in matchers else NO_PA
            mr_res = matchers["mr"].find_best(brand) if "mr" in matchers else NO_MR
            pas, mrs = pa_res["score"], mr_res["score"]

            if ad_all:
                # One row per Adintel match, Pa/MR repeated on first row only
                for j, ad_res in enumerate(ad_all):
                    ads = ad_res["score"]
                    row = {
                        "Input Brand": brand if j == 0 else "",
                        "Ad Subsidiary": ad_res.get("Subsidiary", ""),
                        "Ad Brand Core": ad_res.get("Brand Core", ""),
                        "Ad Brand Extension": ad_res.get("Brand Extension", ""),
                        "Ad Category": ad_res.get("category", ""),
                        "Ad Confidence": status_label(ads),
                    }
                    if j == 0:
                        row.update({
                            "Pa Advertiser": pa_res.get("Advertiser", "") if pas >= SUGGEST_THRESHOLD else "—",
                            "Pa Brand Leaf": pa_res.get("Brand Leaf", "") if pas >= SUGGEST_THRESHOLD else "—",
                            "Pa Category": pa_res.get("category", "") if pas >= SUGGEST_THRESHOLD else "",
                            "Pa Confidence": status_label(pas),
                            "MR Parent": mr_res.get("Parent", "") if mrs >= SUGGEST_THRESHOLD else "—",
                            "MR Product Line": mr_res.get("Product Line", "") if mrs >= SUGGEST_THRESHOLD else "—",
                            "MR Category": mr_res.get("category", "") if mrs >= SUGGEST_THRESHOLD else "",
                            "MR Confidence": status_label(mrs),
                        })
                    else:
                        row.update({
                            "Pa Advertiser": "", "Pa Brand Leaf": "", "Pa Category": "", "Pa Confidence": "",
                            "MR Parent": "", "MR Product Line": "", "MR Category": "", "MR Confidence": "",
                        })
                    row["_ad_score"] = ads
                    row["_pa_score"] = pas if j == 0 else 0
                    row["_mr_score"] = mrs if j == 0 else 0
                    row["_is_first"] = (j == 0)
                    results.append(row)
            else:
                # No Adintel match
                results.append({
                    "Input Brand": brand,
                    "Ad Subsidiary": "—", "Ad Brand Core": "—", "Ad Brand Extension": "—",
                    "Ad Category": "", "Ad Confidence": "❌ Needs Review",
                    "Pa Advertiser": pa_res.get("Advertiser", "") if pas >= SUGGEST_THRESHOLD else "—",
                    "Pa Brand Leaf": pa_res.get("Brand Leaf", "") if pas >= SUGGEST_THRESHOLD else "—",
                    "Pa Category": pa_res.get("category", "") if pas >= SUGGEST_THRESHOLD else "",
                    "Pa Confidence": status_label(pas),
                    "MR Parent": mr_res.get("Parent", "") if mrs >= SUGGEST_THRESHOLD else "—",
                    "MR Product Line": mr_res.get("Product Line", "") if mrs >= SUGGEST_THRESHOLD else "—",
                    "MR Category": mr_res.get("category", "") if mrs >= SUGGEST_THRESHOLD else "",
                    "MR Confidence": status_label(mrs),
                    "_ad_score": 0, "_pa_score": pas, "_mr_score": mrs, "_is_first": True,
                })

            progress.progress((i+1)/len(brands), text=f"Matching {i+1}/{len(brands)}: {brand}")

        progress.empty()

        # Count stats (only from first rows)
        first_rows = [r for r in results if r["_is_first"]]
        n_brands = len(first_rows)
        am = sum(1 for r in results if r["_ad_score"] >= MATCH_THRESHOLD and r["_is_first"])
        pm = sum(1 for r in first_rows if r["_pa_score"] >= MATCH_THRESHOLD)
        mm = sum(1 for r in first_rows if r["_mr_score"] >= MATCH_THRESHOLD)
        total_ad_rows = sum(1 for r in results if r["_ad_score"] >= SUGGEST_THRESHOLD)

        st.subheader(f"📊 Results ({n_brands} brands, {total_ad_rows} Adintel rows)")

        cols = st.columns(4)
        cols[0].metric("Adintel Brands Matched", f"{am}/{n_brands}")
        cols[1].metric("Adintel Total Rows", f"{total_ad_rows}")
        cols[2].metric("Pathmatics Matches", f"{pm}/{n_brands}")
        cols[3].metric("Media Radar Matches", f"{mm}/{n_brands}")

        def hl(v):
            if "✅" in str(v): return "background-color: #C6EFCE"
            elif "⚠️" in str(v): return "background-color: #FFEB9C"
            elif "❌" in str(v): return "background-color: #FFC7CE"
            return ""

        df_r = pd.DataFrame(results)
        show_cols = ["Input Brand",
                     "Ad Subsidiary", "Ad Brand Core", "Ad Brand Extension", "Ad Category", "Ad Confidence",
                     "Pa Advertiser", "Pa Brand Leaf", "Pa Category", "Pa Confidence",
                     "MR Parent", "MR Product Line", "MR Category", "MR Confidence"]
        styled = df_r[show_cols].style.applymap(hl, subset=["Ad Confidence", "Pa Confidence", "MR Confidence"])
        st.dataframe(styled, use_container_width=True, height=min(len(results)*40+50, 800))

        # Download
        st.markdown("---")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as w:
            df_r[show_cols].to_excel(w, index=False, sheet_name='Brand Match Results')
        output.seek(0)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button("📥 Download Results (Excel)", data=output,
            file_name=f"brand_match_results_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    elif run_btn:
        st.warning("Please enter brand names first.")

if __name__ == "__main__":
    main()
