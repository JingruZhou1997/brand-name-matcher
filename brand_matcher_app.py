"""
Brand Name Matcher - Streamlit App (Optimized)
================================================
Matches against Adintel, Pathmatics, and Media Radar.
Fast: deduplicated index, dict exact match, single fuzzy search per source.
"""

import streamlit as st
import pandas as pd
import re
import io
from rapidfuzz import fuzz, process
from datetime import datetime

st.set_page_config(page_title="Brand Name Matcher", page_icon="🔍", layout="wide")

TOP_N_MATCHES = 3
MATCH_THRESHOLD = 95
SUGGEST_THRESHOLD = 90
ADINTEL_REF = "adintel_brands.csv.gz"
PATHMATICS_REF = "pathmatics_brands.csv.gz"
MEDIARADAR_REF = "mediaradar_brands.csv.gz"

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


class FastPairedMatcher:
    """
    Fast matching: deduplicates normalized names from both columns into one
    searchable list. Maps matches back to paired rows.
    """
    def __init__(self, df, col1, col2, cat_col):
        self.df = df
        self.col1 = col1
        self.col2 = col2
        self.cat_col = cat_col

        c1_norms = [normalize(str(v)) for v in df[col1].tolist()]
        c2_norms = [normalize(str(v)) for v in df[col2].tolist()]

        norm_to_row = {}
        for i, n in enumerate(c1_norms):
            if n and n not in norm_to_row:
                norm_to_row[n] = i
        for i, n in enumerate(c2_norms):
            if n and n not in norm_to_row:
                norm_to_row[n] = i

        self.norm_strings = list(norm_to_row.keys())
        self.norm_to_row_idx = list(norm_to_row.values())
        self.exact_lookup = norm_to_row

    def find_best(self, query):
        parts = extract_name_parts(query)
        best_score = 0.0
        best_row_idx = None

        for part in parts:
            pn = normalize(part)
            if not pn:
                continue

            if pn in self.exact_lookup:
                return self._make_result(self.exact_lookup[pn], 100.0)

            seen = set()
            cands = []
            for scorer in [fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.partial_ratio]:
                results = process.extract(pn, self.norm_strings, scorer=scorer,
                                          limit=TOP_N_MATCHES * 3, score_cutoff=60)
                if results:
                    for ns, _, idx in results:
                        if idx not in seen:
                            seen.add(idx)
                            cands.append((ns, idx))

            for ns, idx in cands:
                sc = composite_score(pn, self.norm_strings[idx])
                if sc > best_score:
                    best_score = sc
                    best_row_idx = self.norm_to_row_idx[idx]

            if best_score >= 100:
                break

        if best_row_idx is not None:
            return self._make_result(best_row_idx, best_score)
        return {"col1": "NO MATCH", "col2": "", "category": "", "score": 0.0}

    def _make_result(self, row_idx, score):
        row = self.df.iloc[row_idx]
        return {
            "col1": str(row[self.col1]),
            "col2": str(row[self.col2]),
            "category": str(row.get(self.cat_col, "")),
            "score": score,
        }


# ── Load Data ──
@st.cache_data(show_spinner="Loading reference data...")
def load_reference_data():
    import os
    for d in [os.path.dirname(os.path.abspath(__file__)), os.getcwd(), "/mount/src/brand-name-matcher", "."]:
        ap = os.path.join(d, ADINTEL_REF)
        pp = os.path.join(d, PATHMATICS_REF)
        mp = os.path.join(d, MEDIARADAR_REF)
        if os.path.exists(ap) and os.path.exists(pp) and os.path.exists(mp):
            ad = pd.read_csv(ap, compression="gzip").fillna("")
            pa = pd.read_csv(pp, compression="gzip").fillna("")
            mr = pd.read_csv(mp, compression="gzip").fillna("")
            return ad, pa, mr, None
    return None, None, None, "Reference files not found. Run prepare_reference_data.py first."


@st.cache_resource(show_spinner="Building search index (one-time)...")
def build_matchers(_ad, _pa, _mr):
    ad_matcher = FastPairedMatcher(_ad, "Brand Core", "Subsidiary", "category")
    pa_matcher = FastPairedMatcher(_pa, "Advertiser", "Brand Leaf", "category")
    mr_matcher = FastPairedMatcher(_mr, "Parent", "Product Line", "category")
    return ad_matcher, pa_matcher, mr_matcher


# ── UI ──
def main():
    st.title("🔍 Brand Name Matcher")
    st.markdown("Match competitive brand names against **Adintel**, **Pathmatics**, and **Media Radar** databases.")
    st.markdown("---")

    ad_df, pa_df, mr_df, error = load_reference_data()
    if error: st.error(error); st.stop()

    ad_matcher, pa_matcher, mr_matcher = build_matchers(ad_df, pa_df, mr_df)

    st.success(
        f"✅ Loaded: **{len(ad_df):,}** Adintel pairs ({len(ad_matcher.norm_strings):,} unique) | "
        f"**{len(pa_df):,}** Pathmatics pairs ({len(pa_matcher.norm_strings):,} unique) | "
        f"**{len(mr_df):,}** Media Radar pairs ({len(mr_matcher.norm_strings):,} unique)"
    )

    st.subheader("📋 Enter Brand Names")
    st.markdown("Paste brand names below — **one per line**:")
    brand_input = st.text_area("Brand names", height=300,
        placeholder="111skin\nLimeLife USA LLC\nCovetrus (SmartPak Equine)\n...", label_visibility="collapsed")

    col1, col2 = st.columns([1, 4])
    with col1: run_btn = st.button("🚀 Match Brands", type="primary", use_container_width=True)

    if run_btn and brand_input.strip():
        brands = [b.strip() for b in brand_input.strip().split('\n') if b.strip()]
        if not brands: st.warning("No brand names found."); st.stop()

        results = []
        progress = st.progress(0, text="Matching brands...")

        for i, brand in enumerate(brands):
            ad_res = ad_matcher.find_best(brand)
            pa_res = pa_matcher.find_best(brand)
            mr_res = mr_matcher.find_best(brand)
            ads, pas, mrs = ad_res["score"], pa_res["score"], mr_res["score"]

            def status(score):
                if score >= MATCH_THRESHOLD: return "✅ Match"
                elif score >= SUGGEST_THRESHOLD: return "⚠️ Needs Review"
                else: return "❌ Needs Review"

            def val_or_dash(res, key, score):
                return res[key] if score >= SUGGEST_THRESHOLD else "—"

            def cat_or_empty(res, score):
                return res["category"] if score >= SUGGEST_THRESHOLD else ""

            results.append({
                "Input Brand": brand,
                "Ad Brand Core": val_or_dash(ad_res, "col1", ads),
                "Ad Subsidiary": val_or_dash(ad_res, "col2", ads),
                "Ad Category": cat_or_empty(ad_res, ads),
                "Ad Confidence": status(ads), "Ad Score": round(ads),
                "Pa Advertiser": val_or_dash(pa_res, "col1", pas),
                "Pa Brand Leaf": val_or_dash(pa_res, "col2", pas),
                "Pa Category": cat_or_empty(pa_res, pas),
                "Pa Confidence": status(pas), "Pa Score": round(pas),
                "MR Parent": val_or_dash(mr_res, "col1", mrs),
                "MR Product Line": val_or_dash(mr_res, "col2", mrs),
                "MR Category": cat_or_empty(mr_res, mrs),
                "MR Confidence": status(mrs), "MR Score": round(mrs),
            })
            progress.progress((i+1)/len(brands), text=f"Matching {i+1}/{len(brands)}: {brand}")

        progress.empty()
        st.subheader(f"📊 Results ({len(results)} brands)")
        df_r = pd.DataFrame(results)

        c1,c2,c3,c4 = st.columns(4)
        am = sum(1 for r in results if r["Ad Score"]>=MATCH_THRESHOLD)
        pm = sum(1 for r in results if r["Pa Score"]>=MATCH_THRESHOLD)
        mm = sum(1 for r in results if r["MR Score"]>=MATCH_THRESHOLD)
        allm = sum(1 for r in results if r["Ad Score"]>=MATCH_THRESHOLD and r["Pa Score"]>=MATCH_THRESHOLD and r["MR Score"]>=MATCH_THRESHOLD)
        c1.metric("Adintel Matches", f"{am}/{len(results)}")
        c2.metric("Pathmatics Matches", f"{pm}/{len(results)}")
        c3.metric("Media Radar Matches", f"{mm}/{len(results)}")
        c4.metric("All Three Matched", f"{allm}/{len(results)}")

        def hl(v):
            if "✅" in str(v): return "background-color: #C6EFCE"
            elif "⚠️" in str(v): return "background-color: #FFEB9C"
            elif "❌" in str(v): return "background-color: #FFC7CE"
            return ""

        show_cols = ["Input Brand",
                     "Ad Brand Core", "Ad Subsidiary", "Ad Category", "Ad Confidence",
                     "Pa Advertiser", "Pa Brand Leaf", "Pa Category", "Pa Confidence",
                     "MR Parent", "MR Product Line", "MR Category", "MR Confidence"]
        styled = df_r[show_cols].style.applymap(hl, subset=["Ad Confidence", "Pa Confidence", "MR Confidence"])
        st.dataframe(styled, use_container_width=True, height=min(len(results)*40+50, 600))

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
