"""
Brand Name Matcher - Streamlit App
===================================
Searches Brand Core AND Subsidiary (Adintel), Advertiser AND Brand Leaf (Pathmatics).
Shows paired columns so users see the full brand hierarchy.
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

# ── Scoring ──
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


def find_best_in_paired(query, df_ref, col1_norm, col2_norm, col1_name, col2_name, cat_col):
    """
    Search BOTH columns of a paired reference. For each input query part,
    search col1 and col2 independently, return the best match row.
    """
    parts = extract_name_parts(query)
    best_score = 0.0
    best_row = None

    for part in parts:
        pn = normalize(part)
        if not pn:
            continue

        # Search in both columns
        for norm_list, col_name in [(col1_norm, col1_name), (col2_norm, col2_name)]:
            # Exact match
            for i, ns in enumerate(norm_list):
                if ns == pn and 100.0 > best_score:
                    best_score = 100.0
                    best_row = df_ref.iloc[i]

            # Fuzzy match
            seen = set()
            cands = []
            for scorer in [fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.partial_ratio]:
                for ns, _, idx in process.extract(pn, norm_list, scorer=scorer, limit=TOP_N_MATCHES * 3):
                    if idx not in seen:
                        seen.add(idx)
                        cands.append((ns, idx))

            for ns, idx in cands:
                sc = composite_score(pn, ns)
                if sc > best_score:
                    best_score = sc
                    best_row = df_ref.iloc[idx]

        if best_score >= 100:
            break

    if best_row is not None:
        return {
            "col1": str(best_row[col1_name]),
            "col2": str(best_row[col2_name]),
            "category": str(best_row[cat_col]) if cat_col in best_row.index else "",
            "score": best_score,
        }
    return {"col1": "NO MATCH", "col2": "", "category": "", "score": 0.0}


# ── Load Data ──
@st.cache_data(show_spinner="Loading reference data...")
def load_reference_data():
    import os
    for d in [os.path.dirname(os.path.abspath(__file__)), os.getcwd(), "/mount/src/brand-name-matcher", "."]:
        ap, pp = os.path.join(d, ADINTEL_REF), os.path.join(d, PATHMATICS_REF)
        if os.path.exists(ap) and os.path.exists(pp):
            ad = pd.read_csv(ap, compression="gzip").fillna("")
            pa = pd.read_csv(pp, compression="gzip").fillna("")

            # Normalize both columns for Adintel
            ad_col1_norm = [normalize(str(x)) for x in ad["Brand Core"]]
            ad_col2_norm = [normalize(str(x)) for x in ad["Subsidiary"]]

            # Normalize both columns for Pathmatics
            pa_col1_norm = [normalize(str(x)) for x in pa["Advertiser"]]
            pa_col2_norm = [normalize(str(x)) for x in pa["Brand Leaf"]]

            return ad, ad_col1_norm, ad_col2_norm, pa, pa_col1_norm, pa_col2_norm, None
    return None, None, None, None, None, None, "Reference files not found. Run prepare_reference_data.py first."


# ── UI ──
def main():
    st.title("🔍 Brand Name Matcher")
    st.markdown("Match competitive brand names against **Adintel** and **Pathmatics** databases.")
    st.markdown("---")

    ad, ad_c1n, ad_c2n, pa, pa_c1n, pa_c2n, error = load_reference_data()
    if error: st.error(error); st.stop()

    st.success(f"✅ Loaded: **{len(ad):,}** Adintel pairs | **{len(pa):,}** Pathmatics pairs")

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
            ad_res = find_best_in_paired(brand, ad, ad_c1n, ad_c2n, "Brand Core", "Subsidiary", "category")
            pa_res = find_best_in_paired(brand, pa, pa_c1n, pa_c2n, "Advertiser", "Brand Leaf", "category")

            ads, pas = ad_res["score"], pa_res["score"]

            if ads >= MATCH_THRESHOLD: ad_status = "✅ Match"
            elif ads >= SUGGEST_THRESHOLD: ad_status = "⚠️ Needs Review"
            else: ad_status = "❌ Needs Review"

            if pas >= MATCH_THRESHOLD: pa_status = "✅ Match"
            elif pas >= SUGGEST_THRESHOLD: pa_status = "⚠️ Needs Review"
            else: pa_status = "❌ Needs Review"

            results.append({
                "Input Brand": brand,
                "Ad Brand Core": ad_res["col1"] if ads >= SUGGEST_THRESHOLD else "—",
                "Ad Subsidiary": ad_res["col2"] if ads >= SUGGEST_THRESHOLD else "—",
                "Ad Category": ad_res["category"] if ads >= SUGGEST_THRESHOLD else "",
                "Ad Confidence": ad_status,
                "Ad Score": round(ads),
                "Pa Advertiser": pa_res["col1"] if pas >= SUGGEST_THRESHOLD else "—",
                "Pa Brand Leaf": pa_res["col2"] if pas >= SUGGEST_THRESHOLD else "—",
                "Pa Category": pa_res["category"] if pas >= SUGGEST_THRESHOLD else "",
                "Pa Confidence": pa_status,
                "Pa Score": round(pas),
            })
            progress.progress((i+1)/len(brands), text=f"Matching {i+1}/{len(brands)}: {brand}")

        progress.empty()
        st.subheader(f"📊 Results ({len(results)} brands)")
        df_r = pd.DataFrame(results)

        c1,c2,c3 = st.columns(3)
        am = sum(1 for r in results if r["Ad Score"]>=MATCH_THRESHOLD)
        pm = sum(1 for r in results if r["Pa Score"]>=MATCH_THRESHOLD)
        bm = sum(1 for r in results if r["Ad Score"]>=MATCH_THRESHOLD and r["Pa Score"]>=MATCH_THRESHOLD)
        c1.metric("Adintel Matches", f"{am}/{len(results)}")
        c2.metric("Pathmatics Matches", f"{pm}/{len(results)}")
        c3.metric("Both Matched", f"{bm}/{len(results)}")

        def hl(v):
            if "✅" in str(v): return "background-color: #C6EFCE"
            elif "⚠️" in str(v): return "background-color: #FFEB9C"
            elif "❌" in str(v): return "background-color: #FFC7CE"
            return ""

        show_cols = ["Input Brand", "Ad Brand Core", "Ad Subsidiary", "Ad Category", "Ad Confidence",
                     "Pa Advertiser", "Pa Brand Leaf", "Pa Category", "Pa Confidence"]
        styled = df_r[show_cols].style.applymap(hl, subset=["Ad Confidence", "Pa Confidence"])
        st.dataframe(styled, use_container_width=True, height=min(len(results)*40+50, 600))

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
