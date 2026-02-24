"""
Brand Name Matcher - Streamlit App
===================================
Deploy to Streamlit Cloud or run locally:
    pip install streamlit pandas openpyxl rapidfuzz
    streamlit run brand_matcher_app.py
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

def find_top_matches(query, norm_strings, originals, categories, top_n=3):
    parts = extract_name_parts(query)
    all_scored = {}
    for part in parts:
        pn = normalize(part)
        if not pn: continue
        for i, ns in enumerate(norm_strings):
            if ns == pn: all_scored[i] = (originals[i], 100.0, categories[i])
        seen = set()
        cands = []
        for scorer in [fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.partial_ratio]:
            for ns, _, idx in process.extract(pn, norm_strings, scorer=scorer, limit=top_n*5):
                if idx not in seen: seen.add(idx); cands.append((ns, idx))
        for ns, idx in cands:
            score = composite_score(pn, ns)
            if idx not in all_scored or score > all_scored[idx][1]:
                all_scored[idx] = (originals[idx], score, categories[idx])
    ranked = sorted(all_scored.values(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n] if ranked else [("NO MATCH", 0.0, "")]

# ── Load Data ──
@st.cache_data(show_spinner="Loading reference data...")
def load_reference_data():
    import os
    for d in [os.path.dirname(os.path.abspath(__file__)), os.getcwd(), "/mount/src/brand-name-matcher", "."]:
        ap, pp = os.path.join(d, ADINTEL_REF), os.path.join(d, PATHMATICS_REF)
        if os.path.exists(ap) and os.path.exists(pp):
            ad_df = pd.read_csv(ap, compression="gzip")
            pa_df = pd.read_csv(pp, compression="gzip")
            ad_brands = ad_df["brand"].dropna().astype(str).tolist()
            ad_cats = ad_df["category"].fillna("").astype(str).tolist() if "category" in ad_df.columns else [""]*len(ad_brands)
            pa_brands = pa_df["brand"].dropna().astype(str).tolist()
            pa_cats = pa_df["category"].fillna("").astype(str).tolist() if "category" in pa_df.columns else [""]*len(pa_brands)
            ad_norm = [normalize(b) for b in ad_brands]
            pa_norm = [normalize(b) for b in pa_brands]
            return ad_norm, ad_brands, ad_cats, pa_norm, pa_brands, pa_cats, None
    return None, None, None, None, None, None, "Reference files not found. Run prepare_reference_data.py first."

# ── UI ──
def main():
    st.title("🔍 Brand Name Matcher")
    st.markdown("Match competitive brand names against **Adintel** and **Pathmatics** databases.")
    st.markdown("---")

    data = load_reference_data()
    ad_norm, ad_originals, ad_cats, pa_norm, pa_originals, pa_cats, error = data
    if error: st.error(error); st.stop()

    st.success(f"✅ Reference data loaded: **{len(ad_originals):,}** Adintel brands | **{len(pa_originals):,}** Pathmatics brands")

    st.subheader("📋 Enter Brand Names")
    st.markdown("Paste brand names below — **one per line**:")
    brand_input = st.text_area("Brand names", height=300,
        placeholder="111skin\nAnywhere Real Estate\nLimeLife USA LLC\n...", label_visibility="collapsed")

    col1, col2 = st.columns([1, 4])
    with col1: run_button = st.button("🚀 Match Brands", type="primary", use_container_width=True)

    if run_button and brand_input.strip():
        brands = [b.strip() for b in brand_input.strip().split('\n') if b.strip()]
        if not brands: st.warning("No brand names found."); st.stop()

        results = []
        progress = st.progress(0, text="Matching brands...")

        for i, brand in enumerate(brands):
            ad_top = find_top_matches(brand, ad_norm, ad_originals, ad_cats, top_n=3)
            pa_top = find_top_matches(brand, pa_norm, pa_originals, pa_cats, top_n=3)

            ad_best, ad_score, ad_cat = ad_top[0]
            pa_best, pa_score, pa_cat = pa_top[0]

            ad_alts = ""
            if ad_score < MATCH_THRESHOLD and len(ad_top) > 1:
                al = [f"{n} [{c}] ({s:.0f})" if c else f"{n} ({s:.0f})" for n,s,c in ad_top[1:] if s >= 50]
                if al: ad_alts = " | ".join(al)

            pa_alts = ""
            if pa_score < MATCH_THRESHOLD and len(pa_top) > 1:
                al = [f"{n} [{c}] ({s:.0f})" if c else f"{n} ({s:.0f})" for n,s,c in pa_top[1:] if s >= 50]
                if al: pa_alts = " | ".join(al)

            if ad_score >= MATCH_THRESHOLD: ad_display, ad_status = ad_best, "✅ Match"
            elif ad_score >= SUGGEST_THRESHOLD: ad_display, ad_status = ad_best, "⚠️ Needs Review"
            else: ad_display, ad_status = "—", "❌ Needs Review"

            if pa_score >= MATCH_THRESHOLD: pa_display, pa_status = pa_best, "✅ Match"
            elif pa_score >= SUGGEST_THRESHOLD: pa_display, pa_status = pa_best, "⚠️ Needs Review"
            else: pa_display, pa_status = "—", "❌ Needs Review"

            results.append({
                "Input Brand": brand,
                "Adintel Match": ad_display, "Adintel Category": ad_cat if ad_score >= SUGGEST_THRESHOLD else "",
                "Adintel Confidence": ad_status, "Adintel Score": round(ad_score), "Adintel Alternatives": ad_alts,
                "Pathmatics Match": pa_display, "Pathmatics Category": pa_cat if pa_score >= SUGGEST_THRESHOLD else "",
                "Pathmatics Confidence": pa_status, "Pathmatics Score": round(pa_score), "Pathmatics Alternatives": pa_alts,
            })
            progress.progress((i+1)/len(brands), text=f"Matching {i+1}/{len(brands)}: {brand}")

        progress.empty()
        st.subheader(f"📊 Results ({len(results)} brands)")
        df_results = pd.DataFrame(results)

        c1,c2,c3 = st.columns(3)
        ad_m = sum(1 for r in results if r["Adintel Score"]>=MATCH_THRESHOLD)
        pa_m = sum(1 for r in results if r["Pathmatics Score"]>=MATCH_THRESHOLD)
        both = sum(1 for r in results if r["Adintel Score"]>=MATCH_THRESHOLD and r["Pathmatics Score"]>=MATCH_THRESHOLD)
        c1.metric("Adintel Matches", f"{ad_m}/{len(results)}")
        c2.metric("Pathmatics Matches", f"{pa_m}/{len(results)}")
        c3.metric("Both Matched", f"{both}/{len(results)}")

        def hl(val):
            if "✅" in str(val): return "background-color: #C6EFCE"
            elif "⚠️" in str(val): return "background-color: #FFEB9C"
            elif "❌" in str(val): return "background-color: #FFC7CE"
            return ""

        main_cols = ["Input Brand", "Adintel Match", "Adintel Category", "Adintel Confidence",
                     "Pathmatics Match", "Pathmatics Category", "Pathmatics Confidence"]
        styled = df_results[main_cols].style.applymap(hl, subset=["Adintel Confidence", "Pathmatics Confidence"])
        st.dataframe(styled, use_container_width=True, height=min(len(results)*40+50, 600))

        # Needs Review section
        needs_review = [r for r in results if r["Adintel Score"]<MATCH_THRESHOLD or r["Pathmatics Score"]<MATCH_THRESHOLD]
        if needs_review:
            st.markdown("---")
            st.subheader("🔎 Needs Review — Alternative Suggestions")
            for r in needs_review:
                with st.expander(f"**{r['Input Brand']}**"):
                    ca, cp = st.columns(2)
                    with ca:
                        st.markdown("**Adintel**")
                        if r["Adintel Score"] >= MATCH_THRESHOLD:
                            st.markdown(f"✅ {r['Adintel Match']} — *{r['Adintel Category']}*")
                        else:
                            nm = r["Adintel Match"] if r["Adintel Match"]!="—" else "No close match"
                            cat_str = f" — *{r['Adintel Category']}*" if r["Adintel Category"] else ""
                            st.markdown(f"Best guess: **{nm}**{cat_str} (score: {r['Adintel Score']})")
                            if r["Adintel Alternatives"]: st.markdown(f"Other options: {r['Adintel Alternatives']}")
                    with cp:
                        st.markdown("**Pathmatics**")
                        if r["Pathmatics Score"] >= MATCH_THRESHOLD:
                            st.markdown(f"✅ {r['Pathmatics Match']} — *{r['Pathmatics Category']}*")
                        else:
                            nm = r["Pathmatics Match"] if r["Pathmatics Match"]!="—" else "No close match"
                            cat_str = f" — *{r['Pathmatics Category']}*" if r["Pathmatics Category"] else ""
                            st.markdown(f"Best guess: **{nm}**{cat_str} (score: {r['Pathmatics Score']})")
                            if r["Pathmatics Alternatives"]: st.markdown(f"Other options: {r['Pathmatics Alternatives']}")

        # Download
        st.markdown("---")
        dl_cols = ["Input Brand", "Adintel Match", "Adintel Category", "Adintel Confidence", "Adintel Alternatives",
                   "Pathmatics Match", "Pathmatics Category", "Pathmatics Confidence", "Pathmatics Alternatives"]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as w:
            df_results[dl_cols].to_excel(w, index=False, sheet_name='Brand Match Results')
        output.seek(0)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button("📥 Download Results (Excel)", data=output,
            file_name=f"brand_match_results_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    elif run_button:
        st.warning("Please enter brand names first.")

if __name__ == "__main__":
    main()
