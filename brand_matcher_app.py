"""
Brand Name Matcher - Streamlit App
===================================
Deploy to Streamlit Cloud or run locally:
    pip install streamlit pandas openpyxl rapidfuzz
    streamlit run brand_matcher_app.py

Place these reference files in the SAME FOLDER as this script:
    - adintel_brands.csv        (preprocessed: unique Brand Core + Subsidiary)
    - pathmatics_brands.csv     (preprocessed: unique brand names from all columns)

To generate these reference files from your raw data, run: prepare_reference_data.py
"""

import streamlit as st
import pandas as pd
import re
import io
from rapidfuzz import fuzz, process
from datetime import datetime

st.set_page_config(page_title="Brand Name Matcher", page_icon="🔍", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
TOP_N_MATCHES = 3
MATCH_THRESHOLD = 95
SUGGEST_THRESHOLD = 90

ADINTEL_REF = "adintel_brands.csv"
PATHMATICS_REF = "pathmatics_brands.csv"


# ══════════════════════════════════════════════════════════════════════════════
# TEXT NORMALIZATION (same logic as the standalone script)
# ══════════════════════════════════════════════════════════════════════════════
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
    r"'s\b",
    STRIP_SUFFIXES,
    r'\b(the|a|an|of|for|and|by)\b',
    r'[.,\'\"!?;:()\-/&+@#]',
    r'\.\s*com\b',
    r'\.\s*net\b',
    r'\.\s*org\b',
    r'\.\s*io\b',
]


def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    s = s.lower()
    s = re.sub(r'[^\x00-\x7f]', ' ', s)
    s = re.sub(r'(\d)([a-z])', r'\1 \2', s)
    s = re.sub(r'([a-z])(\d)', r'\1 \2', s)
    for pattern in STRIP_PATTERNS:
        s = re.sub(pattern, ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def extract_name_parts(brand: str) -> list:
    parts = [brand.strip()]
    before_paren = re.split(r'\s*[\(\[]', brand)[0].strip()
    if before_paren and before_paren != brand.strip():
        parts.append(before_paren)
    inside_paren = re.findall(r'[\(\[]([^\)\]]+)[\)\]]', brand)
    for p in inside_paren:
        p = p.strip()
        if p and len(p) > 2:
            parts.append(p)
    if '/' in brand:
        for chunk in brand.split('/'):
            chunk = chunk.strip()
            if chunk and chunk != brand.strip() and len(chunk) > 2:
                parts.append(chunk)
    return parts


# ══════════════════════════════════════════════════════════════════════════════
# MATCHING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def composite_score(query_norm: str, candidate_norm: str) -> float:
    base = (
        fuzz.token_sort_ratio(query_norm, candidate_norm) * 0.30
        + fuzz.token_set_ratio(query_norm, candidate_norm) * 0.30
        + fuzz.partial_ratio(query_norm, candidate_norm)   * 0.20
        + fuzz.ratio(query_norm, candidate_norm)           * 0.20
    )
    q_tokens = set(query_norm.split())
    c_tokens = set(candidate_norm.split())
    if q_tokens and c_tokens:
        overlap = len(q_tokens & c_tokens)
        max_tokens = max(len(q_tokens), len(c_tokens))
        min_tokens = min(len(q_tokens), len(c_tokens))
        overlap_ratio = overlap / max_tokens
        if overlap_ratio >= 0.5 and overlap >= 2:
            base = min(100, base + 5)
        if q_tokens <= c_tokens and len(q_tokens) >= 2:
            base = min(100, base + 8)
        if len(c_tokens) == 1 and len(q_tokens) >= 2 and overlap <= 1:
            base *= 0.70
        if max_tokens >= 2 * min_tokens and overlap_ratio < 0.4:
            base *= 0.85
    return base


def find_best_match(query, norm_strings, originals):
    parts = extract_name_parts(query)
    best_score, best_original = 0.0, "NO MATCH"
    for part in parts:
        part_norm = normalize(part)
        if not part_norm:
            continue
        # Exact match
        for i, ns in enumerate(norm_strings):
            if ns == part_norm:
                return (originals[i], 100.0)
        # Fuzzy match
        seen_idx = set()
        all_candidates = []
        for scorer in [fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.partial_ratio]:
            for norm_str, _, idx in process.extract(part_norm, norm_strings, scorer=scorer, limit=TOP_N_MATCHES * 3):
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    all_candidates.append((norm_str, idx))
        for norm_str, idx in all_candidates:
            score = composite_score(part_norm, norm_str)
            if score > best_score:
                best_score, best_original = score, originals[idx]
        if best_score >= 100:
            break
    return (best_original, best_score)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD REFERENCE DATA (cached so it only loads once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading reference data...")
def load_reference_data():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ad_path = os.path.join(script_dir, ADINTEL_REF)
    pa_path = os.path.join(script_dir, PATHMATICS_REF)

    if not os.path.exists(ad_path) or not os.path.exists(pa_path):
        return None, None, None, None, f"Reference files not found in {script_dir}. Run prepare_reference_data.py first."

    ad_df = pd.read_csv(ad_path)
    pa_df = pd.read_csv(pa_path)

    ad_brands = ad_df.iloc[:, 0].dropna().unique().tolist()
    pa_brands = pa_df.iloc[:, 0].dropna().unique().tolist()

    ad_norm = [normalize(b) for b in ad_brands]
    pa_norm = [normalize(b) for b in pa_brands]

    return (ad_norm, ad_brands, pa_norm, pa_brands, None)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.title("🔍 Brand Name Matcher")
    st.markdown("Match competitive brand names against **Adintel** and **Pathmatics** databases.")
    st.markdown("---")

    # Load reference data
    ad_norm, ad_originals, pa_norm, pa_originals, error = load_reference_data()

    if error:
        st.error(error)
        st.stop()

    st.success(f"✅ Reference data loaded: **{len(ad_originals):,}** Adintel brands | **{len(pa_originals):,}** Pathmatics brands")

    # Input area
    st.subheader("📋 Enter Brand Names")
    st.markdown("Paste brand names below — **one per line**:")

    brand_input = st.text_area(
        "Brand names",
        height=300,
        placeholder="111skin\nAnywhere Real Estate\nBetter Homes & Gardens RE\nBIG Y FOODS, INC.\nPure Fitness (Blink Holdings, Inc.)\n...",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("🚀 Match Brands", type="primary", use_container_width=True)

    if run_button and brand_input.strip():
        brands = [b.strip() for b in brand_input.strip().split('\n') if b.strip()]

        if not brands:
            st.warning("No brand names found. Please enter at least one brand name.")
            st.stop()

        # Run matching
        results = []
        progress = st.progress(0, text="Matching brands...")

        for i, brand in enumerate(brands):
            ad_match, ad_score = find_best_match(brand, ad_norm, ad_originals)
            pa_match, pa_score = find_best_match(brand, pa_norm, pa_originals)

            # Determine display values
            if ad_score >= MATCH_THRESHOLD:
                ad_display, ad_status = ad_match, "✅ Match"
            elif ad_score >= SUGGEST_THRESHOLD:
                ad_display, ad_status = ad_match, "⚠️ Needs Review"
            else:
                ad_display, ad_status = "—", "❌ Needs Review"

            if pa_score >= MATCH_THRESHOLD:
                pa_display, pa_status = pa_match, "✅ Match"
            elif pa_score >= SUGGEST_THRESHOLD:
                pa_display, pa_status = pa_match, "⚠️ Needs Review"
            else:
                pa_display, pa_status = "—", "❌ Needs Review"

            results.append({
                "Input Brand": brand,
                "Adintel Match": ad_display,
                "Adintel Confidence": ad_status,
                "Adintel Score": round(ad_score),
                "Pathmatics Match": pa_display,
                "Pathmatics Confidence": pa_status,
                "Pathmatics Score": round(pa_score),
            })

            progress.progress((i + 1) / len(brands), text=f"Matching {i+1}/{len(brands)}: {brand}")

        progress.empty()

        # Display results
        st.subheader(f"📊 Results ({len(results)} brands)")

        df_results = pd.DataFrame(results)

        # Summary stats
        col1, col2, col3 = st.columns(3)
        ad_matched = sum(1 for r in results if r["Adintel Score"] >= MATCH_THRESHOLD)
        pa_matched = sum(1 for r in results if r["Pathmatics Score"] >= MATCH_THRESHOLD)
        both_matched = sum(1 for r in results if r["Adintel Score"] >= MATCH_THRESHOLD and r["Pathmatics Score"] >= MATCH_THRESHOLD)
        col1.metric("Adintel Matches", f"{ad_matched}/{len(results)}")
        col2.metric("Pathmatics Matches", f"{pa_matched}/{len(results)}")
        col3.metric("Both Matched", f"{both_matched}/{len(results)}")

        # Color-coded table
        def highlight_confidence(val):
            if "✅" in str(val):
                return "background-color: #C6EFCE"
            elif "⚠️" in str(val):
                return "background-color: #FFEB9C"
            elif "❌" in str(val):
                return "background-color: #FFC7CE"
            return ""

        # Display columns (hide score columns from view)
        display_cols = ["Input Brand", "Adintel Match", "Adintel Confidence", "Pathmatics Match", "Pathmatics Confidence"]
        styled_df = df_results[display_cols].style.applymap(
            highlight_confidence, subset=["Adintel Confidence", "Pathmatics Confidence"]
        )
        st.dataframe(styled_df, use_container_width=True, height=min(len(results) * 40 + 50, 600))

        # Download button
        st.markdown("---")

        # Create Excel download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_results[display_cols].to_excel(writer, index=False, sheet_name='Brand Match Results')
        output.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label="📥 Download Results (Excel)",
            data=output,
            file_name=f"brand_match_results_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

    elif run_button:
        st.warning("Please enter brand names first.")


if __name__ == "__main__":
    main()
