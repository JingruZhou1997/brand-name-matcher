"""
Brand Name Matcher - Streamlit App
====================================
Matches against Adintel, Pathmatics, and Media Radar.
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

# ── Common abbreviations (both directions) ──
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
    "technologies": "tech", "tech": "technologies",
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
    "real estate": "re", "re": "real estate",
    "doctor": "dr", "dr": "doctor",
    "university": "univ", "univ": "university",
    "education": "edu", "edu": "education",
    "restaurant": "rest", "rest": "restaurant",
    "restaurants": "rests", "rests": "restaurants",
    "supply": "sup", "sup": "supply",
    "group": "grp", "grp": "group",
    "partners": "ptnrs", "ptnrs": "partners",
    "holdings": "hldgs", "hldgs": "holdings",
    "industries": "ind", "ind": "industries",
    "network": "ntwk", "ntwk": "network",
    "digital": "dgtl", "dgtl": "digital",
    "federal": "fed", "fed": "federal",
    "government": "govt", "govt": "government",
    "construction": "const", "const": "construction",
    "consulting": "consult", "consult": "consulting",
    "furniture": "furn", "furn": "furniture",
    "automotive": "auto", "auto": "automotive",
    "electric": "elec", "elec": "electric",
    "electronics": "elec", "electrical": "elec",
    "foundation": "fdn", "fdn": "foundation",
    "brothers": "bros", "bros": "brothers",
}

def expand_abbreviations(text):
    """Generate an alternate version with abbreviations expanded/contracted."""
    words = text.split()
    changed = False
    new_words = []
    for w in words:
        if w in ABBREVIATIONS:
            new_words.append(ABBREVIATIONS[w])
            changed = True
        else:
            new_words.append(w)
    if changed:
        return " ".join(new_words)
    return None

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
    # CamelCase splitting
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    s = s.lower()
    # Remove non-ASCII
    s = re.sub(r'[^\x00-\x7f]', ' ', s)
    # Separate digits from letters
    s = re.sub(r'(\d)([a-z])', r'\1 \2', s)
    s = re.sub(r'([a-z])(\d)', r'\1 \2', s)
    # Strip patterns (apostrophes, suffixes, stop words, punctuation)
    for p in STRIP_PATTERNS: s = re.sub(p, ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def normalize_nospace(name):
    """Additional variant: remove all spaces (catches L'Occitane -> loccitane)."""
    n = normalize(name)
    return n.replace(" ", "") if n else ""

def get_query_variants(query):
    """
    Generate multiple normalized variants of a query to try matching against.
    Returns list of unique non-empty normalized strings.
    """
    variants = set()

    n = normalize(query)
    if n: variants.add(n)

    # No-space version (L'Occitane -> loccitane)
    ns = normalize_nospace(query)
    if ns: variants.add(ns)

    # Abbreviation-expanded version
    if n:
        expanded = expand_abbreviations(n)
        if expanded: variants.add(expanded)

    # Also try without CamelCase splitting (MDSolarSciences -> mdsolarsciencescorp -> mdsolarsciences)
    s = query.strip().lower()
    s = re.sub(r'[^\x00-\x7f]', ' ', s)
    for p in STRIP_PATTERNS: s = re.sub(p, ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if s and s != n: variants.add(s)

    # No-space of non-CamelCase version
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


class FastPairedMatcher:
    def __init__(self, df, col1, col2, cat_col):
        self.col1_vals = df[col1].astype(str).tolist()
        self.col2_vals = df[col2].astype(str).tolist()
        self.cat_vals = df[cat_col].astype(str).tolist() if cat_col in df.columns else [""] * len(df)
        self.col1 = col1
        self.col2 = col2

        c1_norms = [normalize(v) for v in self.col1_vals]
        c2_norms = [normalize(v) for v in self.col2_vals]

        # Also index no-space versions for matching things like LOccitane
        norm_to_row = {}
        for i, n in enumerate(c1_norms):
            if n and n not in norm_to_row:
                norm_to_row[n] = i
            # No-space variant
            ns = n.replace(" ", "") if n else ""
            if ns and ns not in norm_to_row:
                norm_to_row[ns] = i
        for i, n in enumerate(c2_norms):
            if n and n not in norm_to_row:
                norm_to_row[n] = i
            ns = n.replace(" ", "") if n else ""
            if ns and ns not in norm_to_row:
                norm_to_row[ns] = i

        self.norm_strings = list(norm_to_row.keys())
        self.norm_to_row_idx = list(norm_to_row.values())
        self.exact_lookup = norm_to_row
        self.n_pairs = len(df)

    def find_best(self, query):
        parts = extract_name_parts(query)
        best_score = 0.0
        best_row_idx = None

        for part in parts:
            # Generate all variants of this query part
            variants = get_query_variants(part)
            if not variants:
                continue

            # Try exact match with all variants first
            for v in variants:
                if v in self.exact_lookup:
                    return self._make_result(self.exact_lookup[v], 100.0)

            # Fuzzy match — try each variant, keep best overall
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

            if best_score >= 100:
                break

        if best_row_idx is not None:
            return self._make_result(best_row_idx, best_score)
        return {"col1": "NO MATCH", "col2": "", "category": "", "score": 0.0}

    def _make_result(self, row_idx, score):
        return {
            "col1": self.col1_vals[row_idx],
            "col2": self.col2_vals[row_idx],
            "category": self.cat_vals[row_idx],
            "score": score,
        }


def find_file(filename):
    for d in [os.path.dirname(os.path.abspath(__file__)), os.getcwd(),
              "/mount/src/brand-name-matcher", "."]:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


NO_MATCH = {"col1": "", "col2": "", "category": "", "score": 0.0}


@st.cache_resource(show_spinner="Loading and indexing reference data...")
def load_and_build():
    matchers = {}

    ap = find_file(ADINTEL_REF)
    if ap:
        try:
            df = pd.read_csv(ap, compression="gzip").fillna("")
            matchers["ad"] = FastPairedMatcher(df, "Subsidiary", "Brand Core", "category")
            del df; gc.collect()
        except Exception as e:
            print(f"Error loading Adintel: {e}")

    pp = find_file(PATHMATICS_REF)
    if pp:
        try:
            df = pd.read_csv(pp, compression="gzip").fillna("")
            matchers["pa"] = FastPairedMatcher(df, "Advertiser", "Brand Leaf", "category")
            del df; gc.collect()
        except Exception as e:
            print(f"Error loading Pathmatics: {e}")

    mp = find_file(MEDIARADAR_REF)
    if mp:
        try:
            df = pd.read_csv(mp, compression="gzip").fillna("")
            matchers["mr"] = FastPairedMatcher(df, "Parent", "Product Line", "category")
            del df; gc.collect()
        except Exception as e:
            print(f"Error loading Media Radar: {e}")

    return matchers


def main():
    st.title("🔍 Brand Name Matcher")
    st.markdown("Match competitive brand names against **Adintel**, **Pathmatics**, and **Media Radar** databases.")
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
        placeholder="111skin\nLimeLife USA LLC\nCovetrus (SmartPak Equine)\n...", label_visibility="collapsed")

    col1, col2 = st.columns([1, 4])
    with col1: run_btn = st.button("🚀 Match Brands", type="primary", use_container_width=True)

    if run_btn and brand_input.strip():
        brands = [b.strip() for b in brand_input.strip().split('\n') if b.strip()]
        if not brands: st.warning("No brand names found."); st.stop()

        results = []
        progress = st.progress(0, text="Matching brands...")

        for i, brand in enumerate(brands):
            ad_res = matchers["ad"].find_best(brand) if "ad" in matchers else NO_MATCH
            pa_res = matchers["pa"].find_best(brand) if "pa" in matchers else NO_MATCH
            mr_res = matchers["mr"].find_best(brand) if "mr" in matchers else NO_MATCH
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
                "Ad Subsidiary": val_or_dash(ad_res, "col1", ads),
                "Ad Brand Core": val_or_dash(ad_res, "col2", ads),
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

        cols = st.columns(4)
        am = sum(1 for r in results if r["Ad Score"]>=MATCH_THRESHOLD)
        pm = sum(1 for r in results if r["Pa Score"]>=MATCH_THRESHOLD)
        mm = sum(1 for r in results if r["MR Score"]>=MATCH_THRESHOLD)
        allm = sum(1 for r in results if r["Ad Score"]>=MATCH_THRESHOLD and r["Pa Score"]>=MATCH_THRESHOLD and r["MR Score"]>=MATCH_THRESHOLD)
        cols[0].metric("Adintel Matches", f"{am}/{len(results)}")
        cols[1].metric("Pathmatics Matches", f"{pm}/{len(results)}")
        cols[2].metric("Media Radar Matches", f"{mm}/{len(results)}")
        cols[3].metric("All Three Matched", f"{allm}/{len(results)}")

        def hl(v):
            if "✅" in str(v): return "background-color: #C6EFCE"
            elif "⚠️" in str(v): return "background-color: #FFEB9C"
            elif "❌" in str(v): return "background-color: #FFC7CE"
            return ""

        show_cols = ["Input Brand",
                     "Ad Subsidiary", "Ad Brand Core", "Ad Category", "Ad Confidence",
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
