import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Drug Recommendation (Indication-based)", page_icon="💊", layout="wide")


@st.cache_data
def load_medicine_df(path="medicine_dataset.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(
            f"❌ File not found: {path}. Please place medicine_dataset.csv in the app folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None


def find_column(df, candidates):
    lookup = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def get_display_value(row, colname):
    if colname and colname in row and pd.notna(row[colname]):
        return row[colname]
    return ""

# --- Age Classification Helpers ---


def classify_age(age: int) -> str:
    if age < 12:
        return "Child"
    elif 12 <= age < 18:
        return "Young"
    else:
        return "Adult"


def infer_age_group(text: str) -> str:
    """Infer suitable age group from indication/content text"""
    text = str(text).lower()
    if any(kw in text for kw in ["child", "children", "pediatric"]):
        return "Child"
    elif any(kw in text for kw in ["teen", "adolescent", "youth", "young"]):
        return "Young"
    elif any(kw in text for kw in ["adult", "18+"]):
        return "Adult"
    else:
        return "Adult"  # default if not specified


def main():
    st.markdown("<h1 style='text-align:center; color:#1f77b4;'>💊 Drug Recommendation — Indication Dropdown</h1>",
                unsafe_allow_html=True)
    st.markdown("---")

    df = load_medicine_df("medicine_dataset.csv")
    if df is None:
        st.stop()

    # detect columns
    indication_col = find_column(
        df, ["Indication", "indications", "Indications", "indication"])
    name_col = find_column(
        df, ["Name", "name", "med_name", "drug", "drug_name"])
    generic_col = find_column(df, ["Generic", "generic_name", "generic"])
    manufacturer_col = find_column(df, ["Manufacturer", "manufacturer"])
    price_col = find_column(df, ["Price", "final_price", "MRP", "mrp"])
    classification_col = find_column(df, ["Classification", "classification"])
    dosage_col = find_column(df, ["Dosage Form", "dosage"])
    strength_col = find_column(df, ["Strength", "strength"])
    category_col = find_column(df, ["Category", "category"])
    content_col = find_column(
        df, ["drug_content", "Content", "content", "description"])
    img_col = find_column(df, ["img_urls", "image"])

    # 🔹 Add inferred Age Group column
    if indication_col:
        df["Age_Group"] = df[indication_col].apply(infer_age_group)
    elif content_col:
        df["Age_Group"] = df[content_col].apply(infer_age_group)
    else:
        df["Age_Group"] = "Adult"

    st.markdown("### Enter patient age")
    age = st.number_input("Enter your age:", min_value=0,
                          max_value=120, value=25, step=1)
    patient_group = classify_age(age)

    if not indication_col:
        st.error("❌ Could not find an Indication column in your CSV.")
        st.stop()

    # build indication list
    raw_indications = df[indication_col].dropna().astype(str)
    splitted = raw_indications.str.split(r",|/|;|\|").explode().str.strip()
    indications_list = sorted(splitted[splitted != ""].unique())
    indications_list = ["-- All --"] + indications_list

    st.markdown("### Select your symptom / indication")
    selected_indication = st.selectbox(
        "Pick from indications (based on 'Indication' column):", indications_list
    )

    # 🔽 Recommendation limiter (relocated here)
    limit = st.slider(
        "Limit number of recommendations to display:",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Choose how many recommendations to show below"
    )

    # Find matches
    if st.button("🔍 Find matching medicines"):
        if selected_indication == "-- All --":
            filtered = df.copy()
        else:
            mask_contains = df[indication_col].fillna("").str.contains(
                selected_indication, case=False, na=False
            )
            token_mask = df[indication_col].fillna("").str.split(r",|/|;|\|").apply(
                lambda toks: any(
                    selected_indication.lower() == t.strip().lower()
                    for t in toks
                )
                if isinstance(toks, list) else False
            )
            filtered = df[mask_contains | token_mask].copy()

        # 🔹 Apply age filtering
        filtered = filtered[filtered["Age_Group"] == patient_group]

        if filtered.empty:
            st.warning(
                f"⚠️ No medicines found for indication: **{selected_indication}** suitable for age group: **{patient_group}**")
        else:
            st.success(
                f"✅ Found {len(filtered)} medicine(s) for indication: **{selected_indication}** (Age: {age}, Group: {patient_group})")

            # Show metrics
            cols = st.columns(4)
            cols[0].metric("Results", f"{len(filtered):,}")
            cols[1].metric("Unique Manufacturers",
                           f"{filtered[manufacturer_col].nunique() if manufacturer_col else 'N/A'}")
            cols[2].metric(
                "Unique Categories", f"{filtered[category_col].nunique() if category_col else 'N/A'}")
            cols[3].metric(
                "Sample Strength", f"{get_display_value(filtered.iloc[0], strength_col) if strength_col else 'N/A'}")

            st.markdown("---")
            # Show limited results
            display_count = min(len(filtered), limit)
            for i, (_, row) in enumerate(filtered.head(display_count).iterrows(), start=1):
                med_name = get_display_value(row, name_col) or f"Medicine #{i}"
                generic = get_display_value(row, generic_col)
                manufacturer = get_display_value(row, manufacturer_col)
                price = get_display_value(row, price_col)
                classification = get_display_value(row, classification_col)
                dosage = get_display_value(row, dosage_col)
                strength = get_display_value(row, strength_col)
                category = get_display_value(row, category_col)
                indication_value = get_display_value(row, indication_col)
                content = get_display_value(row, content_col)
                img = get_display_value(row, img_col)
                age_group = row["Age_Group"]

                card_html = f"""
                <div style="background:#0f1720; color:#e6eef8; padding:12px; border-radius:8px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.03);">
                    <div style="display:flex; align-items:flex-start; gap:12px;">
                        <div style="flex:1;">
                            <h3 style="margin:0; color:#9be7ff;">{med_name}</h3>
                            <div style="font-size:0.95rem; color:#d7eaf6; margin-top:6px;">
                                <div><b>Generic:</b> {generic or 'N/A'}</div>
                                <div><b>Manufacturer:</b> {manufacturer or 'N/A'}</div>
                                <div><b>Category:</b> {category or 'N/A'} &nbsp; <b>Dosage:</b> {dosage or 'N/A'} &nbsp; <b>Strength:</b> {strength or 'N/A'}</div>
                                <div><b>Indication:</b> {indication_value or 'N/A'}</div>
                                <div><b>Classification:</b> {classification or 'N/A'}</div>
                                <div><b>Price:</b> {price or 'N/A'}</div>
                                <div><b>✔ Suitable for:</b> {age_group}</div>
                            </div>
                        </div>
                """

                if img and isinstance(img, str) and img.lower().startswith("http"):
                    card_html += f"""
                        <div style="width:150px; flex:0 0 150px;">
                            <img src="{img}" alt="image" style="max-width:150px; border-radius:6px;">
                        </div>
                    """

                card_html += "</div>"
                if content:
                    card_html += f"<div style='margin-top:8px; color:#cfeefc;'>{content}</div>"
                card_html += "</div>"

                st.markdown(card_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
