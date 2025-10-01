"""
Disease-Drug Recommendation System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from difflib import get_close_matches

# Page configuration
st.set_page_config(
    page_title="Drug Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model


@st.cache_data
def load_model():
    try:
        with open('drug_recommendation_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found. Please run preprocessing.py first!")
        return None

# Fuzzy match disease names


def find_matching_disease(user_input, disease_list, threshold=0.6):
    """Find diseases that match the user input using fuzzy matching"""
    user_input = user_input.lower().strip()

    # Exact match
    exact_matches = [d for d in disease_list if user_input == d.lower()]
    if exact_matches:
        return exact_matches

    # Partial match (contains)
    partial_matches = [
        d for d in disease_list if user_input in d.lower() or d.lower() in user_input]
    if partial_matches:
        return partial_matches

    # Fuzzy match
    fuzzy_matches = get_close_matches(
        user_input, [d.lower() for d in disease_list], n=5, cutoff=threshold)
    if fuzzy_matches:
        return [d for d in disease_list if d.lower() in fuzzy_matches]

    return []

# Get drug recommendations


def get_recommendations(disease_name, rules_df, top_n=10):
    disease_rules = rules_df[rules_df['disease'].str.lower()
                             == disease_name.lower()]

    if disease_rules.empty:
        return None

    disease_rules = disease_rules.sort_values(
        ['confidence', 'support'], ascending=False)
    return disease_rules.head(top_n)

# Main app


def main():
    # Header
    st.markdown('<p class="main-header">💊 Drug Recommendation System</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Load model
    model_data = load_model()

    if model_data is None:
        st.stop()

    rules_df = model_data['rules']
    disease_list = model_data['diseases']
    stats = model_data['stats']

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
        st.title("📊 System Info")

        st.metric("Total Rules", f"{stats['total_rules']:,}")
        st.metric("Diseases Covered", stats['unique_diseases'])
        st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
        st.metric("Avg Support", f"{stats['avg_support']:.3f}")

        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This system uses **Association Rule Mining** to recommend drugs based on "
            "disease patterns found in medical data."
        )

        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Enter your disease or symptoms
        2. Click 'Get Recommendations'
        3. View recommended drugs with confidence scores
        4. Explore visualizations
        """)

        st.markdown("---")
        st.warning(
            "⚠️ **Disclaimer**: This is for educational purposes only. Always consult healthcare professionals.")

    # Main content
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Get Recommendations", "📈 Analytics", "📋 All Rules"])

    with tab1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Enter Your Disease or Symptoms</p>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_input(
                "Type the disease name or symptoms:",
                placeholder="e.g., diabetes, high blood pressure, flu, headache...",
                help="Enter the name of the disease or describe your symptoms",
                key="disease_input"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.button(
                "🔍 Get Recommendations", type="primary", use_container_width=True)

        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=20,
            value=5,
            help="Select how many drug recommendations you want to see"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        if submit_button and user_input:
            with st.spinner("🔍 Searching for matching diseases..."):
                matching_diseases = find_matching_disease(
                    user_input, disease_list)

            if matching_diseases:
                if len(matching_diseases) == 1:
                    selected_disease = matching_diseases[0]
                    st.success(f"✅ Found exact match: **{selected_disease}**")
                else:
                    st.info(
                        f"📋 Found {len(matching_diseases)} matching diseases. Showing recommendations for all matches.")
                    selected_disease = st.selectbox(
                        "Select the disease you meant:",
                        options=matching_diseases,
                        help="Multiple diseases matched your input. Please select the correct one."
                    )

                # Get recommendations
                recommendations = get_recommendations(
                    selected_disease, rules_df, num_recommendations)

                if recommendations is not None and len(recommendations) > 0:
                    st.success(
                        f"✅ Found {len(recommendations)} recommendations for **{selected_disease}**")

                    # Display recommendations
                    st.markdown(
                        '<p class="sub-header">Recommended Drugs</p>', unsafe_allow_html=True)

                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            col_drug, col_metrics = st.columns([2, 3])

                            with col_drug:
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h3>#{idx}: {row['drug']}</h3>
                                </div>
                                """, unsafe_allow_html=True)

                            with col_metrics:
                                metric_col1, metric_col2, metric_col3 = st.columns(
                                    3)

                                with metric_col1:
                                    st.metric("Confidence",
                                              f"{row['confidence']:.1%}")
                                with metric_col2:
                                    st.metric(
                                        "Support", f"{row['support']:.3f}")
                                with metric_col3:
                                    st.metric("Lift", f"{row['lift']:.2f}")

                    # Visualization
                    st.markdown(
                        '<p class="sub-header">Recommendation Metrics</p>', unsafe_allow_html=True)

                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Confidence Scores',
                                        'Support vs Confidence'),
                        specs=[[{"type": "bar"}, {"type": "scatter"}]]
                    )

                    # Bar chart
                    fig.add_trace(
                        go.Bar(
                            x=recommendations['drug'],
                            y=recommendations['confidence'],
                            name='Confidence',
                            marker_color='lightblue'
                        ),
                        row=1, col=1
                    )

                    # Scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=recommendations['support'],
                            y=recommendations['confidence'],
                            mode='markers',
                            marker=dict(
                                size=recommendations['lift']*10,
                                color=recommendations['lift'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Lift")
                            ),
                            text=recommendations['drug'],
                            hovertemplate='<b>%{text}</b><br>Support: %{x:.3f}<br>Confidence: %{y:.1%}<extra></extra>'
                        ),
                        row=1, col=2
                    )

                    fig.update_xaxes(title_text="Drug", row=1, col=1)
                    fig.update_yaxes(title_text="Confidence", row=1, col=1)
                    fig.update_xaxes(title_text="Support", row=1, col=2)
                    fig.update_yaxes(title_text="Confidence", row=1, col=2)

                    fig.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Download option
                    csv = recommendations[[
                        'disease', 'drug', 'support', 'confidence', 'lift']].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Recommendations (CSV)",
                        data=csv,
                        file_name=f"recommendations_{selected_disease}.csv",
                        mime="text/csv"
                    )

                else:
                    st.warning(
                        f"⚠️ No recommendations found for **{selected_disease}**")
            else:
                st.error(f"❌ No matching disease found for: **{user_input}**")
                st.info(
                    "💡 **Suggestions:**\n- Check your spelling\n- Try using more general terms\n- Use medical terminology if possible")

                # Show some example diseases
                with st.expander("📋 View some available diseases"):
                    sample_diseases = disease_list[:20]
                    st.write(", ".join(sample_diseases))
                    if len(disease_list) > 20:
                        st.write(
                            f"...and {len(disease_list) - 20} more diseases")

        elif submit_button and not user_input:
            st.warning("⚠️ Please enter a disease or symptom first!")
        else:
            st.info(
                "👆 Enter a disease or symptom and click 'Get Recommendations' to start")

    with tab2:
        st.markdown('<p class="sub-header">System Analytics</p>',
                    unsafe_allow_html=True)

        # Top diseases and drugs
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top 10 Diseases")
            top_diseases = rules_df['disease'].value_counts().head(10)
            fig = px.bar(
                x=top_diseases.values,
                y=top_diseases.index,
                orientation='h',
                labels={'x': 'Number of Rules', 'y': 'Disease'},
                color=top_diseases.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Top 10 Drugs")
            top_drugs = rules_df['drug'].value_counts().head(10)
            fig = px.bar(
                x=top_drugs.values,
                y=top_drugs.index,
                orientation='h',
                labels={'x': 'Number of Rules', 'y': 'Drug'},
                color=top_drugs.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Distribution plots
        st.markdown("#### Rule Metrics Distribution")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig = px.histogram(rules_df, x='confidence',
                               nbins=30, title='Confidence Distribution')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(rules_df, x='support', nbins=30,
                               title='Support Distribution')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = px.histogram(rules_df, x='lift', nbins=30,
                               title='Lift Distribution')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter plot
        st.markdown("#### Confidence vs Support (colored by Lift)")
        fig = px.scatter(
            rules_df.head(500),
            x='support',
            y='confidence',
            color='lift',
            hover_data=['disease', 'drug'],
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<p class="sub-header">All Association Rules</p>',
                    unsafe_allow_html=True)

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.05)
        with col2:
            min_supp = st.slider("Minimum Support", 0.0, 0.5, 0.0, 0.01)
        with col3:
            min_lift = st.slider("Minimum Lift", 0.0, 10.0, 0.0, 0.5)

        # Filter rules
        filtered_rules = rules_df[
            (rules_df['confidence'] >= min_conf) &
            (rules_df['support'] >= min_supp) &
            (rules_df['lift'] >= min_lift)
        ]

        st.info(f"Showing {len(filtered_rules)} out of {len(rules_df)} rules")

        # Display table
        display_rules = filtered_rules[[
            'disease', 'drug', 'support', 'confidence', 'lift']].copy()
        display_rules['confidence'] = display_rules['confidence'].apply(
            lambda x: f"{x:.1%}")
        display_rules['support'] = display_rules['support'].apply(
            lambda x: f"{x:.3f}")
        display_rules['lift'] = display_rules['lift'].apply(
            lambda x: f"{x:.2f}")

        st.dataframe(
            display_rules,
            use_container_width=True,
            height=600
        )

        # Download all rules
        csv = filtered_rules.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Rules (CSV)",
            data=csv,
            file_name="filtered_association_rules.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Drug Recommendation System | Powered by Association Rule Mining | "
        "⚠️ For educational purposes only - Always consult healthcare professionals"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
