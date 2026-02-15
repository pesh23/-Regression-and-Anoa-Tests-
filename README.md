# -Regression-and-Anoa-Tests-
Automation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from linearmodels.panel import PanelOLS, RandomEffects, compare
import streamlit as st
import numpy as np

st.set_page_config(page_title="Interactive Netmigration Panel Dashboard", layout="wide")

# --------------------
# 🎬 Hero Section (Animated Gradient)
# --------------------
st.markdown("""
<style>
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.hero {
    padding: 60px;
    border-radius: 20px;
    text-align: center;
    color: white;
    background: linear-gradient(-45deg, #1f4037, #99f2c8, #134e5e, #71b280);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

.fade-in {
    animation: fadeIn 2s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(30px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>

<div class="hero fade-in">
    <h1>📊 Netmigration Panel Data Dashboard</h1>
    <p>Upload Excel → Run Panel Regression → ANOVA → Visual Insights Instantly</p>
</div>
""", unsafe_allow_html=True)

# --------------------
# 1. FILE UPLOAD
# --------------------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Remove completely empty rows
    df = df.dropna(how="all")

    # --------------------
    # 2. SELECT VARIABLES
    # --------------------
    all_columns = df.columns.tolist()

    entity_col = st.selectbox("Select Entity (Panel Unit)", options=all_columns)
    time_col = st.selectbox("Select Time Column", options=all_columns)

    dependent_var = st.selectbox("Select Dependent Variable", options=all_columns)
    independent_vars = st.multiselect(
        "Select Independent Variables",
        options=[c for c in all_columns if c != dependent_var]
    )

    if st.button("Generate Analysis"):

        try:
            # --------------------
            # DATA CLEANING
            # --------------------
            df = df.dropna(subset=[entity_col, time_col, dependent_var] + independent_vars)

            # Check panel size
            if df[entity_col].nunique() < 2 or df[time_col].nunique() < 2:
                st.error("Need at least 2 entities and 2 time periods.")
                st.stop()

            # Check constant variables
            for col in independent_vars + [dependent_var]:
                if df[col].nunique() <= 1:
                    st.error(f"Variable '{col}' has no variation (constant value). Remove it.")
                    st.stop()

            # --------------------
            # 3. PANEL SETUP
            # --------------------
            df_panel = df.set_index([entity_col, time_col]).sort_index()

            Y = df_panel[dependent_var]
            X = df_panel[independent_vars]
            X = sm.add_constant(X)

            # --------------------
            # 4. PANEL REGRESSION (TABLE OUTPUT)
            # --------------------
            st.subheader("📈 Panel Regression Results")

            # Fixed Effects
            fe_model = PanelOLS(Y, X, entity_effects=True).fit()
            fe_results = pd.DataFrame({
                "Coefficient": fe_model.params,
                "Std Error": fe_model.std_errors,
                "t-Statistic": fe_model.tstats,
                "P-Value": fe_model.pvalues
            })
            st.write("### Fixed Effects Results")
            st.dataframe(fe_results)

            # Pooled OLS
            pooled_model = PanelOLS(Y, X).fit()
            pooled_results = pd.DataFrame({
                "Coefficient": pooled_model.params,
                "Std Error": pooled_model.std_errors,
                "t-Statistic": pooled_model.tstats,
                "P-Value": pooled_model.pvalues
            })
            st.write("### Pooled OLS Results")
            st.dataframe(pooled_results)

            # Random Effects
            re_model = RandomEffects(Y, X).fit()
            re_results = pd.DataFrame({
                "Coefficient": re_model.params,
                "Std Error": re_model.std_errors,
                "t-Statistic": re_model.tstats,
                "P-Value": re_model.pvalues
            })
            st.write("### Random Effects Results")
            st.dataframe(re_results)

            # Model Comparison Table
            comparison_df = pd.DataFrame({
                "Model": ["Pooled OLS", "Fixed Effects", "Random Effects"],
                "R-Squared": [
                    pooled_model.rsquared,
                    fe_model.rsquared,
                    re_model.rsquared
                ],
                "Observations": [
                    pooled_model.nobs,
                    fe_model.nobs,
                    re_model.nobs
                ]
            })
            st.write("### Model Comparison")
            st.dataframe(comparison_df)

            # --------------------
            # 5. ANOVA (SAFE)
            # --------------------
            st.subheader("📊 ANOVA Results")

            df_anova = df.copy()
            formula = f'Q("{dependent_var}") ~ C(Q("{entity_col}"))'
            anova_model = ols(formula, data=df_anova).fit()
            anova_results = anova_lm(anova_model)
            st.dataframe(anova_results)

            # --------------------
            # 6. VISUALIZATIONS
            # --------------------
            st.subheader("📉 Visualizations")

            # Line chart
            st.write(f"### Average {dependent_var} Over Time")
            fig1 = plt.figure()
            df.groupby(time_col)[dependent_var].mean().plot()
            plt.xlabel(time_col)
            plt.ylabel(dependent_var)
            st.pyplot(fig1)

            # Regression plots
            for var in independent_vars:
                st.write(f"### {dependent_var} vs {var}")
                fig2 = plt.figure()
                sns.regplot(x=df[var], y=df[dependent_var])
                st.pyplot(fig2)

            # Bar chart
            st.write(f"### Mean {dependent_var} by {entity_col}")
            fig3 = plt.figure(figsize=(10,5))
            df.groupby(entity_col)[dependent_var].mean().plot(kind="bar")
            st.pyplot(fig3)

            # Correlation Heatmap
            st.write("### Correlation Matrix")
            fig4 = plt.figure()
            corr = df[independent_vars + [dependent_var]].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            st.pyplot(fig4)

        except Exception as e:
            st.error(f"Error: {e}")
