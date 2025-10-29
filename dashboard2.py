import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# Load your dataset
df = pd.read_csv('ai_model_kpi_data2.csv')
df.rename(columns={
    'Ques_ID_Prompt': 'Question_ID',
    'Answer_Rating_0_5': 'Rating',
    'Electricity_consumption_Wh': 'Electricity_Wh',
    'CO2_Emission_gm': 'CO2_g',
    'Inference_Timing_sec': 'Inference_s'
}, inplace=True)

st.set_page_config(layout="wide")
st.title("📊 AI Model KPI Dashboard")

# -------------------- 1. Radar Chart --------------------
st.subheader("🕸️ KPI Profile per Model Category")
kpi_avg = df.groupby('Model_Category')[['Rating', 'Electricity_Wh', 'CO2_g', 'Inference_s']].mean()
kpi_norm = (kpi_avg - kpi_avg.min()) / (kpi_avg.max() - kpi_avg.min())
labels = kpi_norm.columns.tolist()
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig1, ax1 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for idx, row in kpi_norm.iterrows():
    values = row.tolist() + row.tolist()[:1]
    ax1.plot(angles, values, label=idx)
    ax1.fill(angles, values, alpha=0.1)
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(labels)
ax1.set_title('KPI Profile per Model Category')
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
st.pyplot(fig1)

# -------------------- 2. Grouped Bar Chart --------------------
st.subheader("⚡ Electricity & Carbon by Question Category")
agg = df.groupby('Question_Category')[['Electricity_Wh', 'CO2_g']].mean().reset_index()
agg_melted = agg.melt(id_vars='Question_Category', var_name='Metric', value_name='Average')
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(data=agg_melted, x='Question_Category', y='Average', hue='Metric', ax=ax2, palette='coolwarm')
ax2.set_title('Average Electricity and Carbon Emission by Question Category')
st.pyplot(fig2)

# -------------------- 3. Heatmap --------------------
st.subheader("🌐 Rating Heatmap by Model vs Question Category")
pivot = df.pivot_table(index='Model_Category', columns='Question_Category', values='Rating', aggfunc='mean')
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax3)
ax3.set_title('Average Rating by Model and Question Category')
st.pyplot(fig3)

# -------------------- 4. Quality vs Energy --------------------
st.subheader("📈 Quality vs Energy")
fig4 = px.scatter(
    df,
    x='Electricity_Wh',
    y='Rating',
    color='Model_Category',
    symbol='Question_Category',
    hover_data=['Model', 'CO2_g', 'Inference_s'],
    title='Quality vs Energy by Model and Question Category'
)
st.plotly_chart(fig4, use_container_width=True)

# -------------------- 5. Quality vs Cost --------------------
st.subheader("📈 Quality vs Carbon Emission (Cost Proxy)")
fig5 = px.scatter(
    df,
    x='CO2_g',
    y='Rating',
    color='Model_Category',
    symbol='Question_Category',
    hover_data=['Model', 'Electricity_Wh', 'Inference_s'],
    title='Quality vs Carbon Emission by Category'
)
st.plotly_chart(fig5, use_container_width=True)

# -------------------- 6. Latency Box Plot --------------------
st.subheader("⏱️ Latency Distribution by Model Category")
fig6 = px.box(
    df,
    x='Model_Category',
    y='Inference_s',
    color='Model_Category',
    title='Latency Distribution per Model Category'
)
st.plotly_chart(fig6, use_container_width=True)