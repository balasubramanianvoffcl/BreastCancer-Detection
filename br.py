import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("Breast_Cancer.csv")

# Encode categorical features
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Features and target
X = df.drop('Status', axis=1)
y = df['Status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))

# Streamlit UI
st.title("🧠 Breast Cancer Risk Prediction")
st.markdown(f"**Model Accuracy:** {acc:.2%}")
st.subheader("📊 Dataset Insights")

# 1. Alive vs Dead count
status_counts = df['Status'].value_counts()
st.write("### Status Distribution")
st.bar_chart(status_counts)

# 2. Boxplot of Age by Status
st.write("### Age Distribution by Survival Status")
fig1, ax1 = plt.subplots()
sns.boxplot(x='Status', y='Age', data=df, ax=ax1)
st.pyplot(fig1)

# 3. Tumor Size Histogram
st.write("### Tumor Size Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['Tumor Size'], kde=True, bins=30, ax=ax2)
st.pyplot(fig2)

# 4. Optional: Correlation heatmap
st.write("### Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
st.pyplot(fig3)

# User input form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    tumor_size = st.number_input("Tumor Size", min_value=1, value=20)
    node_examined = st.number_input("Regional Node Examined", min_value=0, value=10)
    node_positive = st.number_input("Regional Node Positive", min_value=0, value=1)
    
    estrogen_status = st.selectbox("Estrogen Status", ["Positive", "Negative"])
    progesterone_status = st.selectbox("Progesterone Status", ["Positive", "Negative"])
    
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    # Manual inputs for a minimal set of features (expandable)
    input_df = pd.DataFrame({
        'Age': [age],
        'Tumor Size': [tumor_size],
        'Regional Node Examined': [node_examined],
        'Reginol Node Positive': [node_positive],
        'Estrogen Status': [label_encoders['Estrogen Status'].transform([estrogen_status])[0]],
        'Progesterone Status': [label_encoders['Progesterone Status'].transform([progesterone_status])[0]]

    })

    # Align input_df with model features
    input_full = pd.DataFrame(columns=X.columns)
    for col in input_full.columns:
        if col in input_df.columns:
            input_full[col] = input_df[col]
        else:
            input_full[col] = 0  # default

    prediction = model.predict(input_full)[0]
    result = "🟢 Alive" if prediction == 0 else "🔴 Dead"
    st.success(f"Prediction Result: **{result}**")
