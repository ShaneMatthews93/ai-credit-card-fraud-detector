import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib 

# Load the trained AI model
model = joblib.load("models/fraud_model.pkl")

# Load CSV file
@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df

df = load_data()

st.title("Credit Card Fraud Detection")

st.warning("Use the sidebar to naviate between fraud dectection and fraud visualization.")

st.subheader("Select a Range of Transactions")
row_range = st.slider(
    "Select a range of rows to evaluate",
    min_value=0,
    max_value=len(df)-1,
    value=(0, 5000),  # default range
    step=100
)

start_row, end_row = row_range

# Filter range
subset = df.iloc[start_row:end_row] 
X = subset.drop('Class', axis=1)

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

st.markdown("---")
# Show results
st.subheader("Results")
fraud_indices = []
for i, pred in enumerate(predictions):
    if pred == 1:
        fraud_indices.append(i)

if fraud_indices:
    for i in fraud_indices:
        row_num = start_row + i
        st.error(f"⚠️ Row {row_num} predicted as FRAUD \n Confidence: {probabilities[i]:.2f}")
        st.dataframe(subset.iloc[[i]])
else:
    st.success("✅ No fraud detected in this range.")


#----------------------------------------------------------------------------------------------------


st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.9em; padding: 10px 0;">
         <em>Developed by <strong>Shane Matthews</strong>
    </div>
    """,
    unsafe_allow_html=True
)

