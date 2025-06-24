import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    df.drop(['Time'], axis=1, inplace=True)
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    return df

df = load_data()

st.title("Fraud Dataset Visualizations")

st.warning(
    "This section breaks down the credit card transaction data to show how" \
    " fraud differs from regular activity. The charts highlight patterns that played a " \
    "role in training the model and shaping the way it makes predictions."
)

st.write("")  # adds a blank line
st.markdown("---")
st.write("")  # adds a blank line

#----------------------------------------------------------------------------------------------------

st.subheader("Class Distribution")

sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))

# Draw the count plot
axes = sns.countplot(data=df, x='Class', palette='pastel')

# Set titles and labels
plt.title('Class Distribution (0 = Legit, 1 = Fraud)')
plt.xlabel('Transaction Class')
plt.ylabel('Count')

# Add count labels on top of each bar
for p in axes.patches:
    count = int(p.get_height())
    axes.annotate(f'{count:,}',            
                (p.get_x() + p.get_width() / 2., p.get_height()), # (x,y) coordinate of where the annotation goes
                ha='center', va='bottom',
                fontsize=10, color='black')

st.pyplot(plt.gcf())
plt.clf()
st.markdown("**Figure 1: Class Distribution in the Dataset.**")
st.info("There are far more legitimate transactions than fraudulent ones in this dataset." \
" This uneven class distribution affects how the model learns during training." \
" If we don’t account for the imbalance, the model may focus too heavily on the common transactions"
" and overlook the rare fraud cases. Because fraud is uncommon, we need to take extra steps to help" \
" the model recognize those patterns. The chart above makes this imbalance easy to see and shows why" \
" it’s important to use methods like resampling or evaluation metrics that don’t just reward overall accuracy.")

st.write("")  # adds a blank line
st.markdown("---")
st.write("")  # adds a blank line


#----------------------------------------------------------------------------------------------------

st.subheader("Feature Correlation with Fraud")
filtered_df = df.drop(columns=["Amount", "Amount_scaled"])
correlation_matrix = filtered_df.corr()
correlation_with_class = correlation_matrix["Class"].drop("Class")

# Convert to DataFrame for plotting
corr_df = correlation_with_class.sort_values().to_frame().reset_index()
corr_df.columns = ['Feature', 'Correlation']

# Plot correlation bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=corr_df, x='Correlation', y='Feature', palette='coolwarm')
plt.title("Correlation of Features with Fraud (Class = 1)")
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

st.markdown("**Figure 2: Feature Correlation with Fraud.**")
st.info(
    "This bar chart shows how strongly each feature in the dataset is correlated with fraudulent transactions. "
    "Features such as V10, V12, V14, and V17 have the strongest negative correlation with the fraud label (Class = 1), "
    "indicating that lower values in these features are more common in fraudulent behavior. "
    "This insight guided feature selection and model training by highlighting the most relevant variables for detecting fraud."
)

st.write("")  # adds a blank line
st.markdown("---")
st.write("")  # adds a blank line


#----------------------------------------------------------------------------------------------------

# Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10,8))
filtered_df = df.drop(columns=["Amount_scaled"])
heatmap_df = filtered_df.corr()
sns.heatmap(heatmap_df, cmap='coolwarm', center=0, linewidths=0.5, linecolor='gray')
st.pyplot(plt.gcf())
plt.clf()
st.markdown("**Figure 3: Correlation Heatmap of All Features in the Fraud Dataset**")
st.info("This heatmap shows how different features in the dataset relate to each other." \
" The column labeled “Class,” which marks whether a transaction is fraud or not, has the" \
" strongest negative relationship with features V10, V12, V14, and V17. These specific" \
" features were treated as more important during model training because they had clearer" \
" patterns connected to fraud.")

st.write("")  # adds a blank line
st.markdown("---")
st.write("")  # adds a blank line


#----------------------------------------------------------------------------------------------------

st.subheader("Boxplot of Key Features.")

selected = ['V10', 'V12', 'V14', 'V17', 'Class']
df_melted = pd.melt(df[selected], id_vars="Class", var_name="Feature", value_name="Value")

# Plot boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_melted, x="Feature", y="Value", hue="Class", palette="Set2")
plt.title("Boxplot of Key Features by Transaction Class")
plt.legend(title="Class", labels=["Legit (0)", "Fraud (1)"])
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

st.markdown("**Figure 4: Boxplot of Key Features.**")
st.info(
    "This boxplot compares the distributions of features V10, V12, V14, and V17 between legitimate and fraudulent transactions. "
    "The differences in these distributions help explain why these features were useful for model training as they show distinct patterns in fraud cases."
)
st.write("")  # adds a blank line
st.markdown("---")
st.write("")  # adds a blank line

#----------------------------------------------------------------------------------------------------

st.subheader("Distribution of Original Transaction Amounts")
plt.figure(figsize=(8, 5))
sns.histplot(df["Amount"], bins=50, kde=True, color='teal')
plt.title("Histogram of Original Transaction Amounts")
plt.xlabel("Amount ($)")
plt.ylabel("Frequency")
st.pyplot(plt.gcf())
plt.clf()

st.markdown("**Figure 5: Histogram of Original Transaction Amounts**")
st.info(
    "This histogram shows the original distribution of transaction amounts. "
    "Most transactions fall under lower dollar values, with a long tail of higher-value outliers. "
    "These outliers could disproportionately affect certain models if left unscaled."
)
st.write("")  # adds a blank line
st.markdown("---")
st.write("")  # adds a blank line

# ----------------------------------------------------------------------------------------------------

st.subheader("Distribution of Scaled Transaction Amounts")
plt.figure(figsize=(8, 5))
sns.histplot(df["Amount_scaled"], bins=50, kde=True, color='skyblue')
plt.title("Histogram of Scaled Transaction Amounts")
plt.xlabel("Amount (Standardized)")
plt.ylabel("Frequency")
st.pyplot(plt.gcf())
plt.clf()

st.markdown("**Figure 6: Histogram of Scaled Transaction Amounts**")
st.info(
    "This plot shows the standardized version of the Amount feature. "
    "Scaling centers the data around zero with a consistent spread, allowing it to be more fairly weighed during model training. "
    "This transformation is especially helpful when combining features of different units or ranges."
)
st.write("")  # adds a blank line
st.markdown("---")
st.write("")  # adds a blank line

#----------------------------------------------------------------------------------------------------


st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.9em; padding: 10px 0;">
         <em>Developed by <strong>Shane Matthews</strong>
    </div>
    """,
    unsafe_allow_html=True
)

