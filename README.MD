# AI Credit Card Fraud Detector

This project uses machine learning to detect fraudulent credit card transactions. It leverages a real-world dataset and includes data visualization, model training, evaluation, and a user-friendly Streamlit web app for interactive exploration.

---

## 🚀 Project Features

- Trains a **Random Forest classifier** for binary fraud detection
- Visualizes class distribution, feature correlation, and model performance
- Allows predictions on the original Kaggle dataset (custom uploads must match the same format and transformed features)
- Built with **Python**, **Scikit-learn**, **Matplotlib**, **Seaborn**, and **Streamlit**

---

## 📂 Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle:

🔗 [Download Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> ⚠️ **Important:** This dataset uses PCA-transformed numerical features to protect sensitive information.  
> The model and app will only work with data in this exact format. Custom transaction data **must match the same structure and features** to produce valid results.

After downloading, place the file in:
```
data/creditcard.csv
```

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/ShaneMatthews93/ai-credit-card-fraud-detector.git
cd ai-credit-card-fraud-detector
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📊 Visualizations

- Class imbalance chart
- Feature correlation heatmap
- Boxplots of top fraud-related features
- Model performance metrics (precision, recall, F1-score)

---

## 📁 Project Structure

```
.
├── app.py                    # Main Streamlit app
├── data/                     # Folder for the dataset (not tracked by Git)
├── models/                   # Trained model files (e.g., RandomForest)
├── pages/                    # Additional Streamlit pages (e.g., visuals)
├── notebooks/                # Jupyter notebooks for exploration and training
├── requirements.txt
└── .gitignore
```

---

## 📌 Notes

- Ensure your `scikit-learn` version matches the one used to save the model (or retrain locally).
- The app and models are for educational purposes and not intended for production use without further validation and security considerations.

---

## 📬 Contact

**Author:** Shane Matthews  
Connect on [LinkedIn](https://www.linkedin.com/in/shanematthewscs/)  
Questions? Open an issue on this repo.

---
