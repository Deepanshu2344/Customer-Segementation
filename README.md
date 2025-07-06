# 🛍️ Customer Segmentation and Prediction

🔗 **Try the Live App:** [Streamlit App Link](https://customer-segmentation-1.streamlit.app/)

This project focuses on analyzing customer purchase behavior using transaction data from an online retailer. The goal is to segment customers based on their purchasing patterns and predict their future spending using machine learning techniques.

---

## 📌 Objectives

- Analyze customer behavior using **RFM analysis** (Recency, Frequency, Monetary).
- Segment customers into distinct groups using **K-Means Clustering**.
- Predict **future monetary value** of customers using **Linear Regression**.
- Provide actionable marketing insights based on customer segments.
- Deploy the solution as a **Streamlit Web App**.

---

## 📁 Dataset

- **File:** `OnlineRetail.csv`  
- **Source:** Real-world transactions from an e-commerce company.  
- **Key Features:**
  - `InvoiceNo`, `StockCode`, `Description`
  - `Quantity`, `InvoiceDate`, `UnitPrice`
  - `CustomerID`, `Country`

---

## 🧰 Technologies Used

| Task                    | Tool/Library            |
|-------------------------|-------------------------|
| Data Handling           | `pandas`, `numpy`       |
| Visualization           | `matplotlib`, `seaborn` |
| Clustering              | `KMeans` (from `sklearn`) |
| Prediction              | `LinearRegression`      |
| Web App                 | `Streamlit`             |
| Model Evaluation        | `mean_squared_error`    |

---

## 🔍 Project Workflow

### 1. **Data Cleaning**
- Removed missing `CustomerID` entries.
- Filtered out returns (negative `Quantity`/`Price`).
- Converted `InvoiceDate` to datetime.

### 2. **Feature Engineering**
- Created `TotalPrice = Quantity × UnitPrice`
- Generated RFM metrics per customer:
  - **Recency** – Days since last purchase
  - **Frequency** – Number of purchases
  - **Monetary** – Total amount spent

### 3. **Customer Segmentation**
- Scaled RFM data using `StandardScaler`.
- Applied **KMeans clustering (K=4)** to segment customers.
- Visualized segments using scatter plots.

### 4. **Predictive Modeling**
- Used **Linear Regression** to predict customer `Monetary` value from `Recency` and `Frequency`.
- Evaluated using **RMSE**.

### 5. **Streamlit Web App**
- Built an interactive app to:
  - Upload zipped CSV
  - View customer segments
  - Run predictions
  - Display summary and insights

---

## 💡 Key Insights

| Segment | Behavior Description       | Strategy Suggestion                   |
|---------|----------------------------|----------------------------------------|
| Cluster 2 | High-frequency, high-value, recent | VIP offers, loyalty programs          |
| Cluster 3 | Frequent and moderate-value | Upsell/cross-sell, seasonal offers    |
| Cluster 0 | Occasional buyers         | Personalized promotions, email nudges |
| Cluster 1 | At-risk or churned        | Win-back campaigns, surveys            |

---

## 🚀 Getting Started
### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### 🔧 Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
