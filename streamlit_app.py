import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
import zipfile

# Streamlit page settings
st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("🛍️ Customer Segmentation and Prediction App")

# Upload data
# Upload a zipped CSV
uploaded_file = st.file_uploader("📁 Upload a ZIP file containing OnlineRetail.csv", type="zip")

if uploaded_file is not None:
    with zipfile.ZipFile(uploaded_file) as zip_ref:
        file_list = zip_ref.namelist()
        if len(file_list) > 0:
            with zip_ref.open(file_list[0]) as file:
                df = pd.read_csv(file, encoding='ISO-8859-1')
                st.success(f"✅ Loaded file: {file_list[0]}")
                st.write("Preview of data:")
                st.dataframe(df.head())


    # Data cleaning
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df.drop_duplicates(inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    st.subheader("📊 Sample Cleaned Data")
    st.dataframe(df.head())

    # Feature Engineering - RFM
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.subheader("📈 RFM Cluster Summary")
    st.dataframe(rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Cluster': 'count'
    }).round(2))

    # Plotting
    st.subheader("🧩 Pairplot of Clusters")
    fig, ax = plt.subplots()
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='tab10', ax=ax)
    st.pyplot(fig)

    # Prediction
    X = rfm.drop(columns=['Monetary', 'Cluster'])
    y = rfm['Monetary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("📌 Predictive Model: Linear Regression")
    st.write(f"**RMSE on test set**: ₹{rmse:.2f}")

    # Show predictions (optional)
    prediction_sample = pd.DataFrame({
        'Actual': y_test.values[:10],
        'Predicted': y_pred[:10]
    })
    st.write("Sample Predictions:")
    st.dataframe(prediction_sample.round(2))
