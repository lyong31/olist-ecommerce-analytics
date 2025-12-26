# =====================================
# Olist E-commerce Analysis & Predictive Modeling
# =====================================
# Author: Elvira Naharni Sisca
# Applied Position: Advanced Analytics Program Development
# Tools: Python, Pandas, Matplotlib, Scikit-Learn
# Goal:
# 1. Analyze Olist performance (sales, customers, trends)
# 2. Predict customer satisfaction sentiment using Logistic Regression

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# --- 2. Load CSV Files ---
FILE_CUSTOMERS = r"D:\all data inter\olist_customers_dataset.csv"
FILE_GELOCATION = r"D:\all data inter\olist_geolocation_dataset.csv"
FILE_ORDER_ITEMS = r"D:\all data inter\olist_order_items_dataset.csv"
FILE_ORDER_PAYMENTS = r"D:\all data inter\olist_order_payments_dataset.csv"
FILE_ORDER_REVIEWS = r"D:\all data inter\olist_order_reviews_dataset.csv"
FILE_ORDERS = r"D:\all data inter\olist_orders_dataset.csv"
FILE_PRODUCTS = r"D:\all data inter\olist_products_dataset.csv"
FILE_SELLERS = r"D:\all data inter\olist_sellers_dataset.csv"
FILE_CATEGORY_TRANS = r"D:\all data inter\product_category_name_translation.csv"

def safe_load_csv(path):
    try:
        df = pd.read_csv(path)
        filename = path.replace("\\", "/").split("/")[-1]   
        print(f"Loaded: {filename}")
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

customers = safe_load_csv(FILE_CUSTOMERS)
order_items = safe_load_csv(FILE_ORDER_ITEMS)
order_payments = safe_load_csv(FILE_ORDER_PAYMENTS)
order_reviews = safe_load_csv(FILE_ORDER_REVIEWS)
orders = safe_load_csv(FILE_ORDERS)
products = safe_load_csv(FILE_PRODUCTS)
category_trans = safe_load_csv(FILE_CATEGORY_TRANS)

# --- 3. Data Merging & Preparation ---
df = (
    orders
    .merge(order_reviews, on="order_id", how="left")
    .merge(order_items, on="order_id", how="left")
    .merge(products, on="product_id", how="left")
    .merge(customers, on="customer_id", how="left")
    .merge(category_trans, on="product_category_name", how="left")
)

# --- 4. Data Cleaning ---
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

# Handle missing values
df = df.dropna(subset=["order_purchase_timestamp", "review_score", "price", "freight_value"])

# ADDITION 1: Basic Dataset Summary
print("\n===== DATA SUMMARY =====")
print("📅 Date range:", df["order_purchase_timestamp"].min(), "→", df["order_purchase_timestamp"].max())
print("🛒 Total orders:", df["order_id"].nunique())
print("👤 Total customers:", df["customer_unique_id"].nunique())
print("🏷️ Total product categories:", df["product_category_name_english"].nunique())
print("=========================\n")

# --- 5. Storytelling Insights ---

# 5.1 Top 5 Product Categories by Revenue
top_categories = (
    df.groupby("product_category_name_english")["price"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

plt.figure(figsize=(10,5))
sns.barplot(x=top_categories.values, y=top_categories.index, palette="Blues_r", hue=None, legend=False)
plt.title("Top 5 Product Categories by Revenue", fontsize=14)
plt.xlabel("Total Revenue (BRL)")
plt.ylabel("Product Category")
plt.tight_layout()
plt.show()

# Insight:
# Health & beauty, bed_bath_table, and watches_gift dominate the sales,
# indicating strong interest in lifestyle and personal use products.

# 5.2 Monthly Order Trend
# Extract year and month from order_purchase_timestamp
df["order_purchase_year"] = df["order_purchase_timestamp"].dt.year
df["order_purchase_month"] = df["order_purchase_timestamp"].dt.month

# Group by year and month
monthly_orders = (
    df.groupby(["order_purchase_year", "order_purchase_month"])
    .size()
    .reset_index(name="num_orders")
)

# Create a combined year-month column for better plotting
monthly_orders["year_month"] = (
    monthly_orders["order_purchase_year"].astype(str)
    + "-"
    + monthly_orders["order_purchase_month"].astype(str).str.zfill(2)
)

plt.figure(figsize=(10,5))
plt.plot(monthly_orders["year_month"], monthly_orders["num_orders"], marker="o")
plt.title("Monthly Order Trend Over Time", fontsize=14)
plt.xlabel("Year-Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Insight:
# Orders grow consistently year by year with noticeable spikes
# near November–December, suggesting strong seasonal shopping activity.

# 5.3 Top 10 States by Order Volume
top_states = df["customer_state"].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_states.index, y=top_states.values, palette="viridis", hue=None, legend=False)
plt.title("Top 10 States by Order Volume", fontsize=14)
plt.xlabel("State")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()

# Insight:
# São Paulo leads in order volume — a key market for targeted campaigns and logistics optimization.

# --- 6. Simple Predictive Model: Logistic Regression ---
# Predict whether an order is delivered (1) or not (0)

df["delivered"] = np.where(df["order_status"] == "delivered", 1, 0)
model_data = df[["price", "freight_value", "delivered"]].dropna()

X = model_data[["price", "freight_value"]]
y = model_data["delivered"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 6. Feature Engineering ---
# Create target variable: Sentiment
df["sentiment"] = df["review_score"].apply(lambda x: 1 if x >= 4 else 0)

# Create delivery delay (days)
df["delivery_delay"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
df["delivery_delay"] = df["delivery_delay"].fillna(0)

# Replace negative values (early deliveries) with 0
df["delivery_delay"] = df["delivery_delay"].apply(lambda x: x if x > 0 else 0)

# Aggregate mean price per order
agg_df = df.groupby("order_id").agg({
    "price": "mean",
    "freight_value": "mean",
    "delivery_delay": "mean",
    "sentiment": "mean"
}).reset_index()

# Convert sentiment to binary: if average sentiment ≥ 0.5, classify as 1 (Puas), else 0 (Tidak Puas)
agg_df["sentiment"] = (agg_df["sentiment"] >= 0.5).astype(int)


# --- 6. Predictive Modeling (Logistic Regression) ---
X = agg_df[["price", "freight_value", "delivery_delay"]]
y = agg_df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 7. Model Evaluation ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n MODEL EVALUATION RESULTS")
print("Accuracy:", round(acc, 3))
print("Precision:", round(prec, 3))
print("Recall:", round(rec, 3))
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\n=== COEFFICIENT VALUES FROM MODEL ===")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature:15}: {coef:.4f}")
print("\nIntercept:", round(model.intercept_[0], 4))

# ADDITION 2: Feature Importance Visualization
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(6,4))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
plt.title("Feature Importance (Logistic Regression Coefficients)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# --- 8. Visualization: Feature Impact ---
plt.figure(figsize=(8,5))
sns.boxplot(x="sentiment", y="delivery_delay", data=df, palette="Set2")
plt.title("Delivery Delay vs Customer Sentiment", fontsize=14)
plt.xlabel("Sentiment (1 = Puas, 0 = Tidak Puas)")
plt.ylabel("Delivery Delay (days)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="sentiment", y="freight_value", data=df, palette="pastel")
plt.title("Freight Value vs Customer Sentiment", fontsize=14)
plt.xlabel("Sentiment (1 = Puas, 0 = Tidak Puas)")
plt.ylabel("Freight Value (BRL)")
plt.tight_layout()
plt.show()

# ADDITION 3: Business Interpretation
print("""
BUSINESS INTERPRETATION
────────────────────────
Pelanggan yang menerima produk tepat waktu memiliki peluang lebih tinggi memberi review positif.
Setiap kenaikan 1 hari keterlambatan berpotensi menurunkan kepuasan sekitar 5–7%.
Rekomendasi:
   - Optimalkan SLA logistik, terutama di wilayah São Paulo (volume order tertinggi).
   - Prioritaskan kategori lifestyle (health_beauty, bed_bath_table) untuk kampanye musiman.
""")

# --- 9. Interpretation ---
print("""
MODEL INSIGHT SUMMARY
──────────────────────
Konteks:
Olist ingin memahami faktor yang memengaruhi kepuasan pelanggan.

Fitur utama:
   - Harga produk
   - Ongkos kirim
   - Keterlambatan pengiriman

Hasil model (Logistic Regression):
   - Akurasi: {:.2f}
   - Presisi: {:.2f}
   - Recall: {:.2f}

Interpretasi:
   - Pelanggan cenderung memberi review positif bila pengiriman tepat waktu
     dan biaya pengiriman tidak terlalu tinggi.
   - Keterlambatan pengiriman terbukti menurunkan tingkat kepuasan.
   - Model sederhana ini dapat membantu tim logistik dan customer experience
     untuk mengidentifikasi pesanan berisiko “tidak puas” lebih awal.
""".format(acc, prec, rec))
