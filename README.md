# 🍷 Wine Quality PCA + ML  

This project demonstrates **dimensionality reduction using PCA** and **machine learning classification** on the famous [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).  

We explore the dataset, apply preprocessing, scale features, and use **Principal Component Analysis (PCA)** to visualize the data in lower dimensions. Finally, we build and evaluate ML models to predict wine quality.  

---

## 🚀 Live Demo  

👉 Try it here: [Wine Quality PCA + ML (Hugging Face Space)](https://huggingface.co/spaces/boomiikas/Wine-Quality-PCA-ML)  

---

## 📸 App Screenshot  

<img width="1907" height="307" alt="image" src="https://github.com/user-attachments/assets/bd09fb06-d2fa-4a06-bee2-dcec95d871c6" />
<img width="1917" height="679" alt="image" src="https://github.com/user-attachments/assets/3df32339-fab0-4e4f-ad70-65ebf531d168" />

---

## 📂 Project Structure  

- **Exploratory Data Analysis (EDA)** – Data inspection, summary statistics, correlations  
- **Preprocessing** – Handling missing values, scaling (StandardScaler / RobustScaler)  
- **PCA** – Dimensionality reduction for visualization  
- **ML Models** – Logistic Regression, Random Forest, etc. for quality prediction  
- **Interactive Demo** – Streamlit app hosted on Hugging Face Spaces  

---
## 📊 Exploratory Data Analysis (EDA)

To better understand the distribution of wine quality features, we performed the following visualizations:

### 1️⃣ Kernel Density Estimation (KDE) Plots  
Visualizes the probability distribution of each feature.  
*Helps to see skewness and modality of the data.*  

<img width="1488" height="789" alt="image" src="https://github.com/user-attachments/assets/e05a7294-24ee-4405-bea0-baa5c1fba343" />


---

### 2️⃣ Histograms  
Displays the frequency distribution of each variable.  
*Useful for understanding spread, central tendency, and outliers.*  

<img width="1489" height="790" alt="image" src="https://github.com/user-attachments/assets/ccf27458-4d9f-4b1e-8260-63193748d625" />


---

### 3️⃣ Boxplots  
Highlights the median, quartiles, and potential outliers in each feature.  
*Good for spotting skewed features and extreme values.*  

<img width="1489" height="790" alt="image" src="https://github.com/user-attachments/assets/8844d03b-1dcf-4740-ac00-9a1bc201b301" />


---

### 4️⃣ After `log1p` Transformation  
We applied a **log1p transformation** (`log(1+x)`) to reduce skewness and make the data more normally distributed.  

Visualizing the transformed features using KDE plots shows smoother, more symmetric distributions.  

<img width="1487" height="789" alt="image" src="https://github.com/user-attachments/assets/f0c6f023-5167-4ca9-992b-ac5240473005" />


---

📌 *These plots help guide preprocessing decisions such as scaling, normalization, and PCA application.*

---

## ⚙️ How to Run Locally  

1. Clone this repo  
   ```bash
   git clone https://huggingface.co/spaces/boomiikas/Wine-Quality-PCA-ML
   cd Wine-Quality-PCA-ML
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app  
   ```bash
   streamlit run app.py
   ```

---

## 📊 Dataset  

- **Source:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)  
- **Files Used:** `winequality-red.csv` and `winequality-white.csv`  
- **Features:** 11 physicochemical properties (e.g., acidity, sugar, pH, alcohol)  
- **Target:** Wine quality score (0–10)  

---

## ✨ Features  

- Visualize PCA (2D & 3D scatter plots)  
- Compare scaling methods (Standard vs Robust)  
- Train/test split & accuracy metrics  
- Interactive model selection via Hugging Face demo  

---

## 📌 Future Improvements  

- Add hyperparameter tuning (GridSearchCV)  
- More advanced ML models (XGBoost, LightGBM)  
- Combine red & white wine datasets with classification  

---

## 📜 License  

This project is licensed under the **MIT License**.  
