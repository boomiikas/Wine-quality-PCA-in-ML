import gradio as gr
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Load dataset ---
df = pd.read_csv("cleaned_dataset.csv")
X = df.drop("quality", axis=1)
y = df["quality"]

# --- Scale + PCA ---
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensions using PCA (keep most variance)
pca = PCA(n_components=6)  # you can change to 2 for visualization, 6 for ~90% variance
X_pca = pca.fit_transform(X_scaled)

# --- Train Classifier ---
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

print("Model Evaluation:\n", classification_report(y_test, clf.predict(X_test)))

# --- User Authentication ---
VALID_USERS = {"admin": "1234", "user": "pass"}

def authenticate(username, password):
    return username in VALID_USERS and VALID_USERS[username] == password

def predict_quality(username, password, features):
    if not authenticate(username, password):
        return "❌ Invalid login!", None
    
    try:
        features = [float(x) for x in features.split(",")]
        if len(features) != X.shape[1]:
            return f"⚠️ Please enter {X.shape[1]} valid numbers (comma-separated).", None
        
        # Scale + PCA transform
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        
        # Predict
        prediction = clf.predict(features_pca)[0]
        return f"🔮 Predicted Wine Quality: {prediction}", int(prediction)
    except:
        return "⚠️ Invalid input format.", None


# --- Gradio App ---
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🍇 Wine Quality Classification (PCA + Random Forest)")

    username = gr.Textbox(label="Username")
    password = gr.Textbox(label="Password", type="password")
    features = gr.Textbox(
        label=f"Enter Features (comma separated, {X.shape[1]} values)",
        placeholder="7.4,0.7,0.0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4"
    )
    predict_btn = gr.Button("✨ Predict Quality")
    output_text = gr.Textbox(label="Result")
    output_quality = gr.Number(label="Predicted Quality")

    predict_btn.click(predict_quality, [username, password, features], [output_text, output_quality])

demo.launch()
