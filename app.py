import gradio as gr
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# --- Train inside the app ---
df = pd.read_csv("cleaned_dataset.csv")
X = df.drop("quality", axis=1)
y = df["quality"]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=2, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Build quality mapping
cluster_quality_map = {}
for cluster in np.unique(labels):
    if cluster == -1: continue
    cluster_indices = np.where(labels == cluster)[0]
    avg_quality = y.iloc[cluster_indices].mean()
    if avg_quality >= 5:
        cluster_quality_map[cluster] = "🍷 Good Quality"
    elif avg_quality >= 3.5:
        cluster_quality_map[cluster] = "👌 Medium Quality"
    else:
        cluster_quality_map[cluster] = "⚠️ Low Quality"

# Nearest neighbor for cluster prediction
nn = NearestNeighbors(n_neighbors=1).fit(X_scaled)

VALID_USERS = {"admin": "1234", "user": "pass"}

def authenticate(username, password):
    return username in VALID_USERS and VALID_USERS[username] == password

def predict_cluster(username, password, features):
    if not authenticate(username, password):
        return "❌ Invalid login!", None
    
    try:
        features = [float(x) for x in features.split(",")]
        features_scaled = scaler.transform([features])
    except:
        return "⚠️ Please enter 11 valid numbers (comma-separated).", None
    
    _, idx = nn.kneighbors(features_scaled)
    cluster = labels[idx[0][0]]

# Force assignment to nearest cluster (even if DBSCAN said -1)
    if cluster == -1:
    # Find nearest NON-noise cluster
        for neighbor_idx in idx[0]:
            if labels[neighbor_idx] != -1:
                cluster = labels[neighbor_idx]
                break
    result = cluster_quality_map.get(cluster, "Unknown Cluster")
    return f"🔮 Prediction: {result}", int(cluster)


with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🍇 Wine Quality Clustering (DBSCAN)")

    username = gr.Textbox(label="Username")
    password = gr.Textbox(label="Password", type="password")
    features = gr.Textbox(
        label="Enter Features (comma separated, 11 values)",
        placeholder="7.4,0.7,0.0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4"
    )
    predict_btn = gr.Button("✨ Predict Cluster")
    output_text = gr.Textbox(label="Result")
    output_cluster = gr.Number(label="Cluster ID")

    predict_btn.click(predict_cluster, [username, password, features], [output_text, output_cluster])

demo.launch()
