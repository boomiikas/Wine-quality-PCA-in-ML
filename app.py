import gradio as gr
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

# --- Load saved items ---
model = joblib.load("best_cluster_model.pkl")   # DBSCAN
scaler = joblib.load("scaler.pkl")
labels = joblib.load("cluster_labels.pkl")
quality_map = joblib.load("cluster_quality_map.pkl")
X_scaled = np.load("X_scaled.npy")

# Build Nearest Neighbors for DBSCAN prediction
nn = NearestNeighbors(n_neighbors=1).fit(X_scaled)

# --- Authentication ---
VALID_USERS = {"admin": "1234", "user": "pass"}

def authenticate(username, password):
    return username in VALID_USERS and VALID_USERS[username] == password

# --- Prediction Function ---
def predict_cluster(username, password, features):
    if not authenticate(username, password):
        return "❌ Invalid login!", None
    
    try:
        features = [float(x) for x in features.split(",")]
        features = np.array([features])
        features_scaled = scaler.transform(features)
    except Exception:
        return "⚠️ Please enter 11 valid numeric features, comma-separated!", None

    # Find nearest neighbor cluster label
    _, idx = nn.kneighbors(features_scaled)
    cluster = labels[idx[0][0]]

    # Interpret cluster
    if cluster == -1:
        result = "🚨 Noise/Outlier (Does not belong to any cluster)"
    else:
        result = quality_map.get(cluster, f"Cluster {cluster} (Unknown Quality)")

    return f"🔮 Prediction: {result}", int(cluster)

# --- Gradio App ---
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🍇 Wine Quality Prediction with Clustering (DBSCAN Best)")

    with gr.Tab("🔑 Login & Predict"):
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        features = gr.Textbox(
            label="Enter Features (comma separated, 11 values)",
            placeholder="7.4,0.7,0.0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4"
        )
        predict_btn = gr.Button("✨ Predict Cluster")
        output_text = gr.Textbox(label="Result", interactive=False)
        output_cluster = gr.Number(label="Cluster ID", interactive=False)

        predict_btn.click(
            predict_cluster,
            inputs=[username, password, features],
            outputs=[output_text, output_cluster]
        )

demo.launch()
