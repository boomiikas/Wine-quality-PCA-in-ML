import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- CONFIG ----------
DATA_PATH = "winequality-red.csv"   # change to white if needed
USERS = {"admin": "1234", "boomika": "1234"}  # login users

# ---------- Load & preprocess ----------
df = pd.read_csv(DATA_PATH, sep=";")
X = df.drop(columns=["quality"]).values.astype(float)

# Fit scaler + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)  # keep first 5 PCs
pca.fit(X_scaled)

# Feature names
numeric_cols = df.drop(columns=["quality"]).columns.tolist()

# ---------- Helper Functions ----------
def check_login(username, password):
    return username in USERS and USERS[username] == password

def predict_pca(*user_inputs):
    user_arr = np.array([user_inputs], dtype=float)
    user_scaled = scaler.transform(user_arr)
    user_pca = pca.transform(user_scaled)
    return {
        "PC1": round(user_pca[0,0], 4),
        "PC2": round(user_pca[0,1], 4),
        "PC3": round(user_pca[0,2], 4)
    }

# ---------- Build Gradio App ----------
with gr.Blocks() as demo:
    # --- Login UI ---
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🔐 Login")
            username_input = gr.Textbox(label="Username")
            password_input = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_message = gr.Textbox(label="Status")

    # --- Main UI (hidden until login) ---
    main_ui = gr.Column(visible=False)
    with main_ui:
        gr.Markdown("## 🍷 Wine PCA Predictor")
        gr.Markdown("Enter chemical features to compute PCA components.")
        input_fields = [gr.Number(label=col) for col in numeric_cols]
        output_json = gr.JSON(label="PCA Components")
        predict_btn = gr.Button("Compute PCA")
        predict_btn.click(fn=predict_pca, inputs=input_fields, outputs=output_json)

    # --- Login Action ---
    def login_action(username, password):
        if check_login(username, password):
            return "✅ Login successful! You can now use the PCA tool.", gr.update(visible=True)
        else:
            return "❌ Login failed! Check username/password.", gr.update(visible=False)

    login_btn.click(fn=login_action,
                    inputs=[username_input, password_input],
                    outputs=[login_message, main_ui])

# ---------- Launch ----------
if __name__ == "__main__":
    demo.launch(share=True)
