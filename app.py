import  gradio as gr
import  pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")  # change filename if needed
X = df.drop(columns='quality', axis=1, errors="ignore")
# Scale & fit PCA
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Store login credentials (simple dictionary)
USERS = {"admin": "1234", "user": "pass"}

# Function to check login
def login(username, password):
    if username in USERS and USERS[username] == password:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(value="❌ Invalid credentials", visible=True), gr.update(visible=False)

# PCA prediction function
def predict_pc(*features):
    scaled = scaler.transform([features])
    pcs = pca.transform(scaled)
    return float(pcs[0, 0]), float(pcs[0, 1])

# Build login UI
with gr.Blocks() as demo:
    gr.Markdown("## 🔑 PCA Prediction App (Wine Dataset)")

    # Login section
    with gr.Row(visible=True) as login_row:
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_status = gr.Textbox(label="Status", visible=False)

    # PCA input section (hidden until login)
    with gr.Row(visible=False) as pca_row:
        inputs = []
        for col in X.columns:
            inputs.append(gr.Number(label=col))

        predict_btn = gr.Button("Predict PC1 & PC2")
        output_pc1 = gr.Number(label="PC1")
        output_pc2 = gr.Number(label="PC2")

    # Bind login
    login_btn.click(
        login,
        inputs=[username, password],
        outputs=[login_status, pca_row],
    )

    # Bind PCA prediction
    predict_btn.click(
        predict_pc,
        inputs=inputs,
        outputs=[output_pc1, output_pc2],
    )

# Run app
demo.launch()
