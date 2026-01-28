import os
import numpy as np
import tiktoken
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


# ================= CONFIG =================
PROMPT_DIR = "prompts"          # pasta com .txt
OUTPUT_DIR = "outputs"          # html interativo
WINDOW_SIZE = 24                # tokens por janela
STRIDE = 4                      # sobreposição
ENCODING = "cl100k_base"

os.makedirs(OUTPUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding(ENCODING)


# ================= TOKEN WINDOWS =================
def token_windows(text, window=24, stride=4):
    tokens = enc.encode(text)
    chunks = []
    idx = []
    for i in range(0, len(tokens) - window + 1, stride):
        chunk_tokens = tokens[i:i + window]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        idx.append(i)
    return chunks, idx


# ================= PIPELINE =================
for fname in os.listdir(PROMPT_DIR):
    if not fname.endswith(".txt"):
        continue

    prompt_name = fname.replace(".txt", "")
    with open(os.path.join(PROMPT_DIR, fname), "r", encoding="utf-8") as f:
        text = f.read()

    # 1) janelas densas de tokens
    chunks, starts = token_windows(text, WINDOW_SIZE, STRIDE)

    # 2) ENCODER ONLY: TF-IDF sobre texto tokenizado
    # (visualização do espaço lexical/probabilístico, sem semântica treinada)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^ ]+",
        max_features=3000
    )
    X = vectorizer.fit_transform(chunks).toarray()

    # 3) PCA 3D
    pca = PCA(n_components=3, random_state=42)
    X3 = pca.fit_transform(X)

    # 4) gráfico interativo
    fig = px.scatter_3d(
        x=X3[:, 0],
        y=X3[:, 1],
        z=X3[:, 2],
        color=starts,                      # gradiente temporal
        color_continuous_scale="Viridis",
        title=f"{prompt_name} — Espaço ENCODER (TF-IDF) • Sliding Window",
        labels={"x": "PC1", "y": "PC2", "z": "PC3"},
        hover_data={
            "token_start": starts,
            "window_text": chunks
        }
    )

    fig.update_traces(marker=dict(size=4, opacity=0.85))
    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        ),
        height=800
    )

    out = os.path.join(OUTPUT_DIR, f"{prompt_name}_encoder_interactive.html")
    fig.write_html(out)
