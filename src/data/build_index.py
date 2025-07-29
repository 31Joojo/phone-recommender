"""
build_index.py

Auteurs : BAUDET Quentin & LARMAILLARD-NOIREN Joris

Objectif : Encoder tous les passages JSONL en vecteurs d'embedding via un
      modèle SentenceTransformer, puis construire et sauvegarder un
      index FAISS pour recherches de similarité.

Variables clés :
  - TRANSFORMER : modèle utilisé pour le calcul des embeddings.

Entrée : data/processed/passages.jsonl
Sortie : models/faiss_index.idx
Usage :
    python build_index.py \
      --passages data/processed/passages.jsonl \
      --index models/faiss_index.idx
"""

# ------------------------ Importation des modules ------------------------ #
from sentence_transformers import SentenceTransformer
import faiss
import json

# ---------------- Chargement des variables d'environnement --------------- #
from src.config import TRANSFORMER, DEVICE

# --------------- Chargement du modèle d'embeddings - BERT ---------------- #
embedder = SentenceTransformer(TRANSFORMER, device=DEVICE)

texts: list = []
# ------------------------ Chargement des passages ------------------------ #
with open("../../data/processed/passages.jsonl") as f:
    for line in f:
        j = json.loads(line)
        texts.append(j["text"])

# ------------------------- Calcul des embeddings ------------------------- #
embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# -------------------------- Création de l'index -------------------------- #
dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs)
faiss.write_index(index, "../../data/processed/faiss_index.idx")
