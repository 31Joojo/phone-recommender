"""
rag.py

Auteurs : BAUDET Quentin & LARMAILLARD-NOIREN Joris

Objectifs :
  1. Charger un modèle RAG pré-entraîné (Rétrieve + Génération) depuis
     Hugging Face.
  2. Créer et configurer le RagRetriever pointant vers les passages
     JSONL et l'index FAISS.
  3. Retourner un pipeline ou un couple (model, tokenizer) prêt à générer
     des réponses conditionnées.

Variables clés :
  - MODEL_NAME : identifiant Hugging Face du modèle RAG.
  - PASSAGES_PATH, INDEX_PATH : chemins vers nos ressources locales.

Usage :
    from rag import get_rag_pipeline
    rag_pipeline = get_rag_pipeline()
    rag_pipeline("Votre question ici")
"""

# ------------------------ Importation des modules ------------------------ #
from transformers import (
    RagTokenizer, RagRetriever, RagSequenceForGeneration
)
### Chargement des variables d'environnement
from dotenv import load_dotenv
import os

load_dotenv()

# ----------------------- Chargement des variables ------------------------ #
MODEL_NAME = os.getenv("MODEL_NAME")
PASSAGES_PATH = os.getenv("PASSAGES_PATH")
INDEX_PATH = os.getenv("INDEX_PATH")

# -------------- Chargement du tokenizer et du modèle de base ------------- #
tokenizer = RagTokenizer.from_pretrained(MODEL_NAME)
model = RagSequenceForGeneration.from_pretrained(MODEL_NAME)

# ------------------- Recherche l'index et les passages ------------------- #
retriever = RagRetriever.from_pretrained(
    MODEL_NAME,
    index_name="custom",
    passages_path=PASSAGES_PATH,
    index_path=INDEX_PATH,
    use_dummy_dataset=False  # on utilise bien ton dataset
)

# ------------------- Ajout du retriever dans le modèle ------------------- #
model.set_retriever(retriever)
