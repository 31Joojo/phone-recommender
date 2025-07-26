# PhoneRecommender

Un chatbot de recommandation de smartphones basé sur un pipeline **Retrieval-Augmented Generation (RAG)** et un modèle de langage pré-entraîné.

---

## Contexte et objectifs

Ce projet a pour but de construire un assistant intelligent capable d’aider un utilisateur à choisir un téléphone portable en fonction de critères (prix, performance, autonomie, photo, etc.).
Pour cela, nous exploitons :
- Un **dataset Kaggle** contenant les fiches techniques et les spécifications de centaines de modèles de smartphones.
- Un pipeline **RAG** : on indexe les descriptions produits dans un moteur de similarité (FAISS) puis on combine la récupération de documents pertinents à la génération de texte par un **LLM** (Mistral 7B, GPT-2, LLaMA …, au choix).

---

## Fonctionnalités

1. **Exploration et nettoyage du dataset**
   - Analyse exploratoire (EDAx).
   - Standardisation des noms de champs (RAM, stockage, autonomie…).
2. **Indexation documentaire**
   - Conversion du CSV en fragments textuels.
   - Construction d’un index FAISS pour recherches rapides.
3. **Pipeline RAG**
   - À chaque question utilisateur, récupération des **k fiches les plus pertinentes**.
   - Enrichissement du prompt avec ces extraits.
   - Génération de la réponse via un LLM.
4. **Interface**
   - API REST légère (FastAPI/Flask) pour exposer le chatbot.
   - Client en ligne de commande ou interface web minimale (optionnel).

---

## 📂 Structure du dépôt

