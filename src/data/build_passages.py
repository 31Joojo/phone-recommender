"""
build_passages.py

Auteurs : BAUDET Quentin & LARMAILLARD-NOIREN Joris

Objectif : Transformer chaque enregistrement du CSV nettoyé en un document texte
      « passage » lisible, contenant toutes les spécifications produit.

Entrée : data/processed/Smartphones_cleaned_dataset_processed.csv
Sortie : data/processed/passages.jsonl
Usage :
    python build_passages.py \
      --input data/processed/Smartphones_cleaned_dataset_processed.csv \
      --output data/processed/passages.jsonl
"""

# ------------------------ Importation des modules ------------------------ #
import pandas as pd, json

# ------------------------ Chargement des données ------------------------- #
data = pd.read_csv('data/processed/Smartphones_cleaned_dataset_processed.csv')

passages: list = []
for _, phone in data.iterrows():
    title = f"{phone['model']}"

    ### Formatage du texte pour l'enregistrement dans la base de données jsonl
    text = (
        f"Prix : {phone['price (€)']} €, Segment du prix : {phone['price_segment']}, "
        f"Taille d'écran : {phone['screen_size']}”, Taux de rafraîchissement : {phone['refresh_rate']} Hz, "
        f"Note : {phone['rating']}, Rapport qualité-prix : {phone['quality-price_ratio']}, "
        f"5G : {'Disponible' if phone['has_5G'] else 'Non disponible'}, "
        f"Stockage : {phone['internal_memory']} Go, RAM : {phone['ram_capacity']} Go, "
        f"CPU : {phone['processor_brand']} {phone['num_cores'] * phone['processor_speed']} GHz, "
        f"Batterie : {phone['battery_capacity']} mAh, Autonomie : {phone['quality_battery_autonomy']}, "
        f"Charge rapide : {'Non disponible' if phone['fast_charging_available'] == 0 else 'Disponible'}, "
        f"App. AR : {phone['num_rear_cameras']} ×, App. AV : {phone['num_front_cameras']} ×, "
        f"Caméra prin. AR : {phone['primary_camera_rear']} Mpx, Caméra prin. AV : {phone['primary_camera_front']} Mpx, "
        f"OS : {phone['os']}"
    )

    ### Ajout des données dans la liste des passages
    passages.append({'title': title, 'text': text})

# --------------- Enregistrement des données au format json --------------- #
with open('data/processed/passages.jsonl', 'w') as f:
    for passage in passages:
        f.write(json.dumps(passage, ensure_ascii=False) + "\n")
