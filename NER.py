import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder
import subprocess
import re
from datetime import datetime
import os

# Charger le dataset avec gestion d'exception
try:
    data_cleaned = pd.read_csv('data/dataset_cleaned.csv')
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the path to 'dataset_cleaned.csv'.")
    exit()

# Définir les maladies et les symptômes connus
diseases = data_cleaned['Disease'].unique()
known_symptoms = set(data_cleaned['Symptoms'].str.strip().str.lower().str.split(",").sum())

# Initialiser le modèle et le tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=len(diseases))
    model.load_state_dict(torch.load('model_biobert.pth', map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    print("Error: Model file not found. Please ensure 'model_biobert.pth' is in the correct directory.")
    exit()

tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

# Encoder les étiquettes de maladie
label_encoder = LabelEncoder()
label_encoder.fit(data_cleaned['Disease'])

# Fonction de normalisation des symptômes
def normalize_symptom(symptom):
    return symptom.strip().replace(" ", "_").lower()

# Fonction pour extraire les symptômes du message utilisateur
def extract_symptoms_from_message(message):
    prompt = f"Extract all symptoms mentioned in the following message exactly as they are. Separate each symptom with a comma. Message: '{message}'"
    result = subprocess.run(["ollama", "run", "llama3.2", prompt], capture_output=True, text=True, encoding='utf-8')
    response = result.stdout.strip()

    # Supprimer les textes inutiles
    response = re.sub(r"^.*symptoms extracted from the message:\n*", "", response, flags=re.IGNORECASE)
    symptoms = [normalize_symptom(symptom) for symptom in response.split(',') if symptom.strip()]

    # Ignorer les réponses comme "no", "yes", ou similaires
    symptoms = [symptom for symptom in symptoms if symptom not in {"no", "yes", "none"}]

    valid_symptoms = [symptom for symptom in symptoms if symptom in known_symptoms]
    return valid_symptoms

# Fonction pour récupérer les symptômes d'une maladie
def get_disease_symptoms(disease):
    disease_data = data_cleaned[data_cleaned['Disease'] == disease]
    if disease_data.empty:
        return set()
    disease_symptoms = set(disease_data['Symptoms'].str.strip().str.lower().str.split(",").sum())
    return {normalize_symptom(symptom) for symptom in disease_symptoms}

# Fonction pour trouver les maladies candidates
def find_candidate_diseases(confirmed_symptoms):
    candidates = []
    for disease in diseases:
        disease_symptoms = get_disease_symptoms(disease)
        common_symptoms = confirmed_symptoms.intersection(disease_symptoms)
        score = len(common_symptoms) / len(disease_symptoms)  # Score de similarité
        candidates.append((disease, score, disease_symptoms))
    
    # Trier les maladies par score décroissant
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    # Filtrer pour conserver uniquement les maladies proches de la meilleure correspondance
    if candidates:
        max_score = candidates[0][1]
        threshold = 0.2  # Par exemple, 20 % de différence maximum avec la meilleure correspondance
        candidates = [c for c in candidates if max_score - c[1] <= threshold]
    
    return candidates

# Fonction pour poser des questions discriminantes
def refine_prediction_with_discriminative_symptoms(candidates, confirmed_symptoms, infirmed_symptoms):
    # Extraire les symptômes pour les maladies probables
    disease_1, disease_2 = candidates[0][0], candidates[1][0]
    symptoms_1, symptoms_2 = set(candidates[0][2]), set(candidates[1][2])

    # Symptômes discriminants
    discriminative_symptoms = symptoms_1.symmetric_difference(symptoms_2)

    for symptom in discriminative_symptoms:
        if symptom in confirmed_symptoms or symptom in infirmed_symptoms:
            continue  # Passer si le symptôme a déjà été traité

        # Poser la question à l'utilisateur
        response = input(f"Do you have {symptom.replace('_', ' ')}? (yes/no or describe): ").strip().lower()

        if response == "yes":
            confirmed_symptoms.add(symptom)
        elif "no" in response:
            infirmed_symptoms.add(symptom)
        elif response:  # Si l'utilisateur fournit une description détaillée
            additional_symptoms = extract_symptoms_from_message(response)
            if additional_symptoms:
                confirmed_symptoms.update(additional_symptoms)

    return confirmed_symptoms, infirmed_symptoms

# Fonction pour confirmer les symptômes manquants
def confirm_missing_symptoms(user_symptoms, disease_symptoms):
    missing_symptoms = disease_symptoms - set(user_symptoms)
    confirmed_symptoms = set(user_symptoms)
    infirmed_symptoms = set()  # Liste pour les symptômes infirmés

    for symptom in list(missing_symptoms):
        if symptom in confirmed_symptoms or symptom in infirmed_symptoms:
            continue

        response = input(f"Do you have {symptom.replace('_', ' ')}? (yes/no or describe): ").strip().lower()

        if response == "yes":
            confirmed_symptoms.add(symptom)
        elif "no" in response:
            infirmed_symptoms.add(symptom)
        elif response:
            additional_symptoms = extract_symptoms_from_message(response)
            if additional_symptoms:
                confirmed_symptoms.update(additional_symptoms)

    return confirmed_symptoms, infirmed_symptoms

# Fonction pour sauvegarder le rapport utilisateur
def save_user_report(user_symptoms, confirmed_symptoms, infirmed_symptoms, predicted_disease, disease_symptoms):
    report_filename = "user_reports/all_reports.txt"

    # Symptômes non confirmés
    non_confirmed_symptoms = disease_symptoms - confirmed_symptoms - infirmed_symptoms

    # Symptômes ignorés (non reconnus)
    ignored_symptoms = [symptom for symptom in user_symptoms if symptom not in confirmed_symptoms and symptom not in infirmed_symptoms]

    # Contenu du rapport
    report_content = f"""
==================== User Report ====================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Predicted Disease: {predicted_disease}

Symptoms Provided by User:
{', '.join(user_symptoms)}

Confirmed Symptoms:
{', '.join(confirmed_symptoms)}

Non-Confirmed Symptoms:
{', '.join(non_confirmed_symptoms)}

Infirmed Symptoms:
{', '.join(infirmed_symptoms)}

Ignored Symptoms (Not Recognized):
{', '.join(ignored_symptoms)}

=====================================================
"""

    # Créer le dossier des rapports si nécessaire
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)

    with open(report_filename, "a") as report_file:
        report_file.write(report_content)

    print(f"Report saved to {report_filename}")

# Programme principal
if __name__ == "__main__":
    print("Welcome to the disease prediction system based on your symptoms.")
    
    user_message = input("Please describe your symptoms in a sentence: ")
    symptoms_list = extract_symptoms_from_message(user_message)

    if not symptoms_list:
        print("No valid symptoms were detected. Please provide more detailed descriptions of your symptoms.")
        exit()

    confirmed_symptoms = set(symptoms_list)
    infirmed_symptoms = set()

    # Recherche des maladies probables
    while True:
        candidates = find_candidate_diseases(confirmed_symptoms)

        if len(candidates) > 1:
            print("\n======== Probable Diseases ========")
            for disease, score, _ in candidates:
                print(f"{disease}: {int(score * 100)}% match")
            print("===================================\n")

            # Poser des questions discriminantes si nécessaire
            confirmed_symptoms, infirmed_symptoms = refine_prediction_with_discriminative_symptoms(
                candidates, confirmed_symptoms, infirmed_symptoms
            )
        else:
            break

    # Prédiction finale
    predicted_disease = candidates[0][0]
    disease_symptoms = get_disease_symptoms(predicted_disease)

    # Vérification des symptômes pour confirmer la maladie finale
    if confirmed_symptoms < disease_symptoms:
        confirmed_symptoms, infirmed_symptoms = confirm_missing_symptoms(
            confirmed_symptoms, disease_symptoms
        )

    print(f"Predicted disease: {predicted_disease}")
    save_user_report(symptoms_list, confirmed_symptoms, infirmed_symptoms, predicted_disease, disease_symptoms)
    print("Analysis completed.")
