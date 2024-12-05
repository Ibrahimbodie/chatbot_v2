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
    prompt = f"Extract the symptoms mentioned in the message exactly as they appear, without rephrasing. Separate each symptom with a comma. Message: '{message}'"
    result = subprocess.run(["ollama", "run", "llama3.2", prompt], capture_output=True, text=True, encoding='utf-8')
    response = result.stdout.strip()
    
    # Supprimer les textes inutiles
    response = re.sub(r"^.*symptoms extracted from the message:\n*", "", response, flags=re.IGNORECASE)
    symptoms = [normalize_symptom(symptom) for symptom in response.split(',') if symptom.strip()]
    valid_symptoms = [symptom for symptom in symptoms if symptom in known_symptoms]
    
    return valid_symptoms

# Fonction pour récupérer les symptômes d'une maladie
def get_disease_symptoms(disease):
    disease_data = data_cleaned[data_cleaned['Disease'] == disease]
    if disease_data.empty:
        return set()
    disease_symptoms = set(disease_data['Symptoms'].str.strip().str.lower().str.split(",").sum())
    return {normalize_symptom(symptom) for symptom in disease_symptoms}

# Fonction pour confirmer les symptômes manquants
def confirm_missing_symptoms(user_symptoms, disease_symptoms):
    missing_symptoms = disease_symptoms - set(user_symptoms)
    confirmed_symptoms = set(user_symptoms)

    for symptom in list(missing_symptoms):
        if symptom not in confirmed_symptoms:  # Vérifie si le symptôme n'est pas déjà confirmé
            response = input(f"Do you have {symptom.replace('_', ' ')}? (yes/no or describe): ").strip().lower()

            if response == "yes":
                confirmed_symptoms.add(symptom)

            elif "no" in response:
                additional_symptoms = extract_symptoms_from_message(response)
                if additional_symptoms:
                    print(f"Additional symptoms detected: {', '.join(additional_symptoms)}")
                    confirmed_symptoms.update(additional_symptoms)

            missing_symptoms = disease_symptoms - confirmed_symptoms

            if not missing_symptoms:
                break

    return confirmed_symptoms

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
    return candidates

# Fonction pour demander des symptômes discriminants
# Variable de contrôle pour l'affichage
DISPLAY_DISCRIMINATIVE_SYMPTOMS = False  # Passe à True pour afficher les informations

def refine_prediction_with_discriminative_symptoms(candidates, confirmed_symptoms):
    # Prendre les deux premières maladies candidates
    top_candidates = candidates[:2]

    # Affichage conditionnel des maladies candidates
    if DISPLAY_DISCRIMINATIVE_SYMPTOMS:
        for i, (disease, score, symptoms) in enumerate(top_candidates, 1):
            print(f"{i}. {disease} (Match: {int(score * 100)}%)")

    # Chercher les symptômes discriminants entre les deux maladies principales
    disease_1, disease_2 = top_candidates[0][0], top_candidates[1][0]
    symptoms_1, symptoms_2 = set(top_candidates[0][2]), set(top_candidates[1][2])
    discriminative_symptoms = symptoms_1.symmetric_difference(symptoms_2)  # Symptômes distincts

    for symptom in discriminative_symptoms:
        response = input(f"Do you have {symptom.replace('_', ' ')}? (yes/no or describe): ").strip().lower()
        if response == "yes":
            confirmed_symptoms.add(symptom)
        elif "no" in response:
            additional_symptoms = extract_symptoms_from_message(response)
            if additional_symptoms:
                if DISPLAY_DISCRIMINATIVE_SYMPTOMS:
                    print(f"Additional symptoms detected: {', '.join(additional_symptoms)}")
                confirmed_symptoms.update(additional_symptoms)

    return confirmed_symptoms


# Fonction pour sauvegarder le rapport de l'utilisateur
# Fonction pour sauvegarder le rapport de l'utilisateur

# Function to save the user report

def save_user_report(user_symptoms, confirmed_symptoms, predicted_disease, disease_symptoms):
    report_filename = "user_reports/all_reports.txt"  # Single file for all reports

    # Non-confirmed symptoms (part of the disease symptoms but not confirmed by the user)
    non_confirmed_symptoms = [symptom for symptom in disease_symptoms if symptom not in confirmed_symptoms and not symptom.startswith("not_")]

    # Explicitly rejected symptoms
    rejected_symptoms = [symptom for symptom in confirmed_symptoms if symptom.startswith("not_")]

    # Report content
    report_content = f"""
==================== User Report ====================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

User Actions:
- Provided Symptoms: {', '.join(user_symptoms)}
- Confirmed Symptoms: {', '.join(symptom for symptom in confirmed_symptoms if not symptom.startswith('not_'))}
- Non-Confirmed Symptoms: {', '.join(non_confirmed_symptoms)}
- Rejected Symptoms: {', '.join(rejected_symptoms)}

Predicted Disease: {predicted_disease}
=====================================================
"""

    # Create the reports folder if needed
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)

    # Write to the single report file using UTF-8 encoding
    with open(report_filename, "a", encoding="utf-8") as report_file:
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
    candidates = find_candidate_diseases(confirmed_symptoms)

    # Vérifier si plusieurs maladies sont candidates
    if len(candidates) > 1 and candidates[0][1] - candidates[1][1] < 0.2:  # Différence de score < 20%
        confirmed_symptoms = refine_prediction_with_discriminative_symptoms(candidates, confirmed_symptoms)
        candidates = find_candidate_diseases(confirmed_symptoms)

    # Étape 3 : Prédire la maladie finale avec confirmation des symptômes
    predicted_disease = candidates[0][0]
    disease_symptoms = get_disease_symptoms(predicted_disease)

    if confirmed_symptoms < disease_symptoms:
        print("Not enough symptoms match the predicted disease. Let's confirm some additional symptoms.")
        confirmed_symptoms = confirm_missing_symptoms(confirmed_symptoms, disease_symptoms)

    print("The consultation is done for preconsultation")
    save_user_report(symptoms_list, confirmed_symptoms, predicted_disease, disease_symptoms)
    print("Analysis completed.")
