import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import subprocess
import re

# Charger le dataset
if not os.path.exists('data/dataset_cleaned.csv'):
    print("Erreur : Le fichier 'dataset_cleaned.csv' est introuvable.")
    exit()

data_cleaned = pd.read_csv('data/dataset_cleaned.csv')  # Remplacez par le bon chemin
data_cleaned = data_cleaned[data_cleaned['Symptoms'].notna()]  # Retirer les lignes avec des valeurs NaN
data_cleaned = data_cleaned[data_cleaned['Symptoms'].str.strip() != '']  # Retirer les lignes avec des symptômes vides

if data_cleaned.empty:
    print("Erreur : Le dataset ne contient aucune entrée valide.")
    exit()

# Définir les maladies et les symptômes connus
diseases = data_cleaned['Disease'].unique()
known_symptoms = set(data_cleaned['Symptoms'].str.strip().str.lower().str.split(",").sum())

# Seuil de validation pour la correspondance des symptômes
VALIDATION_THRESHOLD = 0.75

# Initialiser le modèle et le tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=len(diseases))

if os.path.exists('model_biobert.pth'):
    model.load_state_dict(torch.load('model_biobert.pth', map_location=device))
else:
    print("Erreur : Le fichier du modèle 'model_biobert.pth' est introuvable. Entraînez d'abord le modèle.")
    exit()

model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

# Encoder les étiquettes de maladie
label_encoder = LabelEncoder()
label_encoder.fit(data_cleaned['Disease'])

# Fonction de normalisation des symptômes (sans modifier les underscores)
def normalize_symptom(symptom):
    return symptom.strip().lower()

# Fonction pour exécuter la commande ollama
def run_ollama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2", prompt],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode != 0:
            print(f"Erreur d'exécution de la commande ollama : {result.stderr}")
            return None
        return result.stdout.strip()
    except FileNotFoundError:
        print("Erreur : La commande 'ollama' est introuvable. Vérifiez son installation.")
    except Exception as e:
        print(f"Erreur inattendue lors de l'appel à ollama : {e}")
    return None

# Fonction pour extraire les symptômes du message utilisateur
def extract_symptoms_from_message(message):
    prompt = f"Extract all symptoms mentioned in the message and list them separated by commas. Message: '{message}'"
    response = run_ollama(prompt)
    if response is None:
        print("Erreur lors de l'extraction des symptômes via ollama.")
        return []

    symptoms = [normalize_symptom(symptom) for symptom in response.split(',') if symptom.strip()]
    return [symptom for symptom in symptoms if symptom in known_symptoms]

# Fonction pour récupérer les symptômes d'une maladie
def get_disease_symptoms(disease):
    disease_data = data_cleaned[data_cleaned['Disease'] == disease]
    if disease_data.empty:
        return set()
    disease_symptoms = set(data_cleaned.loc[data_cleaned['Disease'] == disease, 'Symptoms'].str.strip().str.lower().str.split(",").sum())
    return {normalize_symptom(symptom) for symptom in disease_symptoms}

# Fonction pour prédire les maladies possibles
def predict_disease(symptoms):
    symptoms_text = ", ".join(symptoms)
    tokens = tokenizer([symptoms_text], padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    predicted_disease = label_encoder.inverse_transform(predictions.cpu().numpy())
    return predicted_disease[0]

# Fonction pour enregistrer un rapport
def save_report(symptoms, confirmed_symptoms, predicted_disease):
    with open("user_symptoms_report.txt", "a") as report_file:
        report_file.write("User Symptoms Report\n")
        report_file.write("===================\n")
        report_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write(f"Symptoms provided by user: {', '.join(symptoms)}\n")
        report_file.write(f"Confirmed symptoms: {', '.join(confirmed_symptoms)}\n")
        report_file.write(f"Predicted disease: {predicted_disease}\n\n")

# Fonction pour poser des questions spécifiques sur les symptômes manquants
def ask_about_missing_symptoms(missing_symptoms, confirmed_symptoms, previous_match_percentage):
    while True:
        print("Let's confirm some additional symptoms.")
        new_symptoms_added = False  # Indique si des symptômes ont été ajoutés
        for symptom in missing_symptoms:
            question = f"Do you also have {symptom.replace('_', ' ')}? (yes/no or describe your condition): "
            print(question, end="")
            response = input().strip().lower()

            if response == "yes":
                confirmed_symptoms.add(symptom)
                new_symptoms_added = True
            elif response == "no":
                continue
            else:
                additional_symptoms = extract_symptoms_from_message(response)
                if additional_symptoms:
                    print(f"Detected additional symptoms from your response: {', '.join(additional_symptoms)}")
                    confirmed_symptoms.update(additional_symptoms)
                    new_symptoms_added = True

        # Si aucun nouveau symptôme n'a été ajouté, arrêter le processus
        if not new_symptoms_added:
            print("No additional symptoms provided. Stopping confirmation process.")
            return confirmed_symptoms, None  # Retourne les symptômes confirmés et `None` pour la maladie

        # Recalculer la prédiction après chaque série de réponses
        predicted_disease = predict_disease(confirmed_symptoms)
        print(f"Updated prediction: {predicted_disease}")

        # Vérifier le pourcentage de correspondance et les nouveaux symptômes manquants
        disease_symptoms = get_disease_symptoms(predicted_disease)
        missing_symptoms = disease_symptoms - confirmed_symptoms
        match_percentage = (
            len(confirmed_symptoms.intersection(disease_symptoms)) / len(disease_symptoms)
            if disease_symptoms else 0
        )
        print(f"Updated match percentage: {match_percentage:.2f}")

        # Si le nouveau pourcentage est plus élevé que le précédent, relancer la boucle avec les nouveaux symptômes
        if match_percentage > previous_match_percentage:
            print(f"New prediction '{predicted_disease}' has a higher match percentage. Restarting confirmation.")
            previous_match_percentage = match_percentage
            continue  # Recommencer avec les symptômes manquants de cette nouvelle prédiction

        # Si le seuil de confiance est atteint, arrêter
        if match_percentage >= VALIDATION_THRESHOLD:
            print(f"High confidence prediction: {predicted_disease}")
            return confirmed_symptoms, predicted_disease

        # Si aucun symptôme supplémentaire n'est manquant, arrêter la boucle
        if not missing_symptoms:
            print("No more symptoms to confirm.")
            return confirmed_symptoms, predicted_disease



def interact_with_user():
    confirmed_symptoms = set()
    predicted_disease = None
    previous_match_percentage = 0

    while True:
        # Étape 1 : Demander les symptômes initiaux ou supplémentaires
        print("Please describe your symptoms or update them (e.g., 'I also have nausea, fever'): ", end="")
        user_message = input().strip()

        # Extraire les symptômes fournis
        new_symptoms = extract_symptoms_from_message(user_message)
        if not new_symptoms:
            print("No symptoms detected. Please provide clear symptom descriptions.")
            continue

        confirmed_symptoms.update(new_symptoms)
        print(f"Updated symptoms: {', '.join(confirmed_symptoms)}")

        # Étape 2 : Prédire la maladie
        predicted_disease = predict_disease(confirmed_symptoms)
        print(f"Predicted disease: {predicted_disease}")

        # Étape 3 : Vérifier les symptômes manquants
        disease_symptoms = get_disease_symptoms(predicted_disease)
        missing_symptoms = disease_symptoms - confirmed_symptoms
        match_percentage = len(confirmed_symptoms.intersection(disease_symptoms)) / len(disease_symptoms) if disease_symptoms else 0

        print(f"Match percentage: {match_percentage:.2f}")

        # Si le seuil de confiance est atteint, terminer
        if match_percentage >= VALIDATION_THRESHOLD:
            print(f"High confidence prediction: {predicted_disease}")
            break

        # Si des symptômes sont manquants, poser des questions
        if missing_symptoms:
            confirmed_symptoms, new_prediction = ask_about_missing_symptoms(missing_symptoms, confirmed_symptoms, previous_match_percentage)

            # Si une nouvelle maladie est prédite, mettre à jour
            if new_prediction is not None:
                predicted_disease = new_prediction

            # Mettre à jour le pourcentage précédent
            disease_symptoms = get_disease_symptoms(predicted_disease)
            previous_match_percentage = len(confirmed_symptoms.intersection(disease_symptoms)) / len(disease_symptoms) if disease_symptoms else 0

            if previous_match_percentage >= VALIDATION_THRESHOLD:
                print(f"High confidence prediction: {predicted_disease}")
                break

    # Étape finale : Sauvegarder le rapport
    save_report(list(confirmed_symptoms), confirmed_symptoms, predicted_disease)
    print("Analysis completed.")





# Programme principal
if __name__ == "__main__":
    interact_with_user()
