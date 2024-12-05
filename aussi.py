import logging
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder
import subprocess
import re
from datetime import datetime
import os

# Configurer le système de journalisation
logging.basicConfig(
    level=logging.INFO,  # Changez en DEBUG pour afficher plus de détails
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Affiche les logs dans la console
)

# Charger le dataset
data_cleaned = pd.read_csv('data/dataset_cleaned.csv')
if data_cleaned.empty:
    raise ValueError("Le fichier de données est vide ou introuvable.")

# Construire la liste des symptômes connus avec normalisation
def normalize_symptom(symptom):
    return symptom.strip().replace(" ", "_").lower()

known_symptoms = set(
    normalize_symptom(symptom) 
    for symptom in data_cleaned['Symptoms'].str.strip().str.lower().str.split(",").sum()
)

# Initialiser le modèle et le tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=len(data_cleaned['Disease'].unique()))
model.load_state_dict(torch.load('model_biobert.pth', map_location=device))
model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

# Encoder les étiquettes de maladie
label_encoder = LabelEncoder()
label_encoder.fit(data_cleaned['Disease'])

# Fonction pour extraire les symptômes du message utilisateur
def extract_symptoms_from_message(message):
    prompt = f"Extract the symptoms mentioned in the message exactly as they appear, without rephrasing. Separate each symptom with a comma. Message: '{message}'"
    try:
        # Appel à Ollama
        result = subprocess.run(
            ["ollama", "run", "llama3.2", prompt],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode != 0:
            logging.error("Erreur lors de l'appel à Ollama : %s", result.stderr)
            return []

        # Réponse brute
        response = result.stdout.strip()
        logging.debug("Réponse brute d'Ollama : %s", response)

        # Extraction et normalisation des symptômes
        symptoms = [normalize_symptom(symptom.strip()) for symptom in response.split(',') if symptom.strip()]
        logging.debug("Symptômes extraits et normalisés : %s", symptoms)

        # Validation contre les symptômes connus
        valid_symptoms = [symptom for symptom in symptoms if symptom in known_symptoms]
        invalid_symptoms = [symptom for symptom in symptoms if symptom not in known_symptoms]
        
        # Logs pour vérifier les symptômes validés et invalides
        logging.debug("Symptômes validés : %s", valid_symptoms)
        logging.debug("Symptômes non validés : %s", invalid_symptoms)

        return valid_symptoms, symptoms  # Retourne les symptômes validés et tous les extraits

    except Exception as e:
        logging.error("Erreur lors de l'extraction des symptômes : %s", str(e))
        return [], []

# Fonction pour récupérer les symptômes associés à une maladie
def get_disease_symptoms(disease):
    disease_data = data_cleaned[data_cleaned['Disease'] == disease]
    if disease_data.empty:
        return set()
    disease_symptoms = set(disease_data['Symptoms'].str.strip().str.lower().str.split(",").sum())
    return {normalize_symptom(symptom) for symptom in disease_symptoms}

# Fonction pour confirmer les symptômes manquants
def confirm_missing_symptoms(user_symptoms, disease_symptoms):
    # Trouver les symptômes manquants
    missing_symptoms = disease_symptoms - set(user_symptoms)
    confirmed_symptoms = set(user_symptoms)  # Inclure les symptômes initiaux

    # Poser des questions sur les symptômes manquants
    for symptom in missing_symptoms:
        # Vérifier si le symptôme est valide avant de poser la question
        if symptom.strip():
            readable_symptom = symptom.replace('_', ' ')  # Rendre lisible
            response = input(f"Do you also have {readable_symptom}? (yes/no): ").strip().lower()
            if response == "yes":
                confirmed_symptoms.add(symptom)  # Ajouter si l'utilisateur répond "oui"
        else:
            logging.warning("Un symptôme vide a été détecté dans la liste des symptômes manquants.")
    
    return confirmed_symptoms



# Fonction pour prédire la maladie
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

# Fonction pour sauvegarder le rapport de l'utilisateur
def save_user_report(user_symptoms, confirmed_symptoms, predicted_disease, raw_symptoms):
    # Créer ou mettre à jour un fichier de rapport unique
    os.makedirs("user_reports", exist_ok=True)
    report_filename = "user_reports/user_symptoms_report.txt"

    # Définir le format du rapport
    report_content = f"""
User Symptoms Report
===================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symptoms provided by user: {', '.join(raw_symptoms)}
Validated symptoms: {', '.join(user_symptoms)}
Confirmed symptoms: {', '.join(confirmed_symptoms)}
Predicted disease: {predicted_disease}
"""
    # Ajouter le rapport au fichier existant ou créer un nouveau fichier
    with open(report_filename, "a") as report_file:
        report_file.write(report_content)
        report_file.write("\n" + "="*50 + "\n")

# Programme principal
if __name__ == "__main__":
    logging.info("Welcome to the disease prediction system based on your symptoms.")
    
    # Étape 1 : Récupérer les symptômes de l'utilisateur
    user_message = input("Please describe your symptoms in a sentence: ").strip()
    if not user_message:
        logging.warning("Message vide. Veuillez décrire vos symptômes clairement.")
    else:
        # Extraire les symptômes
        valid_symptoms, raw_symptoms = extract_symptoms_from_message(user_message)
        
        # Logs pour vérifier les symptômes extraits
        logging.debug("Symptômes extraits bruts : %s", raw_symptoms)
        logging.debug("Symptômes validés contre la base : %s", valid_symptoms)

        if valid_symptoms:
            predicted_disease = predict_disease(valid_symptoms)
            disease_symptoms = get_disease_symptoms(predicted_disease)

            # Logs pour la maladie prédite
            logging.debug("Maladie prédite : %s", predicted_disease)
            logging.debug("Symptômes associés à la maladie prédite : %s", disease_symptoms)

            # Calcul initial du pourcentage de correspondance
            match_percentage = len(set(valid_symptoms).intersection(disease_symptoms)) / len(disease_symptoms)
            
            if match_percentage >= 0.75:
                logging.info("Predicted disease: %s", predicted_disease)
                save_user_report(valid_symptoms, valid_symptoms, predicted_disease, raw_symptoms)
            else:
                logging.info("Let's confirm some additional symptoms.")
                confirmed_symptoms = confirm_missing_symptoms(valid_symptoms, disease_symptoms)
                final_match_percentage = len(set(confirmed_symptoms).intersection(disease_symptoms)) / len(disease_symptoms)
                
                if final_match_percentage >= 0.75:
                    logging.info("The preconsultation is done, thank you.")
                    save_user_report(valid_symptoms, confirmed_symptoms, predicted_disease, raw_symptoms)
                else:
                    logging.warning("Unable to confirm sufficient symptoms for a reliable prediction. Please consult a healthcare professional.")
        else:
            logging.warning("No symptoms detected. Please rephrase your message with clear symptom descriptions.")


 