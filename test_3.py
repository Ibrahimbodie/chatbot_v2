import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder
import subprocess
import re
from datetime import datetime
import os


# Charger le dataset
data_cleaned = pd.read_csv('data/dataset_cleaned.csv')  # Remplace par le bon chemin

# Définir les maladies et les symptômes connus
diseases = data_cleaned['Disease'].unique()
known_symptoms = set(data_cleaned['Symptoms'].str.strip().str.lower().str.split(",").sum())

# Seuil de validation pour la correspondance des symptômes
VALIDATION_THRESHOLD = 0.75

# Initialiser le modèle et le tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=len(diseases))
model.load_state_dict(torch.load('model_biobert.pth', map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

# Encoder les étiquettes de maladie
label_encoder = LabelEncoder()
label_encoder.fit(data_cleaned['Disease'])

# Fonction de normalisation des symptômes
def normalize_symptom(symptom):
    return symptom.strip().replace(" ", "_").lower()

# Fonction pour extraire les symptômes du message utilisateur
def extract_symptoms_from_message(message):
    valid_symptoms=[]
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
    confirmed_symptoms = set(user_symptoms)  # Inclure les symptômes initiaux

    # Poser des questions sur les symptômes manquants
    for symptom in missing_symptoms:
        response = input(f"Do you also have {symptom.replace('_', ' ')}? (yes/no): ").strip().lower()
        if response == "yes":
            confirmed_symptoms.add(symptom)  # Ajouter si l'utilisateur répond "oui"
    
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
def save_user_report(user_symptoms, confirmed_symptoms, predicted_disease):
    # Créer ou mettre à jour un fichier de rapport unique
    report_filename = "user_reports/user_symptoms_report.txt"

    # Définir le format du rapport
    report_content = f"""
User Symptoms Report
===================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symptoms provided by user: {', '.join(user_symptoms)}
Confirmed symptoms: {', '.join(confirmed_symptoms)}
Predicted disease: {predicted_disease}
"""
    
    # Ajouter le rapport au fichier existant ou créer un nouveau fichier
    with open(report_filename, "a") as report_file:
        report_file.write(report_content)
        report_file.write("\n" + "="*50 + "\n")

        
# Programme principal
if __name__ == "__main__":
    print("Welcome to the disease prediction system based on your symptoms.")
    
    # Étape 1 : Récupérer les symptômes de l'utilisateur
    user_message = input("Please describe your symptoms in a sentence: ")
    symptoms_list = extract_symptoms_from_message(user_message)
    
    confirmed_symptoms = symptoms_list  # Initialiser la liste des symptômes confirmés  # Initialiser la liste des symptômes non confirmés
    
    if symptoms_list:
        # Étape 2 : Prédire la maladie initiale
        predicted_disease = predict_disease(symptoms_list)
        disease_symptoms = get_disease_symptoms(predicted_disease)
        
        # Calcul initial du pourcentage de correspondance
        match_percentage = len(set(symptoms_list).intersection(disease_symptoms)) / len(disease_symptoms)
        
        if match_percentage >= VALIDATION_THRESHOLD:
            print(f"Predicted disease: {predicted_disease}")
            # Sauvegarder le rapport avec les symptômes confirmés = symptômes initiaux
            save_user_report(symptoms_list, symptoms_list, predicted_disease)
        else:
            print(" Let's confirm some additional symptoms.")
            
            # Étape 3 : Confirmer les symptômes manquants
            confirmed_symptoms = confirm_missing_symptoms(symptoms_list, disease_symptoms)
                        
            # Recalculer la correspondance avec tous les symptômes confirmés
            final_match_percentage = len(set(confirmed_symptoms).intersection(disease_symptoms)) / len(disease_symptoms)
            
            # Vérifier la correspondance mise à jour
            if final_match_percentage >= VALIDATION_THRESHOLD:
                print("The preconsultation is done, thank you.")
                # Sauvegarder le rapport après confirmation
                save_user_report(symptoms_list, confirmed_symptoms, predicted_disease)
            else:
                print("Unable to confirm sufficient symptoms for a reliable prediction. Please consult a healthcare professional.")
    else:
        print("No symptoms detected. Please rephrase your message with clear symptom descriptions.")
    
    print("Analysis completed.")