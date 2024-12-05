from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Définir l'utilisation du GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALIDATION = 0.75

# Définir le nombre de classes (maladies) que le modèle doit prédire
num_labels = 41  # Remplacez par le nombre correct de classes

# Charger le modèle et le tokenizer
model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=num_labels)
model.load_state_dict(torch.load('model_biobert.pth'))  # Charger les poids sauvegardés
model.to(device)  # Déplacer le modèle sur le GPU
model.eval()  # Mettre le modèle en mode évaluation

# Charger le tokenizer BioBERT
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

# Charger le label_encoder avec les étiquettes de maladies
data_cleaned = pd.read_csv('data/dataset_cleaned.csv')  # Remplacez par le chemin de votre dataset nettoyé
label_encoder = LabelEncoder()
label_encoder.fit(data_cleaned['Disease'])

# Obtenir une liste unique de tous les symptômes disponibles dans le dataset
unique_symptoms = set(data_cleaned['Symptoms'].str.split(",").sum())

# 1. Fonction pour récupérer la liste des symptômes de l'utilisateur
def collect_symptoms():
    symptoms_list = input("Entrez vos symptômes, séparés par des virgules: ").lower().split(",")
    symptoms_list = [symptom.strip() for symptom in symptoms_list]
    return symptoms_list

# 2. Fonction pour demander à l'utilisateur s'il souhaite ajouter d'autres symptômes
def ask_for_more_symptoms(symptoms_list):
    while True:
        more_symptoms = input("Avez-vous d'autres symptômes à ajouter ? (oui/non) : ").strip().lower()
        if more_symptoms == "oui":
            additional_symptoms = input("Entrez les nouveaux symptômes, séparés par des virgules : ").lower().split(",")
            additional_symptoms = [symptom.strip() for symptom in additional_symptoms]
            symptoms_list.extend(additional_symptoms)  # Ajout des nouveaux symptômes à la liste
        elif more_symptoms == "non":
            break
        else:
            print("Veuillez répondre par 'oui' ou 'non'.")
    return symptoms_list

# 3. Fonction pour récupérer les symptômes de la maladie prédite
def get_disease_symptoms(disease):
    # Rechercher la maladie dans le dataset et retourner ses symptômes
    disease_data = data_cleaned[data_cleaned['Disease'] == disease]
    if disease_data.empty:
        return []
    disease_symptoms = disease_data['Symptoms'].values[0].split(',')
    return [symptom.strip().lower() for symptom in disease_symptoms]

# 4. Fonction pour comparer les symptômes donnés par l'utilisateur avec ceux de la maladie prédite
def compare_symptoms(user_symptoms, disease_symptoms):
    matching_symptoms = set(user_symptoms).intersection(set(disease_symptoms))
    percentage_matched = len(matching_symptoms) / len(disease_symptoms)
    return percentage_matched

# 5. Fonction pour demander la confirmation des symptômes restants
def confirm_remaining_symptoms(remaining_symptoms):
    confirmed_symptoms = []
    
    for symptom in remaining_symptoms:
        confirmation = input(f"Avez-vous aussi {symptom}? (oui/non) : ").strip().lower()
        if confirmation == "oui":
            confirmed_symptoms.append(symptom)
    
    return confirmed_symptoms

# Fonction pour prédire la maladie en fonction des symptômes
def predict_disease(symptoms):
    # Tokenisation des symptômes
    tokens = tokenizer([symptoms], padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    
    # Déplacer les tokens sur le GPU
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    # Faire des prédictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Décoder les prédictions en étiquettes de maladies
    predicted_disease = label_encoder.inverse_transform(predictions.cpu().numpy())
    
    return predicted_disease[0]

#------------------------------------------------------------------------------------- Programme principal ajusté-----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Bienvenue dans le système de prédiction de maladies basé sur vos symptômes.")
    
    # 1. Collecter les symptômes de l'utilisateur
    symptoms_list = collect_symptoms()
    
    # Demander à l'utilisateur s'il a tous ses symptômes
    symptoms_list = ask_for_more_symptoms(symptoms_list)
    
    while True:  # Boucle pour continuer à poser des questions jusqu'à ce qu'une maladie soit trouvée
        # 2. Prédire la maladie
        predicted_disease = predict_disease(", ".join(symptoms_list))
        
        # 3. Récupérer les symptômes de la maladie prédite
        disease_symptoms = get_disease_symptoms(predicted_disease)
        
        # 4. Comparer les symptômes donnés par l'utilisateur avec ceux de la maladie prédite
        match_percentage = compare_symptoms(symptoms_list, disease_symptoms)
        
        if match_percentage >= VALIDATION:
            print(f"Maladie prédite : {predicted_disease}")
            break  # Si les symptômes correspondent à 75% ou plus, on arrête la boucle
        else:
            # 5. Demander à l'utilisateur la confirmation des symptômes restants
            remaining_symptoms = list(set(disease_symptoms) - set(symptoms_list))
            confirmed_symptoms = confirm_remaining_symptoms(remaining_symptoms)
            
            if not confirmed_symptoms:
                print("Aucun symptôme supplémentaire confirmé. Nous continuons l'analyse.")
            else:
                # Mise à jour des symptômes de l'utilisateur avec les réponses "oui"
                symptoms_list.extend(confirmed_symptoms)
    
    print("Analyse terminée.")