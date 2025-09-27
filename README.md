# 🩺 Medical Bot v2 – Système de Préconsultation Médicale

Un assistant en ligne de commande qui aide à la **préconsultation médicale**.
Il extrait automatiquement les symptômes décrits par un utilisateur, identifie les maladies candidates à l’aide de **BioBERT**, et génère un rapport détaillé des symptômes confirmés/rejetés.

---

## 🚀 Fonctionnalités

* ✨ Extraction automatique des symptômes à partir du texte utilisateur.
* 🧠 Utilisation du modèle **BioBERT** pour la classification des maladies.
* 🔍 Validation interactive des symptômes manquants ou discriminants.
* 📊 Prédiction de la maladie la plus probable avec un score de similarité.
* 📝 Génération d’un **rapport utilisateur** contenant :

  * Symptômes fournis
  * Symptômes confirmés / rejetés
  * Maladie prédite
  * Date et historique

---

## ⚙️ Installation

1. **Cloner le dépôt :**

```bash
git clone https://github.com/votre-utilisateur/medical-bot-v2.git
cd medical-bot-v2
```

2. **Installer les dépendances :**

```bash
pip install -r requirements.txt
```

3. **Préparer les fichiers nécessaires :**

   * `data/dataset_cleaned.csv` (dataset contenant les maladies et symptômes)
   * `model_biobert.pth` (poids entraînés du modèle BioBERT)

---

## ▶️ Utilisation

Exécuter le programme :

```bash
python medical_bot_v2.py
```

Exemple :

```
Welcome to the disease prediction system based on your symptoms.
Please describe your symptoms in a sentence: I have fever and headache
```

Le bot :

1. Extrait les symptômes.
2. Compare avec les maladies connues.
3. Pose des questions de confirmation.
4. Propose la maladie la plus probable.
5. Sauvegarde un rapport dans `user_reports/all_reports.txt`.

---

## 🛠️ Technologies

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [Transformers (HuggingFace)](https://huggingface.co/transformers/)
* [BioBERT](https://github.com/dmis-lab/biobert)
* [scikit-learn](https://scikit-learn.org/)
* [pandas](https://pandas.pydata.org/)

---

## 📌 Auteur

Projet développé par **Ibrahima Diallo**.
