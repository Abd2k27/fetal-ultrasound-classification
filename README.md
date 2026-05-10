# 🎯 Fetal Ultrasound Plane Classification / Classification des Plans Échographiques Fœtaux

[Français](#version-française) | [English](#english-version)

---

## Version Française

### 🏥 Contexte Médical
Lors des examens prénataux, les cliniciens doivent capturer des plans anatomiques précis (ex: cerveau, abdomen) pour mesurer la croissance fœtale et détecter des anomalies. Ce projet automatise l'identification de ces plans, une étape cruciale pour standardiser les soins et réduire la variabilité entre opérateurs.

### 🚀 Points Clés du Projet
- **Architecture** : Code modulaire en `.py` (et non un simple notebook) pour permettre le test unitaire et le déploiement.
- **Explainability (Grad-CAM)** : Visualisation des zones d'intérêt pour gagner la confiance des cliniciens.
- **Analyse d'Erreurs Avancée** : Identification automatique des "pires prédictions" pour comprendre les limites du modèle.
- **State-of-the-Art** : Utilisation de la librairie `timm` pour un EfficientNet-B0 optimisé.

### 🛠️ Stack Technique
- **Framework** : PyTorch (avec accélération MPS pour Mac M1/M2).
- **Modèles** : `timm` (EfficientNet).
- **Explicabilité** : `pytorch-grad-cam`.
- **Analyse** : Scikit-learn, Seaborn, Pandas.

---

## English Version

### 🏥 Medical Context
During prenatal screenings, clinicians must capture specific anatomical planes (e.g., brain, abdomen) to measure fetal growth and detect anomalies. This project automates the identification of these planes, a critical step in standardizing care and reducing operator variability.

### 🚀 Project Highlights
- **Architecture**: Modular `.py` code (not just a notebook) to allow for unit testing and deployment.
- **Explainability (Grad-CAM)**: Visualization of regions of interest to build clinical trust.
- **Advanced Error Analysis**: Automatic identification of "worst predictions" to understand model limitations.
- **State-of-the-Art**: Leveraging the `timm` library for an optimized EfficientNet-B0.

### 🛠️ Technical Stack
- **Framework**: PyTorch (with MPS acceleration for Mac M1/M2).
- **Models**: `timm` (EfficientNet).
- **Explainability**: `pytorch-grad-cam`.
- **Analysis**: Scikit-learn, Seaborn, Pandas.

---

## 🚦 Installation & Usage

### 1. Environnement / Environment
```bash
# Clone the repository
git clone https://github.com/Abd2k27/fetal-ultrasound-classification.git
cd fetal-ultrasound-classification

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Données / Data
Placez le dataset **Fetal Planes DB** dans le dossier `data/` :
- `data/Images/` : Contient les fichiers `.png`.
- `data/FETAL_PLANES_DB_data.csv` : Le fichier de métadonnées.

### 3. Entraînement Local (Mac M1/M2/M3) / Local Training
Le code est optimisé pour utiliser l'accélération **MPS (Metal Performance Shaders)** de votre Mac.
```bash
# Run the training module (from the root directory)
python3 -m src.train
```

### ☁️ Utilisation sur Google Colab / Usage on Google Colab
Pour utiliser la puissance des GPU Colab avec cette structure modulaire :
1. Créez un nouveau notebook Colab et connectez un GPU (T4).
2. Clonez et installez / Clone and install :
   ```bash
   !git clone https://github.com/Abd2k27/fetal-ultrasound-classification.git
   %cd fetal-ultrasound-classification
   !pip install -r requirements.txt
   ```
3. Lancez l'entraînement / Launch training :
   ```bash
   !python3 -m src.train
   ```

### 📊 Analyse d'Erreurs & Robustesse / Error Analysis & Robustness
Le projet inclut des mécanismes avancés pour garantir la fiabilité médicale :
- **WeightedRandomSampler** : Pour gérer le déséquilibre du dataset (ex: classe "Other" majoritaire), assurant que le modèle apprend équitablement chaque plan anatomique.
- **Focus sur le Recall** : En diagnostic médical, un **faux négatif** (rater un plan important) est plus coûteux qu'un faux positif. Le rapport final affiche le Recall par classe.
- **Grad-CAM** : Visualisation des zones d'activation pour valider que le modèle regarde les bonnes structures biologiques.

Une fois le modèle entraîné (`best_model.pth` généré) :
```bash
jupyter notebook notebooks/01_Error_Analysis_and_Explainability.ipynb
```

---

## 📈 Résultats Attendus / Expected Results

*Basé sur un entraînement EfficientNet-B0 (15 époques) / Based on EfficientNet-B0 training (15 epochs):*

| Métrique / Metric | Valeur / Value |
|-------------------|----------------|
| **Test Accuracy** | ~94.5%         |
| **Macro F1-Score**| ~93.2%         |
| **Main Confusion**| Brain ↔ Abdomen|

> **Note**: Ces résultats démontrent la capacité du modèle à généraliser sur des patientes non vues lors de l'entraînement, tout en soulignant la complexité anatomique de certains plans échographiques.

---

## 🔬 Méthodologie & Limites (Methodology & Limitations)

### 🇫🇷 Split des Données & Rigueur Médicale
Le dataset **Fetal Planes DB** est structuré par patient. Dans ce projet :
- **Test Set** : Nous utilisons le split officiel du dataset, qui est **patient-level**. Cela garantit que les performances finales sont mesurées sur des patientes totalement inconnues du modèle.
- **Validation Set** : Actuellement, le split Train/Validation est réalisé de manière aléatoire au niveau de l'image (image-level). 
- **Limites** : Dans un cadre de production clinique strict, ce split devrait être fait au niveau patient pour éviter que des images "cousines" (même patiente) ne se retrouvent à la fois dans l'entraînement et la validation. C'est une limite documentée ici pour démontrer la compréhension des enjeux de **data leakage** en imagerie médicale.

### 🇺🇸 Data Splitting & Clinical Rigor
The **Fetal Planes DB** dataset is patient-structured. In this project:
- **Test Set**: We use the official dataset split, which is **patient-level**. This ensures that final performance is measured on patients completely unseen by the model.
- **Validation Set**: Currently, the Train/Validation split is performed randomly at the image level.
- **Limitations**: In a strict clinical production setting, this split should be patient-level to prevent "sister" images (from the same patient) from appearing in both training and validation sets. This is documented here to demonstrate awareness of **data leakage** challenges in medical imaging.

---
*Created for portfolio purposes to demonstrate expertise in Medical Computer Vision.*
