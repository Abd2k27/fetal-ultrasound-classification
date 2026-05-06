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

### 📊 Analyse d'Erreurs / Error Analysis
Une fois le modèle entraîné (`best_model.pth` généré) :
```bash
jupyter notebook notebooks/01_Error_Analysis_and_Explainability.ipynb
```

---
*Created for portfolio purposes to demonstrate expertise in Medical Computer Vision.*
