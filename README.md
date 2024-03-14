## Age Detection Project

Projet réalisé par Romane BARD, Vincent CORBAUX, Benjamin CHOMMAUX dans le cadre du projet « Intelligence artificielle pour l’estimation de l’âge basée sur le visage » INGE 3I

# Pour utiliser ce projet
- Cloner ce repository

# Les fichiers notebooks
- Le notebook MasterProject contient tout le code lié à l'entrainement des 6 modèles sur la base de données UTKFace et contient aussi le code pour l'Ensemble Learning
- Le notebook gnn_age_detection contient le code lié au GNN, son architecture, le traitement des images input, et l'entrainement du modèle
- Le notebook training_classes contient l'entrainement du modèle de classification ainsi que la cascade de prédiction


# Le dossier model
- Ce dossier contient tous les fichiers python permettant de construire les modèles de Deep Learning utilisés dans ce projet
  

# Le fichier ensemble_learning.py
- Ce fichier contient la methode afin de mettre en place la prédiction basée sur l'ensemble learning depuis les modèles pré entrainés

# Le fichier webcam.py
- Lancer ce fichier pour tester le projet
- La méthode actuellement utilisée dans ce fichier est la cascade du modèle de classification vers les modèles de régression

# Annexes
Dû à la taille des dossiers, vous trouverez les dossiers contenant les images d'entrainement ainsi que les modèles pré entrainés dans ce google drive:

https://drive.google.com/drive/folders/1_QFOkv_K8og10Lqu0GG19HwA6AZoiiDZ?usp=sharing