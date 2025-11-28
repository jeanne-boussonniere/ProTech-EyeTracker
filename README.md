# Projet : Analyseur de Données d'Eye-Tracking (ProTech)

Ce projet utilise Poetry comme outil de build moderne pour Python.

Il analyse des données d'eye-tracking (fichiers .tsv) et les synchronise avec un fichier vidéo (.mp4) pour générer des visualisations (vidéos tracées ou extraction d'images).

### Prérequis : 

- Python (version 3.10+)
- Poetry 

## Note sur les Fichiers de Données 

Les fichiers de données sources (vidéo .mp4 et data.tsv) ne sont pas inclus dans ce dépôt. Vous devez vous assurer qu'ils sont stockés localement sur votre ordinateur avant de pouvoir les sélectionner via l'interface.

## Installation des Dépendances : 

Cette commande va lire le fichier pyproject.toml, créer un environnement virtuel isolé, et installer automatiquement toutes les dépendances (pandas, moviepy, opencv, etc.).

### Placez-vous dans le dossier du projet dans votre terminal : 
`cd [chemin_vers_votre_dossier]`


### Lancez l'installation :
` poetry install `


## Exécution du Projet :

Le script code.py s'exécute via poetry run.

**Étape 1 :** Lancer le script

Vous devez d'abord lancer le script :

`poetry run python code.py `


**Étape 2 :** Suivre le menu interactif

Une fois le script lancé, il va charger les fichiers et vous présenter une interface :

Suivez simplement les instructions pour choisir vos fichiers, une action, définir les temps de début et de fin, et nommer votre fichier de sortie.

