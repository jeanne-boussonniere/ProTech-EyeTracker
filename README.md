# Projet : Analyseur de Données d'Eye-Tracking (ProTech)

Ce projet utilise Poetry comme outil de build moderne pour Python.

Il analyse des données d'eye-tracking (fichiers .tsv) et les synchronise avec un fichier vidéo (.mp4) pour générer des visualisations (vidéos tracées ou extraction d'images).

### Prérequis : 

- Python (version 3.10+)
- Poetry 

## Note sur les Fichiers de Données 

Les fichiers de données sources (vidéo .mp4 et data.tsv) ne sont pas inclus dans ce dépôt. Il faut les donner en arguments lors de l'exécution du code.

## Installation des Dépendances : 

Cette commande va lire le fichier pyproject.toml, créer un environnement virtuel isolé, et installer automatiquement toutes les dépendances (pandas, moviepy, opencv, etc.).

### Placez-vous dans le dossier du projet : 
`cd [chemin_vers_votre_dossier]`


### Lancez l'installation :
` poetry install `


## Exécution du Projet :

Le script code.py s'exécute via poetry run et nécessite une vidéo ainsi qu'un fichier de données tsv en arguments.

**Étape 1 :** Lancer le script

Vous devez d'abord lancer le script en lui donnant les chemins vers vos deux fichiers de données :

poetry run python code.py [chemin/vers/fichier.tsv] [chemin/vers/video.mp4]


Exemple : `poetry run python code.py Fichiers/données1.tsv Fichiers/vidéo1.mp4`


**Étape 2 :** Suivre le menu interactif

Une fois le script lancé, il va charger les fichiers et vous présenter un menu :

--- Menu ProTech ---  
Quelle action souhaitez-vous effectuer ?    
  1 Extraire un clip vidéo (extrait)  
  2: Générer une vidéo avec tracé (video_chemin)  
  3: Générer des images avec tracé (images_parcours)  
  4: Générer des images avec points (images_points)  
Votre choix (1-4) :

Suivez simplement les instructions pour choisir une action, définir les temps de début et de fin, et nommer votre fichier de sortie.

**Astuce :** Pour voir l'aide complète dans le terminal, tapez :
`poetry run python code.py --help`