# Projet : Analyseur de Données d'Eye-Tracking (ProTech)

Ce projet utilise Poetry comme outil de build moderne pour Python.

Il analyse des données d'eye-tracking (fichiers .tsv) et les synchronise avec un fichier vidéo (.mp4) pour générer des visualisations (vidéos tracées ou extraction d'images).

### Prérequis : 

- Python (version 3.10+)
- Poetry 

## Note sur les Fichiers de Données 

Les fichiers de données sources (vidéo .mp4 et data.tsv) ne sont pas inclus dans ce dépôt. Il faut les placer dans le même dossier que le code.

## Installation des Dépendances : 

Cette commande va lire le fichier pyproject.toml, créer un environnement virtuel isolé, et installer automatiquement toutes les dépendances (pandas, moviepy, opencv, etc.).

### Placez-vous dans le dossier du projet : 
`cd [chemin_vers_votre_dossier]`


### Lancez l'installation :
` poetry install `


## Exécution du Projet :

Le script ProTech.py s'exécute via poetry run et nécessite des arguments pour choisir l'action à effectuer.

**Syntaxe générale :** `poetry run python code.py [ACTION] --start [DEBUT] --end [FIN] --output [NOM_SORTIE]`

--start (ou -s) : Temps de début en secondes (ex: 5.0).

--end (ou -e) : Temps de fin en secondes (ex: 10.0).

--output (ou -o) : Nom du fichier ou dossier de sortie (sans extension).

**Exemples de commandes :**

- Générer une vidéo avec le tracé du regard
Crée un extrait vidéo où le chemin parcouru par le regard est dessiné progressivement.

    ex : `poetry run python code.py video_chemin -s 5 -e 10 -o resultat_video`


- Générer une séquence d'images (Parcours)
Extrait des images fixes où le tracé du regard est visible. L'option -n définit le nombre d'images.

    ex : `poetry run python code.py images_parcours -s 10 -e 15 -o dossier_images -n 20`


- Générer une séquence d'images (Points simples)
Extrait des images avec seulement le point de fixation actuel (sans l'historique du tracé).
    
    ex : `poetry run python code.py images_points -s 10 -e 15 -o dossier_points -n 10`


- Simplement extraire un clip vidéo
Coupe la vidéo sans ajouter d'analyse de données.

    ex : `poetry run python code.py extrait -s 2 -e 5 -o mon_clip`  



**Astuce :** Pour voir l'aide complète dans le terminal, tapez :
`poetry run python code.py --help`