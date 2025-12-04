# Tutoriel

Ce guide a pour but de vous apprendre à utiliser ce code lié à l'Eye-Tracker à travers un scénario concret. Nous allons partir de données brutes pour arriver à des vidéos/photos analysables.

## 1. Le Scénario

Imaginez que vous avez fait utiliser l'Eye-Tracker à un artisan pendant qu'il travaillait.
Vous avez maintenant deux fichiers sur votre ordinateur produits par l'Eye-Tracker:

- données_artisan.tsv : Le fichier rempli de données (coordonnées X, Y, temps...).

- video_artisan.mp4 : La vidéo de ce que voyait l'artisan.

**Objectif :** Nous voulons voir exactement où l'artisan regardait entre la 5ème et la 15ème seconde à travers une série d'images.

## 2. Lancement de l'application

Dans un premier temps, assurez-vous d'avoir bien installé les dépendances via Poetry. Si besoin, se référer au `README.md`.

Ouvrez votre terminal (ou invite de commandes) dans le dossier du projet ProTech.

Lancez l'application avec Poetry :

`poetry run python protech/code.py`

L'interface graphique s'ouvre. Pas de panique, c'est très simple.

## 3. Configuration pas-à-pas

### **Étape A :** Importer les sources

Dans le cadre "1. Fichiers" :

Cliquez sur le bouton "Choisir TSV". Naviguez dans vos dossiers et sélectionnez données_artisan.tsv.

Cliquez sur le bouton "Choisir MP4". Sélectionnez vidéo_artisan.mp4.

### **Étape B :** Choisir le rendu

Dans le cadre "2. Action", nous voulons une série de photos.

Cochez l'option : "4. Images avec point".

**Note :** L'option 1 ne ferait que couper la vidéo sans rien dessiner dessus. L'option 2 dessinerait le tracé du regard sur l'extrait. L'option 3 renverrait une série d'images avec le chemin du regard.

### **Étape C :** Définir les paramètres

Dans le cadre "3. Paramètres", nous allons cibler notre moment clé.

Début (s) : Tapez 5

Fin (s) : Tapez 15

Nom Dossier : Donnez un nom clair, par exemple Analyse_artisan1.

**Important :** C'est le nom du dossier qui sera créé pour ranger tous les résultats.

Sélectionnez si vous souhaitez indiquer le nombre d'images ou la fréquence puis ajoutez en dessous la valeur correspondante.

Cochez si vous voulez que l'image de la vidéo soit en fond ou si vous souhaitez un fond noir.

## 4. Génération et Résultat

Cliquez sur le grand bouton ▶ LANCER en bas.

Regardez la zone "Journal" en bas : vous verrez des messages comme "Extrait brut créé" indiquant l'avancée du programme.

Une fois que le message "Succès !" apparaît :

Ouvrez votre explorateur de fichiers.

Allez dans le nouveau dossier Analyse_artisan1.

Vous trouverez les images créées dans l'ordre (frame_001, frame_002,...).

Bravo ! Vous avez généré vos images et pouvez maintenant analyser le regard de votre artisan.

Pour arrêter le programme, fermez simplement l'interface.

## Aller plus loin :

Si vous voulez d'autres fichiers exploitables, en format vidéo ou sur une autre plage de temps par exemple, pas besoin de quitter l'interface ! Vous changez les paramètres que vous souhaitez et recliquez sur le bouton ▶ LANCER.
