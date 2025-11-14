import openpyxl
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from moviepy import VideoFileClip, TextClip, CompositeVideoClip, VideoClip
import pandas as pd
import csv
from scipy.interpolate import interp1d
import cv2
import numpy as np
import os
import argparse

#Extrait les données du fichier csv par colonne
def Colonne(fichier, nom_Colonne):
    C = []
    with open(fichier, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            C.append(row[nom_Colonne])
    return C


#Garde uniquement les données utiles (les points de fixation)
def Nettoyage (gaze2dx,gaze2dy,MediaTimeStamp, gaze_class):
    fixations_idx = [i for i, gclass in enumerate(gaze_class) if gclass == 'F']
    fixation_gaze2dx = [gaze2dx[i] for i in fixations_idx]
    fixation_gaze2dy = [gaze2dy[i] for i in fixations_idx]
    fixation_time_stamp = [MediaTimeStamp[i] for i in fixations_idx]
    gaze2dx_filtered = []
    gaze2dy_filtered = []
    time_stamp_filtered = []
    for x, y, t in zip(fixation_gaze2dx, fixation_gaze2dy, fixation_time_stamp):
        try:
            gaze2dx_filtered.append(float(x))
            gaze2dy_filtered.append(float(y))
            time_stamp_filtered.append(float(t))
        except ValueError:
            pass
    return gaze2dx_filtered, gaze2dy_filtered, time_stamp_filtered


#Trouve l'indice de la première valeur supérieure à la valeur donnée 
def premier_indice(lst, value):
    for i, v in enumerate(lst):
        if float(v) >= value:
            return i
    return None  


#Trouve l'indice de la dernière valeur inférieure à la valeur donnée 
def dernier_indice(lst, value):
    for i in reversed(range(len(lst))):
        if float(lst[i]) <= value:
            return i
    return None 


#Prend un extrait d'une vidéo selon les timecodes donnés
def prendre_extrait (video,t1,t2,nom_extrait):
    clip = VideoFileClip(video).subclipped(t1,t2)
    clip.write_videofile(nom_extrait+".mp4")


#Extrait les données correspondant à un extrait de vidéo entre t1 et t2
def extrait_donnees (t1, t2, x_filtered, y_filtered, time_filtered) :
    t1_precis = premier_indice(time_filtered,t1)
    t2_precis = dernier_indice(time_filtered,t2)

    x_extrait = x_filtered [t1_precis:t2_precis+1]
    y_extrait = y_filtered [t1_precis:t2_precis+1]
    time_extrait = time_filtered [t1_precis:t2_precis+1]

    return x_extrait, y_extrait, time_extrait


#Trace le chemin du regard pour un extrait de vidéo entre t1 et t2
def video_chemin (video, t1, t2, nom_extrait, x_filtered, y_filtered, time_filtered) :
    prendre_extrait (video, t1, t2, nom_extrait)
    x, y, time_extrait = extrait_donnees(t1, t2, x_filtered, y_filtered, time_filtered)
    time = [tt - t1 for tt in time_extrait] 

    clip = VideoFileClip(nom_extrait + ".mp4")

    current_idx = [0]

    # Dessine, pour un instant t donné, l'image vidéo correspondante avec le tracé précédent complet et le point de fixation actuel.
    def make_frame(t):
        frame = clip.get_frame(t)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        while current_idx[0] + 1 < len(time) and time[current_idx[0] + 1] <= t:
            current_idx[0] += 1

        if t < time[0]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

        points_to_draw = [(int(x[i]), int(y[i])) for i in range(current_idx[0] + 1)]

        for i in range(1, len(points_to_draw)):
            alpha = i / len(points_to_draw)
            g = int(255 * alpha)
            cv2.line(frame, points_to_draw[i-1], points_to_draw[i], (255, g, 0), 4)

        x_t, y_t = int(x[(current_idx[0])]), int(y[current_idx[0]])
        cv2.circle(frame, (x_t, y_t), 8, (255, 255, 0), -1)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    trace_clip = VideoClip(make_frame, duration=clip.duration).with_fps(clip.fps)
    final = CompositeVideoClip([clip, trace_clip])
    final.write_videofile(nom_extrait + "_chemin.mp4", fps=clip.fps)


#Génère le nombre d'images demandé entre t1 et t2 avec le tracé du regard jusqu'à l'instant t de l'image
def generer_images_parcours(video_source, t1, t2, nb_frames, nom_base_images, x_filtered, y_filtered, time_filtered):

    os.makedirs(nom_base_images)

    nom_extrait = f"{nom_base_images}_clip"
    prendre_extrait(video_source, t1, t2, nom_extrait)
    clip = VideoFileClip(nom_extrait + ".mp4")

    x, y, time_extrait = extrait_donnees(t1, t2, x_filtered, y_filtered, time_filtered)

    time = [tt - t1 for tt in time_extrait]

    duree_clip = clip.duration 

    duree_securisee = duree_clip - 0.5

    if duree_securisee < 0:
        duree_securisee = 0

    timestamps_images = np.linspace(0, duree_securisee, nb_frames, endpoint=True)

    data_idx_pointer = 0

    for i, t_frame in enumerate(timestamps_images):
        t_frame = timestamps_images[i] 

        frame = clip.get_frame(t_frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        current_data_idx = -1
        while data_idx_pointer < len(time) and time[data_idx_pointer] <= t_frame:
            current_data_idx = data_idx_pointer 
            data_idx_pointer += 1 

        if current_data_idx == -1 or t_frame < time[0]:
            nom_image = os.path.join(nom_base_images, f"frame_{i+1:03d}.png")
            cv2.imwrite(nom_image, frame)
            print(f"Image sauvegardée (avant data): {nom_image} (t={t_frame:.2f}s)")
            continue 

        points_to_draw = [(int(x[j]), int(y[j])) for j in range(current_data_idx + 1)]

        if len(points_to_draw) > 1:
            for j in range(1, len(points_to_draw)):
                alpha = j / len(points_to_draw)
                g = int(255 * alpha) 
                cv2.line(frame, points_to_draw[j-1], points_to_draw[j], (255, g, 0), 4)

        x_t, y_t = points_to_draw[-1]
        cv2.circle(frame, (x_t, y_t), 8, (255, 255, 0), -1)

        nom_image = os.path.join(nom_base_images, f"frame_{i+1:03d}.png")

        cv2.imwrite(nom_image, frame)
        print(f"Image sauvegardée : {nom_image} (correspondant à t={t_frame:.2f}s)")

    clip.close()

    print(f"\nTerminé. {nb_frames} images ont été générées avec le préfixe '{nom_base_images}'.")


#Génère le nombre d'images demandé entre t1 et t2 avec le point du regard à l'instant t de l'image
def generer_images_points(video_source, t1, t2, nb_frames, nom_base_images, x_filtered, y_filtered, time_filtered):

    os.makedirs(nom_base_images)

    nom_extrait = f"{nom_base_images}_clip"
    prendre_extrait(video_source, t1, t2, nom_extrait)
    clip = VideoFileClip(nom_extrait + ".mp4")

    x, y, time_extrait = extrait_donnees(t1, t2, x_filtered, y_filtered, time_filtered)

    time = [tt - t1 for tt in time_extrait]

    duree_clip = clip.duration 

    duree_securisee = duree_clip - 0.5

    if duree_securisee < 0:
        duree_securisee = 0

    timestamps_images = np.linspace(0, duree_securisee, nb_frames, endpoint=True)

    data_idx_pointer = 0

    for i, t_frame in enumerate(timestamps_images):
        t_frame = timestamps_images[i] 

        frame = clip.get_frame(t_frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        current_data_idx = -1
        while data_idx_pointer < len(time) and time[data_idx_pointer] <= t_frame:
            current_data_idx = data_idx_pointer 
            data_idx_pointer += 1 

        if current_data_idx == -1 or t_frame < time[0]:
            nom_image = os.path.join(nom_base_images, f"frame_{i+1:03d}.png")
            cv2.imwrite(nom_image, frame)
            print(f"Image sauvegardée (avant data): {nom_image} (t={t_frame:.2f}s)")
            continue

        x_t = int(x[current_data_idx])
        y_t = int(y[current_data_idx])

        cv2.circle(frame, (x_t, y_t), 8, (255, 255, 0), -1)

        nom_image = os.path.join(nom_base_images, f"frame_{i+1:03d}.png")

        cv2.imwrite(nom_image, frame)
        print(f"Image sauvegardée : {nom_image} (correspondant à t={t_frame:.2f}s)")

    clip.close()

    print(f"\nTerminé. {nb_frames} images ont été générées avec le préfixe '{nom_base_images}'.")

#Demande un float à l'utilisateur
def ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Erreur : Veuillez entrer un nombre (ex: 5.0).")

#Demande un int à l'utilisateur
def ask_int(prompt, default=None):
    while True:
        val = input(prompt)
        if not val and default is not None:
            return default
        try:
            return int(val)
        except ValueError:
            print("Erreur : Veuillez entrer un nombre entier.")

#Demande une chaîne non vide
def ask_string(prompt):
    while True:
        val = input(prompt)
        if val:
            return val
        print("Erreur : Ce champ ne peut pas être vide.")


def main():
    parser = argparse.ArgumentParser(description="Outil interactif d'analyse Eye-Tracking ProTech.")
    
    parser.add_argument("tsv_file", help="Chemin vers le fichier de données .tsv")
    parser.add_argument("video_file", help="Chemin vers le fichier vidéo .mp4")
    
    args = parser.parse_args()

    fichier_tsv = args.tsv_file
    video_path = args.video_file

    # 2. VÉRIFICATION DES FICHIERS
    if not os.path.exists(fichier_tsv):
        print(f"Erreur: Le fichier TSV '{fichier_tsv}' est introuvable.")
        return
    if not os.path.exists(video_path):
        print(f"Erreur: Le fichier vidéo '{video_path}' est introuvable.")
        return

    print(f"Fichier TSV chargé : {fichier_tsv}")
    print(f"Fichier Vidéo chargé : {video_path}")

    # 3. MENU INTERACTIF
    print("\n--- Menu ProTech ---")
    print("Quelle action souhaitez-vous effectuer ?")
    print("  1: Extraire un clip vidéo (extrait)")
    print("  2: Générer une vidéo avec tracé (video_chemin)")
    print("  3: Générer des images avec tracé (images_parcours)")
    print("  4: Générer des images avec points (images_points)")

    action_choice = ""
    while action_choice not in ["1", "2", "3", "4"]:
        action_choice = input("Votre choix (1-4) : ")

    actions_map = {
        "1": "extrait",
        "2": "video_chemin",
        "3": "images_parcours",
        "4": "images_points"
    }
    action = actions_map[action_choice]

    # 4. DEMANDE DES PARAMÈTRES
    print("\nVeuillez entrer les paramètres :")
    t_start = ask_float("  Temps de début (en secondes) : ")
    t_end = ask_float("  Temps de fin (en secondes) : ")
    output_name = ask_string("  Nom du fichier/dossier de sortie : ")
    
    nb_frames = 10 # Valeur par défaut
    if action in ["images_parcours", "images_points"]:
        nb_frames = ask_int(f"  Nombre d'images (par défaut: {nb_frames}) : ", default=nb_frames)

    if action == "extrait":
        print(f"\nLancement de l'action : {action}...")
        prendre_extrait(video_path, t_start, t_end, output_name)
        print("Terminé.")
        return

    # Pour les autres actions, on charge et nettoie les données
    print(f"\nTraitement du fichier TSV...")
    fichier_csv = os.path.splitext(fichier_tsv)[0] + ".csv"
    
    try:
        df = pd.read_csv(fichier_tsv, sep='\t', comment='#')
        df.to_csv(fichier_csv, index=False)
    except Exception as e:
        print(f"Erreur lors de la lecture ou conversion du TSV : {e}")
        return

    gaze2dx = Colonne(fichier_csv, 'Gaze2dX')
    gaze2dy = Colonne(fichier_csv, 'Gaze2dY')
    gaze_class = Colonne(fichier_csv, 'GazeClass')
    MediaTimeStamp = Colonne(fichier_csv, 'MediaTimeStamp')
    
    x_filtered, y_filtered, time_filtered = Nettoyage(gaze2dx, gaze2dy, MediaTimeStamp, gaze_class)

    # 6. EXÉCUTION DE L'ACTION DEMANDÉE
    print(f"\nLancement de l'action : {action}...")
    
    if action == "video_chemin":
        print(f"Génération de la vidéo avec tracé : {output_name}_chemin.mp4")
        video_chemin(video_path, t_start, t_end, output_name, x_filtered, y_filtered, time_filtered)

    elif action == "images_parcours":
        print(f"Génération de {nb_frames} images (parcours) dans le dossier : {output_name}")
        generer_images_parcours(video_path, t_start, t_end, nb_frames, output_name, x_filtered, y_filtered, time_filtered)

    elif action == "images_points":
        print(f"Génération de {nb_frames} images (points simples) dans le dossier : {output_name}")
        generer_images_points(video_path, t_start, t_end, nb_frames, output_name, x_filtered, y_filtered, time_filtered)

    print("Traitement terminé.")

if __name__ == "__main__":
    main()
