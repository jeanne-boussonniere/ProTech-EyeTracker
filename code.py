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

    os.makedirs("Images avec parcours")

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
            nom_image = os.path.join("Images avec parcours", f"frame_{i+1:03d}.png")
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

        nom_image = os.path.join("Images avec parcours", f"frame_{i+1:03d}.png")

        cv2.imwrite(nom_image, frame)
        print(f"Image sauvegardée : {nom_image} (correspondant à t={t_frame:.2f}s)")

    clip.close()

    print(f"\nTerminé. {nb_frames} images ont été générées avec le préfixe '{nom_base_images}'.")


#Génère le nombre d'images demandé entre t1 et t2 avec le point du regard à l'instant t de l'image
def generer_images_points(video_source, t1, t2, nb_frames, nom_base_images, x_filtered, y_filtered, time_filtered):

    os.makedirs("Images avec points")

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
            nom_image = os.path.join("Images avec points", f"frame_{i+1:03d}.png")
            cv2.imwrite(nom_image, frame)
            print(f"Image sauvegardée (avant data): {nom_image} (t={t_frame:.2f}s)")
            continue

        x_t = int(x[current_data_idx])
        y_t = int(y[current_data_idx])

        cv2.circle(frame, (x_t, y_t), 8, (255, 255, 0), -1)

        nom_image = os.path.join("Images avec points", f"frame_{i+1:03d}.png")

        cv2.imwrite(nom_image, frame)
        print(f"Image sauvegardée : {nom_image} (correspondant à t={t_frame:.2f}s)")

    clip.close()

    print(f"\nTerminé. {nb_frames} images ont été générées avec le préfixe '{nom_base_images}'.")


def main():
    parser = argparse.ArgumentParser(description="Outil d'analyse et de visualisation Eye-Tracking ProTech.")

    parser.add_argument(
        "action", 
        choices=["video_chemin", "images_parcours", "images_points", "extrait"],
        help="L'action à effectuer : générer une vidéo avec chemin, des images avec parcours, des images avec points, ou juste un extrait vidéo."
    )

    parser.add_argument("-s", "--start", type=float, required=True, help="Temps de début en secondes (ex: 5.0)")
    parser.add_argument("-e", "--end", type=float, required=True, help="Temps de fin en secondes (ex: 10.0)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Nom du fichier ou dossier de sortie (sans extension)")
    parser.add_argument("-n", "--frames", type=int, default=10, help="Nombre d'images à générer (uniquement pour les actions 'images_*'). Par défaut: 10.")

    args = parser.parse_args()

    fichier_tsv = "20250424_0001_00.tsv"
    fichier_csv = "20250424_0001_00.csv"
    video_path = '20250424_0001_00.mp4'

    if args.action == "extrait":
        print(f"Extraction vidéo de {args.start}s à {args.end}s...")
        prendre_extrait(video_path, args.start, args.end, args.output)
        print("Terminé.")
        return

    print(f"Traitement du fichier : {fichier_tsv}")

    df = pd.read_csv(fichier_tsv, sep='\t', comment='#')
    df.to_csv(fichier_csv, index=False)

    gaze2dx = Colonne(fichier_csv, 'Gaze2dX')
    gaze2dy = Colonne(fichier_csv, 'Gaze2dY')
    gaze_class = Colonne(fichier_csv, 'GazeClass')
    MediaTimeStamp = Colonne(fichier_csv, 'MediaTimeStamp')
    
    x_filtered, y_filtered, time_filtered = Nettoyage(gaze2dx, gaze2dy, MediaTimeStamp, gaze_class)

    if args.action == "video_chemin":
        print(f"Génération de la vidéo avec tracé : {args.output}_chemin.mp4")
        video_chemin(video_path, args.start, args.end, args.output, x_filtered, y_filtered, time_filtered)

    elif args.action == "images_parcours":
        print(f"Génération de {args.frames} images (parcours) dans le dossier : {args.output}")
        generer_images_parcours(video_path, args.start, args.end, args.frames, args.output, x_filtered, y_filtered, time_filtered)

    elif args.action == "images_points":
        print(f"Génération de {args.frames} images (points simples) dans le dossier : {args.output}")
        generer_images_points(video_path, args.start, args.end, args.frames, args.output, x_filtered, y_filtered, time_filtered)


if __name__ == "__main__":
    main()
