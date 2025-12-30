import os
import sys
import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, colorchooser
import pandas as pd
import cv2
import numpy as np
from moviepy import VideoFileClip, CompositeVideoClip, VideoClip

#Chemin de ffmpeg.exe pour pouvoir utiliser pdoc (car moviepy le demande)
_CHEMIN_FFMPEG = fr"C:\Users\jeann\AppData\Local\Programs\Python\Python313\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg.exe"
if os.path.exists(_CHEMIN_FFMPEG):
    os.environ["IMAGEIO_FFMPEG_EXE"] = _CHEMIN_FFMPEG
else:
    print(f"ATTENTION : FFmpeg non trouvé au chemin : {_CHEMIN_FFMPEG}")

#Configuration des loggings
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s : %(message)s',
    stream=sys.stdout
)

CONSOL_WRITE = sys.__stdout__.write
"""
Sauvegarde de la fonction d'écriture standard du système (Terminal).

Cette variable conserve une référence vers la sortie console d'origine.
Elle permet d'afficher des messages de débogage dans la "vraie" console noire,
même après que `sys.stdout` a été redirigé vers l'interface graphique (Tkinter).
"""

# Variables globales pour les couleurs
bg_color_bgr = (0,0,0)
trace_color_bgr = (255, 255, 0)

# ---------------------------------------------------------
# Traitement des données
# ---------------------------------------------------------

def nettoyage (gaze2dx,gaze2dy,media_time_stamp, gaze_class):
    """
    L'objectif est de supprimer le bruit et les données invalides (clignements, perte de tracking).\n
        • Entrées : Listes brutes Gaze2dX, Gaze2dY, MediaTimeStamp, GazeClass.\n
        • Logique : La fonction itère sur les index. Elle conserve uniquement les points où GazeClass \n
                    est valide (différent de nan, None ou vide).\n
        • Sortie : Listes filtrées et synchronisées x_filtered, y_filtered, time_filtered.
    """

    min_len = min(len(gaze2dx), len(gaze2dy), len(media_time_stamp), len(gaze_class))
    fixations_idx = [i for i in range(min_len) if str(gaze_class[i]).strip() not in ['nan', '', 'None']]
    fixation_gaze2dx = [gaze2dx[i] for i in fixations_idx]
    fixation_gaze2dy = [gaze2dy[i] for i in fixations_idx]
    fixation_time_stamp = [media_time_stamp[i] for i in fixations_idx]
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


def premier_indice(lst, value):
    """
    Trouve l'index de la première valeur ≥ value.\n
        • Entrées : Une liste triée (lst), une valeur seuil (value). \n
        • Logique : Parcourt la liste du début à la fin. \n
                    Retourne l'index du premier élément qui est supérieur ou égal à la valeur seuil. \n
        • Sorties : Un entier (index).
    """

    for i, v in enumerate(lst):
        if float(v) >= value:
            return i
    return None


def dernier_indice(lst, value):
    """
    Trouve l'index de la denière valeur ≤ value.\n
        • Entrées : Une liste triée (lst), une valeur seuil (value). \n
        • Logique : Parcourt la liste du début à la fin. \n
                    Retourne l'index du denier élément qui est inférieur ou égal à la valeur seuil. \n
        • Sorties : Un entier (index).
    """

    for i in reversed(range(len(lst))):
        if float(lst[i]) <= value:
            return i
    return None


def extrait_donnees_utiles (t1, t2, x_filtered, y_filtered, time_filtered) :
    """
    Cette fonction gère la synchronisation entre le temps absolu du fichier TSV et l'intervalle demandé par l'utilisateur (t1 à t2).\n
        • Entrées : Temps début (t1), temps fin (t2), listes filtrées (x,y,t).\n
        • Logique : Appelle premier_indice pour t1 et dernier_indice pour t2. \n
                    Découpe les listes Python entre ces deux bornes.\n
        • Sorties : Sous-listes extraites correspondant à l'intervalle temporel.
    """

    if not time_filtered:
        return [], [], []

    t1_precis = premier_indice(time_filtered,t1)
    t2_precis = dernier_indice(time_filtered,t2)

    if t1_precis is None or t2_precis is None or t1_precis > t2_precis:
        return [], [], []

    x_extrait = x_filtered [t1_precis:t2_precis+1]
    y_extrait = y_filtered [t1_precis:t2_precis+1]
    time_extrait = time_filtered [t1_precis:t2_precis+1]

    return x_extrait, y_extrait, time_extrait


def sauvegarder_csv_extrait(df, t1, t2, dossier_sortie):
    """
    Exporte les données filtrées dans un fichier CSV distinct pour faciliter l'analyse statistique externe.\n
        • Entrées : DataFrame global (df), t1, t2, dossier de sortie.\n
        • Logique : Applique un masque booléen sur la colonne 'MediaTimeStamp' pour ne garder \n
                    que les lignes entre t1 et t2.\n
        • Sortie : Fichier _Extrait.csv écrit sur le disque.
    """

    if 'MediaTimeStamp' not in df.columns:
        logging.error("Colonne 'MediaTimeStamp' introuvable.")
        return

    masque_temps = (df['MediaTimeStamp'] >= t1) & (df['MediaTimeStamp'] <= t2)
    df_extrait = df[masque_temps]

    if df_extrait.empty:
        logging.warning("Aucune donnée trouvée dans cet intervalle.")
        return

    nom_fichier_final = os.path.join(dossier_sortie, f"{dossier_sortie}_Extrait.csv")

    df_extrait.to_csv(nom_fichier_final, index=False, sep=',')
    logging.info(f" -> Fichier extrait sauvegardé : {nom_fichier_final}")

# ---------------------------------------------------------
# Traitement vidéos et images
# ---------------------------------------------------------

def prendre_extrait (video,t1,t2,dossier_sortie):
    """
    Génère un fichier vidéo coupé selon les temps donnés, sans modification graphique ni réencodage inutile.\n
        • Entrées : Chemin vidéo source, t1, t2, dossier de sortie.\n
        • Logique : Utilise VideoFileClip.subclipped(t1,t2) pour couper la vidéo.\n
        • Sortie : Fichier .mp4 écrit sur le disque.
    """

    chemin_sortie = os.path.join(dossier_sortie, f"{dossier_sortie}.mp4")
    clip = VideoFileClip(video).subclipped(t1, t2)
    clip.write_videofile(chemin_sortie, codec="libx264", audio_codec="aac", logger=None)
    clip.close()


def video_chemin (video, t1, t2, dossier_sortie, x_filtered, y_filtered, time_filtered, epaisseur, trace_color=(255, 255, 0)) :
    """
    Crée une nouvelle vidéo superposant le tracé évolutif du regard (chemin coloré) sur l'extrait original.\n
        • Entrées : Chemin vidéo, t1, t2, dossier sortie, listes de données nettoyées.\n
        • Logique : Définit une fonction interne make_frame(t). Pour chaque instant t, calcule l'index \n
                    des données correspondant, dessine l'historique des points sur la frame avec un \n
                    dégradé de couleur et renvoie l'image modifiée.\n
        • Sortie : Fichier vidéo _chemin.mp4 écrit sur le disque.
    """

    prendre_extrait (video, t1, t2, dossier_sortie)
    chemin_brut = os.path.join(dossier_sortie, f"{dossier_sortie}.mp4")
    x, y, time_extrait = extrait_donnees_utiles(t1, t2, x_filtered, y_filtered, time_filtered)
    if not x:
        print("ERREUR : Pas de données de regard sur cet intervalle.")
        return

    time_vals = [tt - t1 for tt in time_extrait]
    clip = VideoFileClip(chemin_brut)

    fps = clip.fps
    duration = clip.duration
    total_frames = int(duration * fps)

    donnees_csv = []
    csv_ptr = 0

    for i in range(total_frames):
        t_current = i / fps

        while csv_ptr + 1 < len(time_vals) and time_vals[csv_ptr + 1] <= t_current:
            csv_ptr += 1

        if len(time_vals) > 0 and t_current >= time_vals[0]:
            donnees_csv.append({
                "Frame_Index": i + 1,
                "Video_Time": t_current,
                "Gaze_Time": time_extrait[csv_ptr],
                "Gaze2dX": x[csv_ptr],
                "Gaze2dY": y[csv_ptr]
            })
        else:
            donnees_csv.append({
                "Frame_Index": i + 1,
                "Video_Time": t_current,
                "Gaze_Time": "",
                "Gaze2dX": "",
                "Gaze2dY": ""
            })

    nom_csv_final = os.path.join(dossier_sortie, f"{dossier_sortie}_données_intermédiaires.csv")
    df_out = pd.DataFrame(donnees_csv)
    df_out.to_csv(nom_csv_final, index=False, sep=',')

    current_idx = [0]


    def make_frame(t):
        frame = clip.get_frame(t)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        while current_idx[0] + 1 < len(time_vals) and time_vals[current_idx[0] + 1] <= t:
            current_idx[0] += 1

        if t < time_vals[0]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

        points_to_draw = [(int(x[i]), int(y[i])) for i in range(current_idx[0] + 1)]

        for i in range(1, len(points_to_draw)):
            alpha = i / len(points_to_draw)
            current_color = (
                int(trace_color[0] * alpha),
                int(trace_color[1] * alpha),
                int(trace_color[2] * alpha)
            )
            cv2.line(frame, points_to_draw[i-1], points_to_draw[i], current_color, epaisseur)

        x_t, y_t = int(x[(current_idx[0])]), int(y[current_idx[0]])
        cv2.circle(frame, (x_t, y_t), epaisseur, trace_color, -1)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    chemin_final = os.path.join(dossier_sortie, f"{dossier_sortie}_chemin.mp4")
    trace_clip = VideoClip(make_frame, duration=clip.duration).with_fps(clip.fps)
    final = CompositeVideoClip([clip, trace_clip])
    final.write_videofile(chemin_final, fps=clip.fps, codec="libx264", logger=None)
    clip.close()
    logging.info(f" -> Vidéo finale prête : {chemin_final}")


def generer_images_boucle(video, t1, t2, nb_frames, dossier_sortie, x_filtered, y_filtered, time_filtered, epaisseur, mode="parcours",
                          show_bg=True,  history_frames=0, bg_color=(0, 0, 0),trace_color=(255, 255, 0)):
    """
    Produit une séquence d'images fixes basées sur un échantillonnage temporel régulier.\n
        • Entrées : Vidéo, temps, nombre d'images, dossier de sortie, variables nettoyées,\n
                    options (mode tracé/point, fond noir/image).\n
        • Logique : Calcule N timestamps équidistants. Boucle sur dessus, extrait l'image correspondante\n
                    (ou crée une image noire), dessine les données eye-tracking synchrones et sauvegarde.\n
        • Sortie : Série de fichiers frame_XXX.png écrits sur le disque.
    """

    prendre_extrait(video, t1, t2, dossier_sortie)
    chemin_temp = os.path.join(dossier_sortie, f"{dossier_sortie}.mp4")

    clip = VideoFileClip(chemin_temp)
    x, y, time_extrait = extrait_donnees_utiles(t1, t2, x_filtered, y_filtered, time_filtered)
    time_vals = [tt - t1 for tt in time_extrait] if x else []
    timestamps = np.linspace(0, max(0, clip.duration-0.1), nb_frames)
    w, h = clip.size
    data_indices = []
    ptr = 0
    donnees_csv = []

    for t_frame in timestamps:
        while ptr < len(time_vals) and time_vals[ptr] <= t_frame:
            ptr += 1
        data_indices.append(ptr - 1)

    for i, t_frame in enumerate(timestamps):

        if show_bg:
            try:
                frame = cv2.cvtColor(clip.get_frame(t_frame), cv2.COLOR_RGB2BGR)
            except IOError:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                frame[:] = bg_color
        else:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:] = bg_color

        idx_end_data = data_indices[i]

        if idx_end_data >= 0:
            donnees_csv.append({
                "Num_Image": i + 1,
                "Time": time_extrait[idx_end_data],
                "Gaze2dX": x[idx_end_data],
                "Gaze2dY": y[idx_end_data]
            })
        else:
            donnees_csv.append({
                "Num_Image": i + 1,
                "Time": t1 + t_frame,
                "Gaze2dX": "",
                "Gaze2dY": ""
            })

        if idx_end_data >= 0:
            if mode == "parcours":
                idx_start_data = 0
                if history_frames > 0:
                    prev_img_idx = max(0, i - history_frames)
                    idx_start_data = data_indices[prev_img_idx]

                    idx_start_data = max(0, idx_start_data)

                pts = [(int(x[j]), int(y[j])) for j in range(idx_start_data, idx_end_data + 1)]
                for j in range(1, len(pts)):
                    alpha = j / len(pts)
                    current_color = (
                        int(trace_color[0] * alpha),
                        int(trace_color[1] * alpha),
                        int(trace_color[2] * alpha)
                    )
                    cv2.line(frame, pts[j-1], pts[j], current_color, epaisseur)

                if len(pts) > 0:
                    cv2.circle(frame, pts[-1], epaisseur, trace_color, -1)
            else:
                cv2.circle(frame, (int(x[idx_end_data]), int(y[idx_end_data])), epaisseur, trace_color, -1)

        cv2.imwrite(os.path.join(dossier_sortie, f"frame_{i+1:03d}.png"), frame)
        logging.debug(f"      Image {i+1}/{nb_frames} OK")

    nom_csv_final = os.path.join(dossier_sortie, f"{dossier_sortie}_données_intermédiaires.csv")
    df_out = pd.DataFrame(donnees_csv)
    df_out.to_csv(nom_csv_final, index=False, sep=',')
    logging.info(f" -> CSV des images généré : {nom_csv_final}")

    clip.close()

# ---------------------------------------------------------
# Interface utilisateur
# ---------------------------------------------------------

def choisir_tsv():
    """
    Ouvre l'explorateur de fichiers du système pour permettre la sélection des fichiers de données.\n
        • Logique : Déclenche l'ouverture de la fenêtre de sélection native du système en appliquant \n
                    un filtre sur les extensions (.tsv).\n
        • Sortie : Met à jour la variable Tkinter stockant le chemin du fichier.
    """

    fichier = filedialog.askopenfilename(filetypes=[("TSV Files", "*.tsv"), ("All Files", "*.*")])
    if fichier:
        var_tsv_path.set(fichier)

def choisir_video():
    """
    Ouvre l'explorateur de fichiers du système pour permettre la sélection des fichiers vidéos.\n
        • Logique : Déclenche l'ouverture de la fenêtre de sélection native du système en appliquant \n
                    un filtre sur les extensions (.mp4).\n
        • Sortie : Met à jour la variable Tkinter stockant le chemin du fichier.
    """

    fichier = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")])
    if fichier:
        var_video_path.set(fichier)

def choisir_couleur_trace():
    """
    Ouvre la palette de couleurs du système pour permettre à l'utilisateur de personnaliser l'apparence du rendu (couleur du tracé)\n
        • Logique : Déclenche l'ouverture du sélecteur de couleur natif (colorchooser). \n
                    Si une couleur est validée, elle est convertie du format RGB (interface) \n
                    vers le format BGR (utilisé par le moteur vidéo OpenCV).\n
        • Sortie : Met à jour la variable globale correspondante (bg_color_bgr ou trace_color_bgr)\n
                    et actualise le texte du bouton pour afficher les valeurs R, V, B sélectionnées.
    """
    global trace_color_bgr
    color = colorchooser.askcolor(title="Choisir la couleur du tracé")
    if color[1]:
        r, g, b = int(color[0][0]), int(color[0][1]), int(color[0][2])
        trace_color_bgr = (b, g, r)
        btn_color1.config(text=f"Couleur tracé (R:{r} V:{g} B:{b})")


def choisir_couleur_fond():
    """
    Ouvre la palette de couleurs du système pour permettre à l'utilisateur de personnaliser l'apparence du rendu (couleur du fond)\n
        • Logique : Déclenche l'ouverture du sélecteur de couleur natif (colorchooser). \n
                    Si une couleur est validée, elle est convertie du format RGB (interface) \n
                    vers le format BGR (utilisé par le moteur vidéo OpenCV).\n
        • Sortie : Met à jour la variable globale correspondante (bg_color_bgr ou trace_color_bgr) \n
                    et actualise le texte du bouton pour afficher les valeurs R, V, B sélectionnées.
    """
    global bg_color_bgr
    color = colorchooser.askcolor(title="Choisir la couleur de fond")
    if color[1]:
        r, g, b = int(color[0][0]), int(color[0][1]), int(color[0][2])
        bg_color_bgr = (b, g, r)
        btn_color2.config(text=f"Couleur fond (R:{r} V:{g} B:{b})")

def lancer_traitement():
    """
    Initialise le processus de calcul dans un fil d'exécution séparé pour maintenir la fluidité de l'interface.\n
        • Logique : Démarre un nouveau Thread pointant vers la fonction traitement_background \n
                    pour ne pas figer l'interface.
    """

    threading.Thread(target=traitement_background, daemon=True).start()

def traitement_background():
    """
    Fonction principale du backend qui orchestre la logique métier, la gestion des erreurs et les appels aux modules de traitement.\n
        • Entrées : Variables globales de l'interface (chemins, temps, options).\n
        • Logique : Orchestre tout le processus : vérification des champs, création dossier, \n
                    chargement CSV, appel des fonctions de traitement vidéo/image selon le mode choisi, \n
                    gestion des erreurs.\n
        • Sortie : Popup messagebox (Succès ou Erreur) et fichiers générés.
    """

    tsv = var_tsv_path.get()
    video = var_video_path.get()
    dossier_sortie = entry_name.get().strip()
    avec_fond = var_bg.get()

    if not tsv or not video:
        messagebox.showwarning("Attention", "Il faut choisir les fichiers TSV et Vidéo.")
        return
    if not dossier_sortie:
        messagebox.showwarning("Attention", "Donnez un nom au dossier de sortie.")
        return

    try:
        t1 = float(entry_start.get().replace(',', '.'))
        t2 = float(entry_end.get().replace(',', '.'))
        mode_image = var_image.get()
        val_image_str = entry_frames.get().replace(',', '.')
        thick_str = entry_thickness.get().strip()

        hist_str = entry_hist.get().strip()
        if hist_str == "":
            hist_val = 0
        else:
            hist_val = int(float(hist_str))

        val_thick = int(float(thick_str.replace(',', '.')))
        val_thick = max(1, val_thick)

        if mode_image == "i":
            nb_imgs = int(float(val_image_str))
            if nb_imgs <= 0:
                messagebox.showerror("Erreur", "Le nombre d'images doit être > 0.")
                return
        else:
            freq = float(val_image_str)
            duree = t2 - t1
            nb_imgs = int(round(duree * freq))
            if nb_imgs <= 0:
                messagebox.showerror(
                    "Erreur",
                    "La fréquence est trop faible ou la durée trop courte (Durée < 1s ou Fréquence < 1fps)."
                    )
                return

    except ValueError:
        messagebox.showerror("Erreur", "Les valeurs de temps/fréquence ne sont pas valides.")
        return

    btn_run.config(state="disabled")
    print("\n=== DÉMARRAGE DU TRAITEMENT ===")

    try:
        if not os.path.exists(dossier_sortie):
            os.makedirs(dossier_sortie)
            logging.debug(f"Dossier créé : {dossier_sortie}")
        else:
            logging.debug(f"Dossier existant : {dossier_sortie}")

        choix = var_action.get()
        df_global = None
        x_filt, y_filt, t_filt = [], [], []

        df_global = pd.read_csv(tsv, sep='\t', comment='#')
        nom_full = os.path.join(dossier_sortie, f"{dossier_sortie}_FULL.csv")
        df_global.to_csv(nom_full, index=False, sep=',')
        logging.info(f" -> Fichier csv sauvegardé : {nom_full}")

        if choix != "1":
            gaze2dx = df_global.get('Gaze2dX', pd.Series()).tolist()
            gaze2dy = df_global.get('Gaze2dY', pd.Series()).tolist()
            media_ts = df_global.get('MediaTimeStamp', pd.Series()).tolist()
            gaze_cls = df_global.get('GazeClass', pd.Series()).tolist()

            if not gaze_cls and gaze2dx:
                gaze_cls = ['F'] * len(gaze2dx)

            x_filt, y_filt, t_filt = nettoyage(gaze2dx, gaze2dy, media_ts, gaze_cls)

        if choix == "1":
            prendre_extrait(video, t1, t2, dossier_sortie)
        elif choix == "2":
            video_chemin(video, t1, t2, dossier_sortie,  x_filt, y_filt, t_filt, val_thick, trace_color=trace_color_bgr)
        elif choix == "3":
            generer_images_boucle(video, t1, t2, nb_imgs, dossier_sortie, x_filt, y_filt, t_filt, val_thick, "parcours",
                                  show_bg=avec_fond, history_frames=hist_val, bg_color=bg_color_bgr, trace_color=trace_color_bgr)
        elif choix == "4":
            generer_images_boucle(video, t1, t2, nb_imgs, dossier_sortie, x_filt, y_filt, t_filt, val_thick, "points",
                                  show_bg=avec_fond,  bg_color=bg_color_bgr, trace_color=trace_color_bgr)

        if df_global is not None:
            sauvegarder_csv_extrait(df_global, t1, t2, dossier_sortie)

        print("\n=== TERMINÉ AVEC SUCCÈS ===")
        messagebox.showinfo("Succès", f"Fini ! Vérifiez le dossier : {dossier_sortie}")

    except Exception as e:
        logging.critical(f"Arrêt inattendu du traitement : {e}", exc_info=True)
        messagebox.showerror("Erreur Critique", str(e))

    finally:
        btn_run.config(state="normal")

def rediriger_stdout(text):
    """
    Capture les messages de la console pour les afficher en temps réel dans la zone de journal de l'application.\n
        • Entrées : Texte envoyé vers la console.\n
        • Logique : Insère le texte dans le widget ScrolledText de l'interface graphique et force \n
                    le défilement vers le bas.\n
        • Sortie : Affichage visuel dans la zone "Journal".
    """

    try:
        log_widget.configure(state='normal')
        log_widget.insert(tk.END, text)
        log_widget.see(tk.END)
        log_widget.configure(state='disabled')
        log_widget.update_idletasks()
    except Exception:
        try:
            CONSOL_WRITE(text)
        except Exception:
            pass

def mise_a_jour_interface():
    """
    Adapte les champs visibles du formulaire en fonction de l'action sélectionnée par l'utilisateur.\n
        • Entrées : Changement d'état des boutons radio "Action".\n
        • Logique : Utilise .grid() ou .grid_remove() pour afficher ou cacher les options \n
                    spécifiques aux images (nombre de frames, fond noir) selon l'action choisie.\n
        • Sortie : Mise à jour dynamique de la fenêtre.
    """

    choix = var_action.get()
    coche = var_bg.get()
    mode_image = var_image.get()
    if choix != "1":
        lbl_thick.grid(row=4, column=0, sticky="w",pady=5)
        entry_thickness.grid(row=4, column=1, sticky="w", pady=5)
        btn_color1.grid(row=4, column=2, columnspan=2, sticky="w", pady=5)

    else :
        lbl_thick.grid_remove()
        entry_thickness.grid_remove()
        btn_color1.grid_remove()

    if choix in ["3", "4"]:
        btn1.grid(row=2, column=0, columnspan=2, sticky="w", padx=5)
        btn2.grid(row=3, column=0, columnspan=2, sticky="w", padx=5)
        if mode_image=="i":
            entry_frames.grid(row=2, column=2, sticky="w", padx=5)
        else :
            entry_frames.grid(row=3, column=2, sticky="w", padx=5)
        chk_bg.grid(row=5, column=0, columnspan=4, sticky="w", pady=5)
        if not coche :
            btn_color2.grid(row=6, column=0, columnspan=2, sticky="w", pady=5)
        else :
            btn_color2.grid_remove()

    else:
        btn1.grid_remove()
        btn2.grid_remove()
        entry_frames.grid_remove()
        chk_bg.grid_remove()

    if choix == "3":
        lbl_hist.grid(row=7, column=0, sticky="w", pady=5)
        entry_hist.grid(row=7, column=1, sticky="w", padx=5)
        lbl_hist_info.grid(row=7, column=2, columnspan=2, sticky="w")

    else :
        lbl_hist.grid_remove()
        entry_hist.grid_remove()
        lbl_hist_info.grid_remove()

# ---------------------------------------------------------
# Programme pincipal permettant l'exécution
# ---------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.title("ProTech Eye-Tracking")
    root.geometry("600x750")

    style = ttk.Style()
    style.theme_use('clam')

    var_tsv_path = tk.StringVar()
    var_video_path = tk.StringVar()
    var_action = tk.StringVar(value="1")
    var_bg = tk.BooleanVar(value=True)
    var_image = tk.StringVar(value="i")

    # --- BLOC 1 : Choix des fichiers ---
    frame_files = ttk.LabelFrame(root, text="1. Fichiers", padding=10)
    frame_files.pack(fill="x", padx=10, pady=5)

    ttk.Button(frame_files,
               text="Choisir TSV",
               command=choisir_tsv
               ).grid(row=0, column=0, padx=5, pady=5)
    ttk.Label(frame_files,
              textvariable=var_tsv_path,
              wraplength=400
              ).grid(row=0, column=1, sticky="w")

    ttk.Button(frame_files, text="Choisir MP4", command=choisir_video).grid(row=1, column=0, padx=5, pady=5)
    ttk.Label(frame_files, textvariable=var_video_path, wraplength=400).grid(row=1, column=1, sticky="w")

    # --- BLOC 2 : Choix de l'action ---
    frame_action = ttk.LabelFrame(root, text="2. Action", padding=10)
    frame_action.pack(fill="x", padx=10, pady=5)

    ttk.Radiobutton(frame_action,
                    text="1. Couper vidéo (simple)",
                    variable=var_action,
                    value="1",
                    command=mise_a_jour_interface
                    ).pack(anchor="w")
    ttk.Radiobutton(frame_action,
                    text="2. Vidéo avec tracé",
                    variable=var_action,
                    value="2",
                    command=mise_a_jour_interface
                    ).pack(anchor="w")
    ttk.Radiobutton(frame_action,
                    text="3. Images avec tracé",
                    variable=var_action,
                    value="3",
                    command=mise_a_jour_interface
                    ).pack(anchor="w")
    ttk.Radiobutton(frame_action,
                    text="4. Images avec point",
                    variable=var_action,
                    value="4",
                    command=mise_a_jour_interface
                    ).pack(anchor="w")


    # --- BLOC 3 : Choix des paramètres ---
    frame_params = ttk.LabelFrame(root, text="3. Paramètres", padding=10)
    frame_params.pack(fill="x", padx=10, pady=5)

    ttk.Label(frame_params, text="Début (s) :").grid(row=0, column=0, sticky="w")
    entry_start = ttk.Entry(frame_params, width=10)
    entry_start.grid(row=0, column=1, padx=5, sticky="w")

    ttk.Label(frame_params, text="Fin (s) :").grid(row=0, column=2, sticky="w")
    entry_end = ttk.Entry(frame_params, width=10)
    entry_end.grid(row=0, column=3, padx=5, sticky="w")

    ttk.Label(frame_params, text="Nom Dossier :").grid(row=1, column=0, sticky="w", pady=10)
    entry_name = ttk.Entry(frame_params, width=30)
    entry_name.grid(row=1, column=1, columnspan=3, sticky="w")

    btn1 = ttk.Radiobutton(frame_params,
                           text="Nombre d'images",
                           variable=var_image,
                           value="i",
                           command=mise_a_jour_interface)
    btn2 = ttk.Radiobutton(frame_params,
                           text="Fréquence (images par seconde)",
                           variable=var_image,
                           value="f",
                           command=mise_a_jour_interface)

    entry_frames = ttk.Entry(frame_params, width=10)
    entry_frames.insert(0, "10")

    lbl_thick = ttk.Label(frame_params, text="Epaisseur trait/point :")
    entry_thickness = ttk.Entry(frame_params, width=10)
    entry_thickness.insert(0, "5")

    btn_color1 = ttk.Button(frame_params,
                            text="Choisir couleur traits/points (Bleu par défaut)",
                            command=choisir_couleur_trace)

    chk_bg = ttk.Checkbutton(frame_params,
                             text="Afficher l'image en fond",
                             variable=var_bg,
                             command=mise_a_jour_interface)
    btn_color2 = ttk.Button(frame_params,
                            text="Choisir couleur fond (Noir par défaut)",
                            command=choisir_couleur_fond)

    lbl_hist = ttk.Label(frame_params, text="Historique (nb images) :")
    entry_hist = ttk.Entry(frame_params, width=10)
    lbl_hist_info = ttk.Label(frame_params,
                              text="(0 = tout garder)",
                              font=("Arial", 8, "italic"),
                              foreground="gray")

    # --- BLOC 4 : Exécution et journal ---
    btn_run = ttk.Button(root, text="▶ LANCER", command=lancer_traitement)
    btn_run.pack(fill="x", padx=20, pady=10)

    frame_log = ttk.LabelFrame(root, text="Journal", padding=10)
    frame_log.pack(fill="both", expand=True, padx=10, pady=5)

    log_widget = scrolledtext.ScrolledText(frame_log, height=8, state='disabled')
    log_widget.pack(fill="both", expand=True)

    sys.stdout.write = rediriger_stdout
    sys.stdout.flush = lambda: None

    mise_a_jour_interface()

    sys.stdout.write = sys.__stdout__.write

    root.mainloop()
