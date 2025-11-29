import logging
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pandas as pd
import cv2
import numpy as np
from moviepy import VideoFileClip, CompositeVideoClip, VideoClip

logging.basicConfig(
    level=logging.DEBUG,  # On capture tout (du plus bas au plus haut)
    format='%(levelname)s : %(message)s', # Format : Date - Niveau - Message
    stream=sys.stdout # IMPORTANT : On envoie les logs vers stdout pour que ton interface les affiche
)

CONSOL_WRITE = sys.__stdout__.write

# Garde uniquement les données utiles (les points de fixation)
def nettoyage (gaze2dx,gaze2dy,media_time_stamp, gaze_class):
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


# Trouve l'indice de la première valeur supérieure à la valeur donnée
def premier_indice(lst, value):
    for i, v in enumerate(lst):
        if float(v) >= value:
            return i
    return None


# Trouve l'indice de la dernière valeur inférieure à la valeur donnée
def dernier_indice(lst, value):
    for i in reversed(range(len(lst))):
        if float(lst[i]) <= value:
            return i
    return None


# Prend un extrait d'une vidéo selon les timecodes donnés
def prendre_extrait (video,t1,t2,dossier_sortie):
    chemin_sortie = os.path.join(dossier_sortie, f"{dossier_sortie}.mp4")
    clip = VideoFileClip(video).subclipped(t1, t2)
    clip.write_videofile(chemin_sortie, codec="libx264", audio_codec="aac", logger=None)
    clip.close()


# Extrait les données correspondant à un extrait de vidéo entre t1 et t2
def extrait_donnees_utiles (t1, t2, x_filtered, y_filtered, time_filtered) :
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

# Sauvegarde au format CSV les données entre t1 et t2.
def sauvegarder_csv_extrait(df, t1, t2, dossier_sortie):
    """Sauvegarde le CSV filtré dans le dossier de sortie."""

    if 'MediaTimeStamp' not in df.columns:
        logging.error("Colonne 'MediaTimeStamp' introuvable.")
        return

    masque_temps = (df['MediaTimeStamp'] >= t1) & (df['MediaTimeStamp'] <= t2)
    df_extrait = df[masque_temps]

    if df_extrait.empty:
        logging.warning("Aucune donnée trouvée dans cet intervalle.")
        return

    nom_fichier_final = os.path.join(dossier_sortie, f"{dossier_sortie}_Extrait.csv")

    df_extrait.to_csv(nom_fichier_final, index=False, sep=';')
    logging.info(f" -> Fichier extrait sauvegardé : {nom_fichier_final}")


# Trace le chemin du regard pour un extrait de vidéo entre t1 et t2
def video_chemin (video, t1, t2, dossier_sortie, x_filtered, y_filtered, time_filtered) :
    prendre_extrait (video, t1, t2, dossier_sortie)
    chemin_brut = os.path.join(dossier_sortie, f"{dossier_sortie}.mp4")
    x, y, time_extrait = extrait_donnees_utiles(t1, t2, x_filtered, y_filtered, time_filtered)
    time = [tt - t1 for tt in time_extrait]

    clip = VideoFileClip(chemin_brut)

    current_idx = [0]

    # Dessine, pour un instant t donné, l'image vidéo correspondante avec au choix :
    # le tracé précédent complet ou le point de fixation actuel.
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

    chemin_final = os.path.join(dossier_sortie, f"{dossier_sortie}_chemin.mp4")
    trace_clip = VideoClip(make_frame, duration=clip.duration).with_fps(clip.fps)
    final = CompositeVideoClip([clip, trace_clip])
    final.write_videofile(chemin_final, fps=clip.fps, codec="libx264", logger=None)
    clip.close()
    logging.info(f" -> Vidéo finale prête : {chemin_final}")


# Génère le nombre d'images demandé entre t1 et t2 avec le tracé du regard
# jusqu'à l'instant t de l'image ou la position du regard à l'instant t
def generer_images_boucle(video_source, t1, t2, nb_frames, dossier_sortie, x_filtered, y_filtered, time_filtered, mode="parcours", show_bg=True):
    prendre_extrait(video_source, t1, t2, dossier_sortie)
    chemin_temp = os.path.join(dossier_sortie, f"{dossier_sortie}.mp4")

    clip = VideoFileClip(chemin_temp)
    x, y, time_extrait = extrait_donnees_utiles(t1, t2, x_filtered, y_filtered, time_filtered)
    time_vals = [tt - t1 for tt in time_extrait] if x else []
    timestamps = np.linspace(0, max(0, clip.duration-0.1), nb_frames)
    ptr = 0
    idx_dessin = -1
    w, h = clip.size

    for i, t_frame in enumerate(timestamps):

        if show_bg:
            frame = cv2.cvtColor(clip.get_frame(t_frame), cv2.COLOR_RGB2BGR)
        else:
            frame = np.zeros((h, w, 3), dtype=np.uint8) # Fond noir

        if time_vals:
            while ptr < len(time_vals) and time_vals[ptr] <= t_frame:
                ptr += 1
            current_valid_idx = ptr - 1
            if current_valid_idx >= 0 :
                idx_dessin = current_valid_idx
            if idx_dessin >= 0:
                if mode == "parcours":
                    pts = [(int(x[j]), int(y[j])) for j in range(idx_dessin + 1)]
                    for j in range(1, len(pts)):
                        cv2.line(frame, pts[j-1], pts[j], (255, int(255*(j/len(pts))), 0), 4)
                    cv2.circle(frame, pts[-1], 8, (255, 255, 0), -1)
                else:
                    cv2.circle(frame, (int(x[idx_dessin]), int(y[idx_dessin])), 8, (255, 255, 0), -1)

        cv2.imwrite(os.path.join(dossier_sortie, f"frame_{i+1:03d}.png"), frame)
        logging.debug(f"      Image {i+1}/{nb_frames} OK")
    clip.close()
    try:
        os.remove(chemin_temp)
    except OSError:
        pass

# ---------------------------------------------------------
# Interface utilisateur
# ---------------------------------------------------------

# Parcours le dossier contenant le code pour trouver tous les fichiers tsv.
def choisir_tsv():
    fichier = filedialog.askopenfilename(filetypes=[("TSV Files", "*.tsv"), ("All Files", "*.*")])
    if fichier:
        var_tsv_path.set(fichier)

# Parcours le dossier contenant le code pour trouver tous les fichiers mp4.
def choisir_video():
    fichier = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")])
    if fichier:
        var_video_path.set(fichier)

# Lancement du traitement
def lancer_traitement():
    threading.Thread(target=traitement_background, daemon=True).start()

# Interface qui intéragit avec l'utilisateur.
def traitement_background():
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

        if choix != "1":
            df_global = pd.read_csv(tsv, sep='\t', comment='#')

            nom_full = os.path.join(dossier_sortie, f"{dossier_sortie}_FULL.csv")
            df_global.to_csv(nom_full, index=False, sep=';')
            logging.info(f" -> Fichier extrait sauvegardé : {nom_full}")

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
            video_chemin(video, t1, t2, dossier_sortie,  x_filt, y_filt, t_filt)
        elif choix == "3":
            generer_images_boucle(video, t1, t2, nb_imgs, dossier_sortie, x_filt, y_filt, t_filt, "parcours", show_bg=avec_fond)
        elif choix == "4":
            generer_images_boucle(video, t1, t2, nb_imgs, dossier_sortie, x_filt, y_filt, t_filt, "points", show_bg=avec_fond)

        if df_global is not None:
            sauvegarder_csv_extrait(df_global, t1, t2, dossier_sortie)

        print("\n=== TERMINÉ AVEC SUCCÈS ===")
        messagebox.showinfo("Succès", f"Fini ! Vérifiez le dossier : {dossier_sortie}")

    except Exception as e:
        logging.critical(f"Arrêt inattendu du traitement : {e}", exc_info=True)
        messagebox.showerror("Erreur Critique", str(e))

    finally:
        btn_run.config(state="normal")

# Redirige les affichages dans le journal de l'interface.
def rediriger_stdout(text):
    try:
        log_widget.configure(state='normal')
        log_widget.insert(tk.END, text)
        log_widget.see(tk.END)
        log_widget.configure(state='disabled')
    except Exception:
        try:
            CONSOL_WRITE(text)
        except Exception:
            pass

# Affiche ou non les paramètres spécifique aux images.
def mise_a_jour_interface():
    choix = var_action.get()
    if choix in ["3", "4"]:
        btn1.grid(row=2, column=0, sticky="w", padx=5)
        btn2.grid(row=2, column=1, sticky="w", padx=5)
        entry_frames.grid(row=3, column=0, sticky="w", padx=5)
        chk_bg.grid(row=4, column=0, columnspan=4, sticky="w", pady=5)
    else:
        btn1.grid_remove()
        btn2.grid_remove()
        entry_frames.grid_remove()
        chk_bg.grid_remove()

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

    chk_bg = ttk.Checkbutton(frame_params, text="Afficher l'image en fond", variable=var_bg)

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
