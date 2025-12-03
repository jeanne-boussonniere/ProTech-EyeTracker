from protech.code import premier_indice, dernier_indice, nettoyage

def test_premier_indice_trouve():
    liste = [0, 1, 2, 3, 4, 5, 6]
    valeur_cherchee = 1.3
    resultat = premier_indice(liste, valeur_cherchee)
    assert resultat == 2

def test_dernier_indice_trouve():
    liste = [0, 1, 2, 3, 4, 5, 6]
    valeur_cherchee = 3.8
    resultat = dernier_indice(liste, valeur_cherchee)
    assert resultat == 3

def test_premier_indice_hors_limites():
    liste = [0, 1, 2, 3, 4, 5, 6]
    valeur_cherchee = 10
    resultat = premier_indice(liste, valeur_cherchee)
    assert resultat is None

def test_nettoyage_donnees_sales():
    gaze_x = [100, 200, 300, 400, 500, 600, 700]
    gaze_y = [10, 20, 30, 40, 50, 60, 70]
    timestamp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    classes = ['F', 'nan', 'None', 'F','S','P','nan']
    res_x, res_y, res_t = nettoyage(gaze_x, gaze_y, timestamp, classes)
    assert res_x == [100, 400, 500, 600]
    assert res_y == [10, 40, 50, 60]
    assert res_t == [1.0, 4.0, 5.0, 6.0]
