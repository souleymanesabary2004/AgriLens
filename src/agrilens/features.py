"""
features.py  (V10 : GLCM sur 32 niveaux)

Extraction de caractéristiques manuelles pour AgriLens.

Objectif
--------
Transformer une image en un vecteur numérique de caractéristiques
(features) utilisable par des modèles de Machine Learning classiques
(Random Forest, Régression Logistique, etc.).

Ces caractéristiques servent de baseline avant l'utilisation
de modèles de Deep Learning comme MobileNetV2.
"""

import cv2
import numpy as np
from skimage.feature import (
    canny,
    graycomatrix,
    graycoprops,
    local_binary_pattern,
)


def load_image(filepath):
    """
    Charge une image depuis le disque.

    Paramètres
    ----------
    filepath : str
        Chemin de l'image à charger.

    Retour
    ------
    numpy.ndarray
        Image RGB.
    """

    img = cv2.imread(filepath)

    if img is None:
        raise FileNotFoundError(
            f"Impossible de charger l'image : {filepath}"
        )

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    return img_rgb


def resize_image(img_rgb, size=(224, 224)):
    """
    Redimensionne une image.

    Paramètres
    ----------
    img_rgb : numpy.ndarray
        Image RGB.

    size : tuple
        Taille cible (largeur, hauteur).

    Retour
    ------
    numpy.ndarray
        Image redimensionnée.
    """

    img_dimensionnee = cv2.resize(
        img_rgb,
        size,
        interpolation=cv2.INTER_AREA
    )

    return img_dimensionnee


def extract_rgb_stats(img_dimensionnee):
    """
    Calcule les statistiques RGB.
    """

    r = img_dimensionnee[:, :, 0]
    g = img_dimensionnee[:, :, 1]
    b = img_dimensionnee[:, :, 2]

    return {
        "rgb_mean_r": np.mean(r),
        "rgb_std_r": np.std(r),

        "rgb_mean_g": np.mean(g),
        "rgb_std_g": np.std(g),

        "rgb_mean_b": np.mean(b),
        "rgb_std_b": np.std(b),
    }


def extract_hsv_stats(img_dimensionnee):
    """
    Calcule les statistiques HSV.
    """

    img_hsv = cv2.cvtColor(
        img_dimensionnee,
        cv2.COLOR_RGB2HSV
    )

    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    return {
        "hsv_mean_h": np.mean(h),
        "hsv_std_h": np.std(h),

        "hsv_mean_s": np.mean(s),
        "hsv_std_s": np.std(s),

        "hsv_mean_v": np.mean(v),
        "hsv_std_v": np.std(v),
    }


def extract_hsv_histogram(img_dimensionnee, bins=16):
    """
    Calcule les histogrammes HSV normalisés.
    """

    img_hsv = cv2.cvtColor(
        img_dimensionnee,
        cv2.COLOR_RGB2HSV
    )

    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    h_hist, _ = np.histogram(
        h,
        bins=bins,
        range=(0, 180)
    )

    s_hist, _ = np.histogram(
        s,
        bins=bins,
        range=(0, 256)
    )

    v_hist, _ = np.histogram(
        v,
        bins=bins,
        range=(0, 256)
    )

    h_hist = h_hist / h_hist.sum()
    s_hist = s_hist / s_hist.sum()
    v_hist = v_hist / v_hist.sum()

    hsv_hist = {}

    for i in range(bins):
        hsv_hist[f"h_hist_{i}"] = h_hist[i]
        hsv_hist[f"s_hist_{i}"] = s_hist[i]
        hsv_hist[f"v_hist_{i}"] = v_hist[i]

    return hsv_hist


def extract_lbp(img_dimensionnee):
    """
    Calcule les features LBP.
    """

    img_gray = cv2.cvtColor(
        img_dimensionnee,
        cv2.COLOR_RGB2GRAY
    )

    lbp = local_binary_pattern(
        img_gray,
        P=8,
        R=1,
        method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp,
        bins=10,
        range=(0, 10)
    )

    lbp_hist = lbp_hist / lbp_hist.sum()

    lbp_features = {}

    for i in range(10):
        lbp_features[f"lbp_{i}"] = lbp_hist[i]

    return lbp_features


def extract_glcm(img_dimensionnee):
    """
    Calcule les statistiques GLCM.
    """

    img_gray = cv2.cvtColor(
        img_dimensionnee,
        cv2.COLOR_RGB2GRAY
    )

    # V10-2 : quantification 256 -> 32 niveaux (matrice dense, plus stable et plus rapide)
    img_gray = (img_gray // 8).astype(np.uint8)

    glcm = graycomatrix(
        img_gray,
        distances=[1],
        angles=[0],
        levels=32,
        symmetric=True,
        normed=True
    )

    return {
        "glcm_contrast":
            graycoprops(glcm, "contrast")[0, 0],

        "glcm_homogeneity":
            graycoprops(glcm, "homogeneity")[0, 0],

        "glcm_energy":
            graycoprops(glcm, "energy")[0, 0],

        "glcm_correlation":
            graycoprops(glcm, "correlation")[0, 0],
    }


def extract_edge_density(img_dimensionnee):
    """
    Calcule la densité de contours.
    """

    img_gray = cv2.cvtColor(
        img_dimensionnee,
        cv2.COLOR_RGB2GRAY
    )

    edges = canny(img_gray)

    edge_density = np.sum(edges) / edges.size

    return {
        "edge_density": edge_density
    }


def extract_features(filepath):
    """
    Extrait toutes les features d'une image.
    """

    img_rgb = load_image(filepath)

    img_dimensionnee = resize_image(img_rgb)

    features = {}

    features.update(
        extract_rgb_stats(img_dimensionnee)
    )

    features.update(
        extract_hsv_stats(img_dimensionnee)
    )

    features.update(
        extract_hsv_histogram(img_dimensionnee)
    )

    features.update(
        extract_lbp(img_dimensionnee)
    )

    features.update(
        extract_glcm(img_dimensionnee)
    )

    features.update(
        extract_edge_density(img_dimensionnee)
    )

    return features