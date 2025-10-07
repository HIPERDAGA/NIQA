# -*- coding: utf-8 -*-
"""
Evaluación de NIQE y PIQE integrada con MLflow
Autor: Diego Guevara
"""

# === Dependencias ===
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skvideo.measure as skvm
from pypiqe import piqe
import scipy.misc
import mlflow

# === Parche imresize para SciPy moderno ===
if not hasattr(scipy.misc, "imresize"):
    def imresize(arr, size, interp='bicubic', mode=None):
        if isinstance(size, float):
            new_size = (int(arr.shape[1] * size), int(arr.shape[0] * size))
        elif isinstance(size, (list, tuple)):
            new_size = (size[1], size[0])
        else:
            raise ValueError("El parámetro size debe ser float o (alto, ancho)")
        return cv2.resize(arr, new_size, interpolation=cv2.INTER_CUBIC)
    scipy.misc.imresize = imresize

# === Parche NumPy moderno ===
if not hasattr(np, "int"):
    np.int = int

# === Funciones auxiliares ===
def read_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def evaluar_imagen(path):
    """Calcula NIQE y PIQE de una imagen."""
    img_rgb = read_image_rgb(path)
    img_gray = to_gray(img_rgb)

    niqe_score = skvm.niqe(img_gray)
    piqe_score, *_ = piqe(img_rgb)

    if isinstance(niqe_score, np.ndarray):
        niqe_score = niqe_score.item()
    if isinstance(piqe_score, np.ndarray):
        piqe_score = float(piqe_score)

    return niqe_score, piqe_score

# === MLflow Configuración ===
mlflow.set_tracking_uri("file:./mlruns")  # Local
mlflow.set_experiment("Evaluacion_NIQE_PIQE")

# === Carpeta de imágenes ===
DATA_DIR = "data/imagenes"  # <- ajusta según tu estructura

if not os.path.exists(DATA_DIR):
    raise RuntimeError(f"No se encontró la carpeta {DATA_DIR}")

imagenes = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"📂 {len(imagenes)} imágenes detectadas en {DATA_DIR}")

# === Evaluar y registrar en MLflow ===
for img_path in imagenes:
    niqe_score, piqe_score = evaluar_imagen(img_path)

    with mlflow.start_run(run_name=os.path.basename(img_path)):
        mlflow.log_param("imagen", os.path.basename(img_path))
        mlflow.log_metric("NIQE", niqe_score)
        mlflow.log_metric("PIQE", piqe_score)
        mlflow.log_artifact(img_path)

    print(f"✅ {os.path.basename(img_path)} -> NIQE={niqe_score:.2f}, PIQE={piqe_score:.2f}")
