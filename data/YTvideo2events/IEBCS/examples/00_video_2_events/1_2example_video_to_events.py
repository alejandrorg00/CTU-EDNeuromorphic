# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:47:54 2025

@author: c3055922
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 22:02:19 2025
@author: c3055922
Adapted from IEBCS example, Joubert Damien, 03-02-2020 - updated by AvS 22-02-2024

Script converting a video into events. 
Please run get_video_youtube.py before executing this script.
"""

import cv2, sys, os
from tqdm import tqdm
sys.path.append("../../src")
from event_buffer   import EventBuffer
from dvs_sensor     import DvsSensor
from event_display  import EventDisplay
from arbiter        import SynchronousArbiter

# === Parámetros DVS ===
th_pos   = 0.4        # ON threshold = 50% (ln(1.5) = 0.4)
th_neg   = 0.4        # OFF threshold = 50%
th_noise = 0.01       # std dev of threshold noise
lat      = 100        # latency in µs
tau      = 40         # time constant at 1 klux in µs
jit      = 10         # temporal jitter in µs
bgnp     = 0.1        # ON event noise rate (events/pixel/s)
bgnn     = 0.01       # OFF event noise rate (events/pixel/s)
ref      = 100        # refractory period in µs
dt       = 1000       # time between frames in µs

# === Vídeo de entrada y carpeta de salida ===
filename = "../../data/mug.mp4"
cap = cv2.VideoCapture(filename)
os.makedirs("./outputs", exist_ok=True)

# === 1) Inicializar el DVS con la resolución de los .npy de ruido: 720×480 ===
dvs = DvsSensor("MySensor")
dvs.initCamera(
    720, 480,                         # ancho, alto fijos
    lat=lat, jit=jit, ref=ref, tau=tau,
    th_pos=th_pos, th_neg=th_neg, th_noise=th_noise,
    bgnp=bgnp, bgnn=bgnn
)
# Si quieres usar los histogramas de ruido medidos, déjalo; si no, coméntalo:
dvs.init_bgn_hist(
    "../../data/noise_pos_161lux.npy",
    "../../data/noise_neg_161lux.npy"
)

# === Recuperar la resolución interna que espera el sensor ===
sensor_w, sensor_h = dvs.shape  # (width, height)

# === 2) Saltar primeras 50 frames para “warm-up” ===
for _ in range(50):
    cap.read()

# === 3) Leer y REDIMENSIONAR el primer frame ANTES de init_image ===
ret, im = cap.read()
if not ret:
    raise RuntimeError("No se pudo leer el primer frame")
im = cv2.resize(im, (sensor_w, sensor_h))
im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0] / 255.0 * 1e4
dvs.init_image(im)

# === 4) Crear buffer, arbiter y display con las mismas dimensiones ===
ev_full = EventBuffer(1)
ea      = SynchronousArbiter(0.1, 0, sensor_h)
ed      = EventDisplay("Events", sensor_w, sensor_h, dt, 1)

# === 5) Procesar el bucle, redimensionando cada frame ===
num_frames = 250
for _ in tqdm(range(num_frames), desc="Converting video to events"):
    ret, im = cap.read()
    if not ret:
        break
    # REDIMENSIONAR
    im = cv2.resize(im, (sensor_w, sensor_h))
    # Convertir a LUV luminance (10 klux)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0] / 255.0 * 1e4

    ev = dvs.update(im, dt)
    if ev is None:
        raise RuntimeError("Sensor devolvió None; shape no coincide")
    ed.update(ev, dt)
    ev_full.increase_ev(ev)

cap.release()

# === 6) Guardar eventos en archivo .dat ===
outname = f'outputs/ev_{lat}_{jit}_{ref}_{tau}_{th_pos}_{th_noise}.dat'
ev_full.write(outname)
print(f"Eventos guardados en {outname}")
