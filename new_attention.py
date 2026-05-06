# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 23:04:43 2025

@author: c3055922
"""

import numpy as np
from helpers.helpers import initialise_attention, run_attention
import torch
import cv2
import matplotlib.pyplot as plt
import math
from matplotlib import gridspec

# Configuration class to store attention parameters
class Config:
    ATTENTION_PARAMS = {
        'size_krn': 16,             # Size of the kernel used in the attention mechanism
        'r0': 14,                   # Radius shift from the center for the attention arc
        'rho': 0.05,                # Scale coefficient to control the arc length
        'theta': np.pi * 3 / 2,     # Base angle for the attention arc
        'thetas': np.arange(0, 2 * np.pi, np.pi / 4),  # Multi-directional angles
        'thick': 3,                 # Arc thickness in the attention map
        'fltr_resize_perc': [2, 2], # Resize percentage for filters
        'offsetpxs': 0,             # Pixel offset (half the kernel size)
        'offset': (0, 0),           # Offset for attention positioning
        'num_pyr': 6,               # Number of pyramid levels in the attention network
        'tau_mem': 0.3,             # Memory time constant for the attention mechanism
        'stride': 1,                # Stride for attention processing
        'out_ch': 1                 # Number of output channels
    }

# Choose device for PyTorch (MPS on macOS if available, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize configuration
config = Config()

# Load event data from a .npy file
data = np.load('data/twoobjects/cup_300fps.npy')
x = data[:, 0].astype(int)
y = data[:, 1].astype(int)
p = data[:, 2]
t = data[:, 3] * 1e3  # convert to milliseconds

# Determine the spatial resolution
width  = x.max() + 1
height = y.max() + 1
resolution = (height, width)

# Initialize saliency map and container for peak coordinates
saliency_map = np.zeros(resolution, dtype=np.float32)
salmax_coords = np.zeros((2,), dtype=np.int32)

# Initialize the attention network
net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

# Time-window parameters
window_period = 1  # ms per window
current_time  = window_period
window_tensor = torch.zeros((1, height, width), dtype=torch.float32)

# History containers
attention_xy   = []
pantilt_deltas = []
prev_x, prev_y = 0, 0

# Prepare cropping parameters: 1% of height & width
crop_h = max(1, int(height * 0.1))
crop_w = max(1, int(width  * 0.1))
half_h = crop_h // 2
half_w = crop_w // 2

# Store raw colorized crops (no attention mark)
crops = []

# Main event loop
for xi, yi, pi, ti in zip(x, y, p, t):
    if ti <= current_time:
        window_tensor[0, yi, xi] = 255
    else:
        # Compute saliency and peak
        saliency_map[:], salmax_coords[:] = run_attention(
            window_tensor, net_attention, device, resolution,
            config.ATTENTION_PARAMS['num_pyr']
        )
        peak_x = int(salmax_coords[1])
        peak_y = int(salmax_coords[0])
        attention_xy.append((peak_x, peak_y))

        # Pan/tilt delta
        dx = peak_x - prev_x
        dy = peak_y - prev_y
        pantilt_deltas.append((dx, dy))
        prev_x, prev_y = peak_x, peak_y

        # Colorize for display and for raw crop
        frame_uint8 = window_tensor.detach().cpu().numpy().squeeze().astype(np.uint8)
        colored = cv2.applyColorMap(frame_uint8, cv2.COLORMAP_JET)
        raw_frame = colored.copy()  # keep unannotated version for crop

        # Annotate display frame
        cv2.putText(colored, 'Event Window', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(colored, (peak_x, peak_y), 6, (255, 255, 255), 4)
        cv2.putText(colored, 'Attention Peak', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show in real time
        cv2.imshow('Event Window and Saliency', colored)
        cv2.waitKey(1)

        # Crop 1%×1% region centered at peak, from raw_frame
        x1 = max(peak_x - half_w, 0)
        x2 = min(peak_x + half_w, width)
        y1 = max(peak_y - half_h, 0)
        y2 = min(peak_y + half_h, height)
        crops.append(raw_frame[y1:y2, x1:x2])

        # Advance time window
        current_time += window_period
        window_tensor = torch.zeros((1, height, width), dtype=torch.float32)

# Cleanup
cv2.destroyAllWindows()

# Convert history to numpy array
attention_xy = np.array(attention_xy)

# ========== Plot 1: saccadic path ==========
fig1, ax1 = plt.subplots(figsize=(6, 6))
H, W = height, width

# line + scatter
ax1.plot(attention_xy[:, 0], attention_xy[:, 1],
         '-', alpha=0.5, linewidth=2, color='royalblue')
ax1.scatter(attention_xy[:, 0], attention_xy[:, 1],
            s=80, edgecolors='white', linewidth=1, color='royalblue')

ax1.set_aspect('equal', 'box')
ax1.set_xlim(0, W-1)
ax1.set_ylim(H-1, 0)    # invert y-axis
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
ax1.tick_params(direction='in', length=6, width=2,
                labelbottom=False, labelleft=False)
for spine in ax1.spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
fig1.savefig('figures/ev_cup_att.svg', dpi=600)
fig1.savefig('figures/ev_cup_att.pdf', dpi=600)
plt.show()

# ========== Plot 2: pan-tilt deltas over first frame ==========
first_frame = window_uint8 = window_uint8 = window_tensor.detach().cpu().numpy().squeeze().astype(np.uint8)  # placeholder
# In practice, save the very first 'frame_uint8' when loop starts for a true first_frame

# Normalize time for color mapping
N = len(attention_xy)
t_vals = np.linspace(0, 1, N)
xs = attention_xy[:, 0]
ys = attention_xy[:, 1]
dxs = xs[1:] - xs[:-1]
dys = ys[1:] - ys[:-1]

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.imshow(first_frame, cmap='gray', origin='upper')
sc = ax2.scatter(xs, ys, c=t_vals, cmap='winter', s=35, edgecolor='none', zorder=2)
ax2.quiver(xs[:-1], ys[:-1], dxs, dys, t_vals[:-1],
           cmap='winter', scale_units='xy', angles='xy',
           scale=1, width=0.004, headwidth=4, headlength=7, zorder=3)
ax2.axis('off')
cax = fig2.add_axes([1.0, 0.5, 0.02, 0.36])
cbar = fig2.colorbar(sc, cax=cax, ticks=[0,1])
cbar.set_label("Time (normalized)")
plt.tight_layout()
plt.show()

# ========== Plot 3: grid of crops ==========
num_crops = len(crops)
cols = 8
rows = math.ceil(num_crops / cols)
fig3, axes3 = plt.subplots(rows, cols,
                           figsize=(cols*2, rows*2),
                           constrained_layout=True)
axes3 = axes3.flatten()
for idx, ax in enumerate(axes3):
    ax.axis('off')
    if idx < num_crops:
        rgb = cv2.cvtColor(crops[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
plt.show()


# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# === 0) tensor → uint8 window ===
window_uint8 = window_tensor.detach().cpu().numpy().squeeze().astype(np.uint8)

# === 1) first video frame ===
video_path = r"H:\Desktop\Github\CTU-EDNeuromorphic\data\YTvideo2events\IEBCS\data\cofeecup_5s_1000fps_720x480.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame_bgr = cap.read()
cap.release()
if not ret:
    raise IOError(f"Could not read first frame from {video_path}")
first_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
H, W = first_frame_rgb.shape[:2]

# === 2) data prep ===
attention_xy = np.array(attention_xy)
first_three = crops[:3]
last_one    = crops[-1]

# === 3) 2×2 layout w/ tighter vertical spacing ===
fig = plt.figure(figsize=(12, 12))
outer = gridspec.GridSpec(
    2, 2, 
    figure=fig, 
    wspace=0.1, 
    hspace=0.025   # much less vertical gap
)

# -- (0,0): original RGB frame --
ax1 = fig.add_subplot(outer[0, 0])
ax1.imshow(first_frame_rgb)
ax1.axis('off')
ax1.set_title("Setup (RGB frame)", fontweight='bold', fontsize=18, pad=20)

# -- (0,1): window frame --
ax2 = fig.add_subplot(outer[0, 1])
ax2.imshow(window_uint8, cmap='gray', vmin=0, vmax=255)
ax2.axis('off')
ax2.set_title("DVS events", fontweight='bold', fontsize=18, pad=20)

# -- (1,0): full saccadic path title --
ax3 = fig.add_subplot(outer[1, 0])
ax3.plot(attention_xy[:,0], attention_xy[:,1],
         '-', linewidth=2, alpha=0.5, color='royalblue')
ax3.scatter(attention_xy[:,0], attention_xy[:,1],
            s=80, edgecolors='white', linewidth=1, color='royalblue')
ax3.set_aspect('equal', 'box')
ax3.set_xlim(0, W-1)
ax3.set_ylim(H-1, 0)
ax3.tick_params(
    axis='both', which='both',
    direction='in', length=6, width=2,
    labelbottom=False, labelleft=False
)
for spine in ax3.spines.values():
    spine.set_linewidth(2)
ax3.set_xlabel('x (480 px)', fontsize=16)
ax3.set_ylabel('y (720 px)', fontsize=16)
ax3.set_title(
    r"$\mathbf{Saccades\,(pan\ and\ tilt\ movements)}$" "\n"
    r"$\{\ (\Delta x_1,\Delta y_1),\dots,(\Delta x_n,\Delta y_n)\ \}$",
    fontweight='bold',
    color='royalblue',
    pad=35,         # your requested pad
    fontsize=18
)

# -- (1,1): ROIs with title closer to images --
ax4 = fig.add_subplot(outer[1, 1])
ax4.axis('off')
ax4.set_title("Regions of Interest", fontweight='bold',
              color='crimson', fontsize=18, pad=5)  # tightened pad

inner = gridspec.GridSpecFromSubplotSpec(
    2, 2,
    subplot_spec=outer[1, 1],
    wspace=0.05,
    hspace=-0.4
)

for idx, (ax, img, label) in enumerate([
    (fig.add_subplot(inner[0, 0]), first_three[0], r'$\mathrm{ROI}_1$'),
    (fig.add_subplot(inner[0, 1]), first_three[1], r'$\mathrm{ROI}_2$'),
    (fig.add_subplot(inner[1, 0]), first_three[2], r'$\mathrm{ROI}_3$'),
    (fig.add_subplot(inner[1, 1]), last_one,      r'$\mathrm{ROI}_n$')
]):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label, fontsize=18, pad=-10, color='crimson')

# === 4) Save & show ===
#fig.savefig('figures/abstract_fig_2x2.svg', format='svg', dpi=600, bbox_inches='tight')
#fig.savefig('figures/abstract_fig_2x2.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()


# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# === 0) tensor → uint8 window ===
window_uint8 = window_tensor.detach().cpu().numpy().squeeze().astype(np.uint8)

# === 1) first video frame ===
video_path = r"H:\Desktop\Github\CTU-EDNeuromorphic\data\YTvideo2events\IEBCS\data\cofeecup_5s_1000fps_720x480.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame_bgr = cap.read()
cap.release()
if not ret:
    raise IOError(f"Could not read first frame from {video_path}")
first_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
H, W = first_frame_rgb.shape[:2]

# === 2) data prep ===
attention_xy = np.array(attention_xy)
first_three = crops[:3]
last_one    = crops[-1]

# === 3) 2×2 layout w/ tighter vertical spacing ===
fig = plt.figure(figsize=(12, 12))
outer = gridspec.GridSpec(
    2, 2, 
    figure=fig, 
    wspace=0.1,
    hspace=0.025   # much less vertical gap
)

# -- (0,0): original RGB frame --
ax1 = fig.add_subplot(outer[0, 0])
ax1.imshow(first_frame_rgb)
ax1.axis('off')
ax1.set_title("Setup (RGB frame)", fontweight='bold', fontsize=18, pad=20)

# -- (0,1): window frame --
ax2 = fig.add_subplot(outer[0, 1])
ax2.imshow(window_uint8, cmap='gray', vmin=0, vmax=255)
ax2.axis('off')
ax2.set_title("DVS events", fontweight='bold', fontsize=18, pad=20)

# === ROI select ===
roi_indices = [0, 2, 38, 67]
roi_labels = [rf'$\mathrm{{ROI}}_{{{i + 1}}}$' for i in roi_indices]
selected_coords = attention_xy[roi_indices]
selected_crops  = [crops[i] for i in roi_indices]


# === Subplot 3: DVS image + saccadic path + ROI boxes ===
ax3 = fig.add_subplot(outer[1, 0])

# Mostrar imagen DVS (en gris invertido)
ax3.imshow(window_uint8, cmap='gray_r', vmin=0, vmax=255)

# Añadir trayectoria sacádica
ax3.plot(attention_xy[:, 0], attention_xy[:, 1],
         '-', linewidth=2, alpha=0.5, color='royalblue')
ax3.scatter(attention_xy[:, 0], attention_xy[:, 1],
            s=10, edgecolors='royalblue', linewidth=1, color='royalblue')

# Dibujar los rectángulos de ROI
for i, (x, y) in enumerate(selected_coords):
    rect = plt.Rectangle(
        (x - half_w, y - half_h),
        crop_w, crop_h,
        edgecolor='crimson', facecolor='none',
        linewidth=2
    )
    ax3.add_patch(rect)
    ax3.text(x, y + 60, roi_labels[i], color='crimson',
             fontsize=14, ha='center', fontweight='bold')

# Estética del subplot
ax3.set_aspect('equal', 'box')
ax3.set_xlim(0, W - 1)
ax3.set_ylim(H - 1, 0)
ax3.tick_params(axis='both', which='both', direction='in', length=6, width=2,
                labelbottom=False, labelleft=False)
for spine in ax3.spines.values():
    spine.set_linewidth(2)
ax3.set_xlabel('x (720 px)', fontsize=16)
ax3.set_ylabel('y (480 px)', fontsize=16)
ax3.set_title(
    r"$\mathbf{Saccades\,(pan\ and\ tilt\ movements)}$" "\n"
    r"$\{\ (\Delta x_1,\Delta y_1),\dots,(\Delta x_n,\Delta y_n)\ \}$",
    fontweight='bold',
    color='royalblue',
    pad=60,
    fontsize=18
)

# === Subplot 4: ROIs en grid 2×2 sin líneas, sin huecos ===
ax4 = fig.add_subplot(outer[1, 1])
ax4.axis('off')
ax4.set_title(
    r"$\mathbf{Regions\ of\ Interest\ (ROIs)}$" "\n"
    r"$\{\ ROI_1,\dots,\ ROI_{n}\ \}$",
    fontweight='bold',
    color='crimson',
    pad=0,
    fontsize=18
)

# Grid 2×2 para los 4 ROIs con mínimo espacio
inner = gridspec.GridSpecFromSubplotSpec(
    2, 2,
    subplot_spec=outer[1, 1],
    wspace=0.02,
    hspace=-0.4
)

# Mostrar cada ROI con padding mínimo
for idx, (img, label) in enumerate(zip(selected_crops, roi_labels)):
    i, j = divmod(idx, 2)
    ax = fig.add_subplot(inner[i, j])
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label, fontsize=13, pad=1, color='crimson')


# === 4) Save & show ===
fig.savefig('figures/abstract_fig2x2.svg', format='svg', dpi=600, bbox_inches='tight')
fig.savefig('figures/abstract_fig2x2.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()