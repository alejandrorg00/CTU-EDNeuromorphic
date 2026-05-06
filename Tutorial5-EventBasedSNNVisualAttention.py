"""

Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script implements a saliency-based attention mechanism using event-driven data.
It processes a series of events representing two objects, visualizing their spatial attention
through a saliency map. The attention mechanism is designed to highlight the most significant
features of the scene, using configurable parameters that define the characteristics of the
attention arcs and kernels.

The script leverages PyTorch for efficient tensor operations and OpenCV for real-time visualization
of the saliency maps. The primary steps include loading event data, initializing the attention
network, and iterating through the events to update and visualize the saliency map in response
to incoming data.

Key components:
- `Config`: A class that holds the parameters for the attention mechanism.
- `initialise_attention`: A helper function to set up the attention network.
- `run_attention`: A function that computes the saliency map based on the current window of events.

This implementation aims to demonstrate how attention mechanisms can be applied to dynamic
visual data, enabling robots and systems to focus on relevant features in real-time.
"""

import numpy as np
from helpers.helpers import initialise_attention, run_attention
import torch
import cv2
import matplotlib.pyplot as plt

# Configuration class to store attention parameters
class Config:
    # Attention Parameters
    ATTENTION_PARAMS = {
        'size_krn': 16,  # Size of the kernel used in the attention mechanism
        'r0': 14,  # Radius shift from the center for the attention arc
        'rho': 0.05,  # Scale coefficient to control the arc length
        'theta': np.pi * 3 / 2,  # Angle to control the orientation of the arc
        'thetas': np.arange(0, 2 * np.pi, np.pi / 4),  # Array of angles for multi-directional attention
        'thick': 3,  # Thickness of the arc in the attention map
        'fltr_resize_perc': [2, 2],  # Resize percentage for filters
        'offsetpxs': 0,  # Offset in pixels (half the size of the kernel)
        'offset': (0, 0),  # Offset for attention positioning
        'num_pyr': 6,  # Number of pyramid levels in the attention network
        'tau_mem': 0.3,  # Memory time constant for the attention mechanism
        'stride': 1,  # Stride for attention processing
        'out_ch': 1  # Number of output channels
    }


# Set the device for PyTorch (using Metal Performance Shaders on macOS if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the configuration
config = Config()

# Load event data from a .npy file containing two objects
data = np.load('data/twoobjects/twoobjects.npy')
###
#x = data[:,0]   # x coordinates
#y = data[:,1]   # y coordinates
#p = data[:,2]   # polarity (0,1)
#t = data[:,3]   # time
###
# Extract coordinates and properties from the data
x, y, p, t = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2], data[:, 3] * 1e3

# Determine the resolution based on the maximum coordinates
max_x = x.max() + 1  # Maximum x coordinate + 1 for resolution
max_y = y.max() + 1  # Maximum y coordinate + 1 for resolution
resolution = (int(max_y), int(max_x))  # Resolution tuple for attention processing

# Initialize saliency map and coordinates for maximum saliency
saliency_map = np.zeros((max_y, max_x), dtype=np.float32)  # Saliency map initialized to zero
salmax_coords = np.zeros((2,), dtype=np.int32)  # Array to hold coordinates of maximum saliency

##### Attention Mechanism #####
# Initialize the attention modules with the specified device and parameters
net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

# Set the time window period for processing events (in milliseconds)
window_period = 100  # Time window in milliseconds
time = window_period  # Initialize the time variable
window = torch.zeros((1, max_y, max_x), dtype=torch.float32)  # Create a tensor to hold the current window of events

###
# --- before the loop ---
attention_xy       = []  # list of (x,y) peaks
pantilt_deltas     = []  # list of (Δx, Δy) between successive peaks
prev_x, prev_y     = 0, 0  # start at home (0,0)
first_frame    = None
###

# Iterate through the event data
for xi, yi, pi, ti in zip(x, y, p, t):
    if ti <= time:
        # If the event time is within the current time window, update the window
        window[0][yi][xi] = 255  # Mark the pixel corresponding to the event
    else:
        # If the event time exceeds the current time window, process the attention
        saliency_map[:], salmax_coords[:] = run_attention(window, net_attention, device, resolution,
                                                          config.ATTENTION_PARAMS['num_pyr'])
        
        ###
        # record the peak (convert row,col → x,y)
        peak_x = int(salmax_coords[1])
        peak_y = int(salmax_coords[0])
        attention_xy.append((peak_x, peak_y))
        # save first frame to plot
        if first_frame is None:
            first_frame = window[0].detach().cpu().numpy().copy()
        # compute delta from previous
        dx = peak_x - prev_x
        dy = peak_y - prev_y
        pantilt_deltas.append((dx, dy))
        # update “previous” for next iteration
        prev_x, prev_y = peak_x, peak_y
        ###

        # Apply a color map to the window for better visualization
        window_map_jet = cv2.applyColorMap(window.detach().cpu().numpy().squeeze(0).astype(np.uint8), cv2.COLORMAP_JET)
        # Add a title to the visualization
        cv2.putText(window_map_jet, 'Events map', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)

        # Draw a circle at the location of maximum saliency on the visualization
        cv2.circle(window_map_jet, (salmax_coords[1], salmax_coords[0]), 6, (255, 255, 255), 4)
        cv2.putText(window_map_jet, 'Visual Attention', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Display the events map and saliency map
        cv2.imshow('Events map and Saliency Map', window_map_jet)

        # Wait for a key press to update the display
        cv2.waitKey(1)

        # Increment the time by the window period for the next iteration
        time += window_period

        # Reset the window for the next time period
        window = torch.zeros((1, max_y, max_x), dtype=torch.float32)

# Clean up by closing the OpenCV window after processing all events
cv2.destroyAllWindows()

###
# %%
# --- after the loop: convert lists to arrays ---
attention_xy = np.array(attention_xy)     # shape (N,2)

# Compute true bounds from the data
H, W = first_frame.shape

fig, ax = plt.subplots(figsize=(6, 6))

# 1) draw connecting line
ax.plot(
    attention_xy[:,0], attention_xy[:,1],
    '-',                            
    color='royalblue',
    alpha=0.5,                      
    linewidth=2                     
)
# 2) scatter big markers on top
ax.scatter(
    attention_xy[:,0], attention_xy[:,1],
    s=80,                           
    color='royalblue',
    edgecolor='white',              
    linewidth=1
)

# Enforce 1:1 aspect ratio
ax.set_aspect('equal', 'box')

# Use actual image dimensions for limits
ax.set_xlim(0, W-1)
ax.set_ylim(H-1, 0)  # invert so y=0 is at top

# Axis labels
ax.set_xlabel('x (pixels)', fontsize=16)
ax.set_ylabel('y (pixels)', fontsize=16)

# Show ticks inward but hide tick labels
ax.tick_params(
    axis='both',
    which='both',
    direction='in',
    length=6,
    width=2,
    labelbottom=False,
    labelleft=False
)

# Thicken the box (spines)
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()

# --- Save figure as SVG and PDF at 600 dpi ---
fig.savefig('figures/ev_cup_att.svg', format='svg', dpi=600, bbox_inches='tight')
fig.savefig('figures/ev_cup_att.pdf', format='pdf', dpi=600, bbox_inches='tight')

plt.show()

# %%
# --- 2) pan/tilt delta over the first image ---
# Normalize time from 0→1 for color mapping
N = len(attention_xy)
t_vals = np.linspace(0, 1, N)

# Split into coordinates and deltas
xs  = attention_xy[:, 0]
ys  = attention_xy[:, 1]
dxs = xs[1:] - xs[:-1]
dys = ys[1:] - ys[:-1]

# Create figure & main axes
fig, ax = plt.subplots(figsize=(8, 6))

# 1) Event map: show the first frame in gray
ax.imshow(first_frame, cmap='gray', origin='upper')  # origin='upper' matches OpenCV

# 2) Attentional points: scatter with jet gradient over time
sc = ax.scatter(
    xs, ys,
    c=t_vals,
    cmap='winter',
    s=35,               # marker size
    edgecolor='none',   # no outline
    zorder=2
)

# 3) Pan-tilt movements: use quiver for consistent arrows
ax.quiver(
    xs[:-1], ys[:-1],  # arrow origins
    dxs, dys,          # arrow vectors
    t_vals[:-1],       # color by time
    cmap='winter',
    scale_units='xy',
    angles='xy',
    scale=1,
    width=0.004,        # shaft width
    headwidth=4,        # arrow head width
    headlength=7,       # arrow head length
    zorder=3
)

# Tidy up main plot
ax.axis('off')

# 4) Add a small colorbar just outside the top‐right corner
#    [left, bottom, width, height] in figure fraction coordinates
cax = fig.add_axes([1.0, 0.50, 0.02, 0.36])  
cbar = fig.colorbar(sc, cax=cax, ticks=[0.0, 1.0])
cbar.set_label("Time (normalized)", fontsize=10)
cbar.set_ticklabels(['0', 'T'])
cax.yaxis.set_tick_params(labelsize=9)

plt.tight_layout()

# --- Save figure as SVG and PDF at 600 dpi ---
#fig.savefig('figures/ev_cup.svg', format='svg', dpi=600, bbox_inches='tight')
#fig.savefig('figures/ev_cup.pdf', format='pdf', dpi=600, bbox_inches='tight')

plt.show()


# %%

# --- Load first frame from video ---
video_path = r"H:\Desktop\Github\CTU-EDNeuromorphic\data\YTvideo2events\IEBCS\data\cofeecup_5s_1000fps_720x480.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame_bgr = cap.read()
cap.release()
if not ret:
    raise IOError(f"Could not read first frame from {video_path}")

# Convert BGR (OpenCV) to RGB for matplotlib
first_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# --- Prepare attention data (assuming attention_xy exists) ---
attention_xy = np.array(attention_xy)  # shape (N,2)
N = len(attention_xy)
t_vals = np.linspace(0, 1, N)
xs = attention_xy[:, 0]
ys = attention_xy[:, 1]
dxs = xs[1:] - xs[:-1]
dys = ys[1:] - ys[:-1]

# --- Create figure with two subplots ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Top: original first frame
ax1.imshow(first_frame_rgb)
ax1.axis('off')
#ax1.set_title('Original First Frame')

# Bottom: attention overlaid on first_frame (as grayscale background)
# Convert first_frame_bgr to grayscale for background
first_frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
ax2.imshow(first_frame, cmap='gray', origin='upper')

# Scatter attention peaks with a colormap
sc = ax2.scatter(
    xs, ys,
    c=t_vals,
    cmap='winter',
    s=35,
    edgecolor='none',
    zorder=2
)

# Quiver for pan-tilt deltas, colored by time
ax2.quiver(
    xs[:-1], ys[:-1],
    dxs, dys,
    t_vals[:-1],
    cmap='winter',
    scale_units='xy',
    angles='xy',
    scale=1,
    width=0.004,
    headwidth=4,
    headlength=7,
    zorder=3
)

ax2.axis('off')

# Add a colorbar for time mapping
cax = fig.add_axes([0.8125, 0.235, 0.02, 0.25])
cbar = fig.colorbar(sc, cax=cax, ticks=[0.0, 1.0])
cbar.set_label("Time", fontsize=16)
cbar.set_ticklabels(['0', 't'], fontsize=14)

plt.tight_layout(rect=[0, 0, 0.9, 1.0])
# --- Save figure as SVG and PDF at 600 dpi ---
fig.savefig('figures/ev_cup.svg', format='svg', dpi=600, bbox_inches='tight')
fig.savefig('figures/ev_cup.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()

# %%
# assume `crops` is your list of H×W×3 uint8 images
first_three = crops[:3]
last_one    = crops[-1]

# Square figure: 4" × 4"
fig, axes = plt.subplots(1, 5, figsize=(4, 4))
plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95)

# First three crops
for i in range(3):
    ax = axes[i]
    ax.imshow(first_three[i])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{i+1}")

# Ellipsis
ax_mid = axes[3]
ax_mid.text(0.5, 0.5, "…", fontsize=32, ha="center", va="center")
ax_mid.set_aspect('equal')
ax_mid.axis('off')

# Last crop with title 'n'
ax_last = axes[4]
ax_last.imshow(last_one)
ax_last.set_aspect('equal')
ax_last.axis('off')
ax_last.set_title("n")

plt.show()


# %%

# === 1) Load first frame from video ===
video_path = r"H:\Desktop\Github\CTU-EDNeuromorphic\data\YTvideo2events\IEBCS\data\cofeecup_5s_1000fps_720x480.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame_bgr = cap.read()
cap.release()
if not ret:
    raise IOError(f"Could not read first frame from {video_path}")
first_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
H, W = first_frame_rgb.shape[:2]

# === 2) Prepare attention data ===
attention_xy = np.array(attention_xy)  # shape (N,2)

# === 3) Prepare ROIs/crops ===
first_three = crops[:3]
last_one    = crops[-1]

# === 4) Create figure with 1×3 layout ===
fig = plt.figure(figsize=(15, 5))
outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3, left=0.05, right=0.95, top=0.88)

# --- Subplot 1: Setup ---
ax1 = fig.add_subplot(outer[0])
ax1.imshow(first_frame_rgb)
ax1.axis('off')
ax1.set_title("Setup", fontweight='bold', color='black',fontsize=18, pad=50)

# --- Subplot 2: Saccadic path ---
ax2 = fig.add_subplot(outer[1])
ax2.plot(
    attention_xy[:,0], attention_xy[:,1],
    '-', linewidth=2, alpha=0.5, color='royalblue'
)
ax2.scatter(
    attention_xy[:,0], attention_xy[:,1],
    s=80, edgecolors='white', linewidth=1, color='royalblue'
)
ax2.set_aspect('equal', 'box')
ax2.set_xlim(0, W-1)
ax2.set_ylim(H-1, 0)
ax2.tick_params(
    axis='both', which='both',
    direction='in', length=6, width=2,
    labelbottom=False, labelleft=False
)
for spine in ax2.spines.values():
    spine.set_linewidth(2)
ax2.set_xlabel('x (480 px)', fontsize=16)
ax2.set_ylabel('y (720 px)', fontsize=16)
ax2.set_title(
    r"$\mathbf{Saccades\,(pan\ and\ tilt\ movements)}$" "\n"
    r"$\{\ (\Delta x_1,\Delta y_1),\dots,(\Delta x_n,\Delta y_n)\ \}$",
    fontweight='bold', color='royalblue', pad=35, fontsize=18
)

# --- Subplot 3: Regions of interest (ROIs) ---
ax3 = fig.add_subplot(outer[2])
ax3.axis('off')
ax3.set_title("Regions of interest (ROIs)", fontweight='bold', color='crimson',fontsize=18, pad=-20)

# carve out a 2×2 block inside ax3's area
inner = gridspec.GridSpecFromSubplotSpec(
    2, 2,
    subplot_spec=outer[2],
    wspace=0.05,    # tightened as requested
    hspace=-0.4     # negative vertical gap
)

# ROI_1 (top-left)
ax31 = fig.add_subplot(inner[0, 0])
ax31.imshow(first_three[0])
ax31.axis('off')
ax31.set_title(r'$\mathrm{ROI}_1$', fontsize=16, pad=0,color='crimson')

# ROI_2 (top-right)
ax32 = fig.add_subplot(inner[0, 1])
ax32.imshow(first_three[1])
ax32.axis('off')
ax32.set_title(r'$\mathrm{ROI}_2$', fontsize=16, pad=0,color='crimson')

# ROI_3 (bottom-left)
ax33 = fig.add_subplot(inner[1, 0])
ax33.imshow(first_three[2])
ax33.axis('off')
ax33.set_title(r'$\mathrm{ROI}_3$', fontsize=16, pad=0,color='crimson')

# ROI_n (bottom-right)
ax34 = fig.add_subplot(inner[1, 1])
ax34.imshow(last_one)
ax34.axis('off')
ax34.set_title(r'$\mathrm{ROI}_n$', fontsize=16, pad=0,color='crimson')


# --- Save figure as SVG and PDF at 600 dpi ---
fig.savefig('figures/abstract_fig.svg', format='svg', dpi=600, bbox_inches='tight')
fig.savefig('figures/abstract_fig.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()
