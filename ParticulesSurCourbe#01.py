import math
import numpy as np
import taichi as ti
# MODIFIÉ : import simpleaudio as sa   # désactivé pour diagnostic crash audio

ti.init(arch=ti.cpu)

# ============================================================
# Paramètres généraux
# ============================================================

W, H = 1000, 700
N = 64
DT = 1.0 / 240.0
SUBSTEPS = 2
GRAVITY = 1.2
PARTICLE_RADIUS = 4.0

# ============================================================
# Notes de la gamme tempérée
# Exemple : gamme de Do majeur sur plusieurs octaves
# ============================================================

NOTES_HZ = [
    261.63, 293.66, 329.63, 349.23,
    392.00, 440.00, 493.88, 523.25
]

# ============================================================
# Champs Taichi
# ============================================================

pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
hit = ti.field(dtype=ti.i32, shape=N)

# ============================================================
# Courbe analytique : y = f(x)
# Domaine GUI : x,y dans [0,1]
# ============================================================

@ti.func
def curve_y(x):
    return 0.25 + 0.12 * ti.sin(8.0 * x)

@ti.func
def curve_dy(x):
    return 0.96 * ti.cos(8.0 * x)

@ti.func
def curve_normal(x):
    # Pour F(x,y)=y-f(x), on a grad(F)=(-f'(x), 1)
    n = ti.Vector([-curve_dy(x), 1.0])
    return n.normalized()

# ============================================================
# Initialisation
# ============================================================

@ti.kernel
def init_particles():
    for i in range(N):
        x = 0.08 + 0.84 * (i / max(1, N - 1))
        y = 0.78 + 0.08 * ti.random()
        pos[i] = ti.Vector([x, y])
        vel[i] = ti.Vector([
            0.25 * (ti.random() - 0.5),
            -0.15 * ti.random()
        ])
        hit[i] = 0

# ============================================================
# Simulation
# ============================================================

@ti.kernel
def step():
    for i in range(N):
        hit[i] = 0

        # Gravité uniforme vers le bas
        vel[i].y -= GRAVITY * DT

        # Avance
        pos[i] += vel[i] * DT

        # Rebonds sur les bords latéraux et supérieur
        if pos[i].x < 0.0:
            pos[i].x = 0.0
            vel[i].x = -vel[i].x

        if pos[i].x > 1.0:
            pos[i].x = 1.0
            vel[i].x = -vel[i].x

        if pos[i].y > 1.0:
            pos[i].y = 1.0
            vel[i].y = -vel[i].y

        # Collision avec la courbe
        yc = curve_y(pos[i].x)
        if pos[i].y <= yc:
            n = curve_normal(pos[i].x)

            # Reprojection simple au-dessus de la courbe
            pos[i].y = yc + 1e-4

            # Réflexion élastique seulement si la particule arrive sur la courbe
            vn = vel[i].dot(n)
            if vn < 0.0:
                vel[i] = vel[i] - 2.0 * vn * n
                hit[i] = 1

# ============================================================
# Génération audio
# ============================================================

# MODIFIÉ : bloc audio complètement neutralisé pour vérifier que le crash
#           vient bien de simpleaudio / Python 3.12 sous Windows.

def make_tone(freq, duration=0.12, sample_rate=44100, volume=0.25):
    return None  # AJOUTÉ : stub de diagnostic

tone_bank = [make_tone(f) for f in NOTES_HZ]

def play_note(note_index):
    pass  # AJOUTÉ : aucun son joué pendant le diagnostic

# ============================================================
# Préparation du dessin de la courbe
# ============================================================

def build_curve_segments(n_segments=512):
    xs = np.linspace(0.0, 1.0, n_segments + 1, dtype=np.float32)
    ys = 0.25 + 0.12 * np.sin(8.0 * xs)

    begin = np.stack([xs[:-1], ys[:-1]], axis=1)
    end = np.stack([xs[1:], ys[1:]], axis=1)
    return begin, end

curve_begin, curve_end = build_curve_segments()

# ============================================================
# Boucle principale
# ============================================================

init_particles()
gui = ti.GUI("Taichi - courbe analytique + particules musicales", res=(W, H), background_color=0x101417)

while gui.running:
    if gui.get_event(ti.GUI.ESCAPE):
        break

    for _ in range(SUBSTEPS):
        step()

    p = pos.to_numpy()
    h = hit.to_numpy()

    # Déclenchement audio sur impacts
    # Ici : une note dépend de l'abscisse de la particule
    impact_indices = np.where(h == 1)[0]
    for i in impact_indices:
        note_index = int(np.clip(p[i, 0] * len(NOTES_HZ), 0, len(NOTES_HZ) - 1))
        play_note(note_index)

    gui.clear(0x101417)

    # Dessin de la courbe
    gui.lines(begin=curve_begin, end=curve_end, radius=2.0, color=0x66CCFF)

    # Dessin des particules
    colors = np.full((N,), 0xFFE08A, dtype=np.uint32)
    colors[h == 1] = 0xFF7070
    gui.circles(p, radius=PARTICLE_RADIUS, color=colors)

    gui.text("ECHAP pour quitter", pos=(0.02, 0.96), font_size=18, color=0xFFFFFF)
    gui.show()