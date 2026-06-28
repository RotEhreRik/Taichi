import math
import numpy as np
import taichi as ti
import pygame  # MODIFIÉ : remplace simpleaudio

ti.init(arch=ti.gpu)

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
# Paramètres audio
# ============================================================

SAMPLE_RATE = 44100                     # AJOUTÉ
AUDIO_DURATION = 0.22                  # AJOUTÉ
AUDIO_VOLUME = 0.30                    # AJOUTÉ
TIMBRE = "warm_synth"                  # AJOUTÉ : "sine", "soft_bell", "warm_synth", "pluck"

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
# Audio synthétique avec pygame
# ============================================================

# AJOUTÉ : initialisation explicite du mixer
pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
pygame.init()
pygame.mixer.init()
pygame.mixer.set_num_channels(32)

def adsr_envelope(n_samples, sample_rate, attack, decay, sustain_level, release):
    # AJOUTÉ
    env = np.zeros(n_samples, dtype=np.float32)

    a = int(attack * sample_rate)
    d = int(decay * sample_rate)
    r = int(release * sample_rate)

    remain = n_samples - a - d - r
    s = max(0, remain)

    i0 = 0
    i1 = min(n_samples, i0 + a)
    if i1 > i0:
        env[i0:i1] = np.linspace(0.0, 1.0, i1 - i0, endpoint=False)

    i0 = i1
    i1 = min(n_samples, i0 + d)
    if i1 > i0:
        env[i0:i1] = np.linspace(1.0, sustain_level, i1 - i0, endpoint=False)

    i0 = i1
    i1 = min(n_samples, i0 + s)
    if i1 > i0:
        env[i0:i1] = sustain_level

    i0 = i1
    i1 = n_samples
    if i1 > i0:
        start = env[i0 - 1] if i0 > 0 else sustain_level
        env[i0:i1] = np.linspace(start, 0.0, i1 - i0, endpoint=True)

    return env

def make_tone(freq, duration=AUDIO_DURATION, sample_rate=SAMPLE_RATE, volume=AUDIO_VOLUME, timbre=TIMBRE):
    # MODIFIÉ : nouvelle synthèse sonore paramétrable
    n = int(duration * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate

    if timbre == "sine":
        wave = np.sin(2.0 * np.pi * freq * t)
        env = adsr_envelope(n, sample_rate, 0.005, 0.03, 0.85, 0.05)

    elif timbre == "soft_bell":
        wave = (
            1.00 * np.sin(2.0 * np.pi * freq * t) +
            0.35 * np.sin(2.0 * np.pi * 2.01 * freq * t + 0.2) +
            0.18 * np.sin(2.0 * np.pi * 3.90 * freq * t + 0.5)
        )
        env = adsr_envelope(n, sample_rate, 0.002, 0.12, 0.20, 0.10)

    elif timbre == "pluck":
        wave = (
            0.90 * np.sin(2.0 * np.pi * freq * t) +
            0.30 * np.sin(2.0 * np.pi * 2.0 * freq * t) +
            0.12 * np.sin(2.0 * np.pi * 3.0 * freq * t)
        )
        env = adsr_envelope(n, sample_rate, 0.001, 0.05, 0.15, 0.08)

    else:  # "warm_synth"
        detune = 0.997
        wave = (
            0.55 * np.sin(2.0 * np.pi * freq * t) +
            0.40 * np.sin(2.0 * np.pi * (freq * detune) * t + 0.3) +
            0.22 * np.sin(2.0 * np.pi * 2.0 * freq * t) +
            0.10 * np.sin(2.0 * np.pi * 3.0 * freq * t)
        )
        env = adsr_envelope(n, sample_rate, 0.004, 0.06, 0.55, 0.10)

    # Légère mise en forme douce du son
    wave = np.tanh(1.4 * wave)  # AJOUTÉ

    # Application enveloppe
    mono = (volume * wave * env).astype(np.float32)

    # Stéréo simple
    left = mono
    right = mono
    stereo = np.stack([left, right], axis=1)

    # Conversion int16 pour pygame
    audio = np.clip(stereo * 32767.0, -32767, 32767).astype(np.int16)

    return pygame.sndarray.make_sound(audio)

# MODIFIÉ : banque de sons pygame
tone_bank = [make_tone(f) for f in NOTES_HZ]

def play_note(note_index):
    # MODIFIÉ : lecture via pygame
    sound = tone_bank[note_index % len(tone_bank)]
    sound.play()

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
    gui.text(f"Timbre: {TIMBRE}", pos=(0.02, 0.92), font_size=18, color=0xFFFFFF)
    gui.show()

pygame.mixer.quit()
pygame.quit()