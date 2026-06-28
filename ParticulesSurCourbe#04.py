import math
import time
import numpy as np
import taichi as ti
import pygame

ti.init(arch=ti.gpu)

# ============================================================
# Paramètres généraux
# ============================================================

W, H = 1000, 700
N = 1
DT = 1.0 / 240.0
SUBSTEPS = 2
GRAVITY = 1.2
PARTICLE_RADIUS = 4.0

# ============================================================
# Paramètres audio
# ============================================================

SAMPLE_RATE = 44100
AUDIO_DURATION = 0.22
AUDIO_VOLUME = 0.30
TIMBRE = "warm_synth"

# ============================================================
# Profilage / options de test
# ============================================================

PROFILE = True
PROFILE_EVERY = 240
ENABLE_RENDER = True
ENABLE_AUDIO = True

# ============================================================
# Notes de la gamme tempérée
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

# AJOUTÉ : framebuffer RGB pour fast_gui + set_image
image = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

# ============================================================
# Couleurs
# ============================================================

BG_COLOR = ti.Vector([16 / 255, 20 / 255, 23 / 255])          # AJOUTÉ
CURVE_COLOR = ti.Vector([102 / 255, 204 / 255, 255 / 255])    # AJOUTÉ
PARTICLE_COLOR = ti.Vector([1.0, 224 / 255, 138 / 255])       # AJOUTÉ
HIT_COLOR = ti.Vector([1.0, 112 / 255, 112 / 255])            # AJOUTÉ

# ============================================================
# Courbe analytique : y = f(x)
# ============================================================

@ti.func
def curve_y(x):
    return 0.25 + 0.12 * ti.sin(8.0 * x)

@ti.func
def curve_dy(x):
    return 0.96 * ti.cos(8.0 * x)

@ti.func
def curve_normal(x):
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

        vel[i].y -= GRAVITY * DT
        pos[i] += vel[i] * DT

        if pos[i].x < 0.0:
            pos[i].x = 0.0
            vel[i].x = -vel[i].x

        if pos[i].x > 1.0:
            pos[i].x = 1.0
            vel[i].x = -vel[i].x

        if pos[i].y > 1.0:
            pos[i].y = 1.0
            vel[i].y = -vel[i].y

        yc = curve_y(pos[i].x)
        if pos[i].y <= yc:
            n = curve_normal(pos[i].x)
            pos[i].y = yc + 1e-4
            vn = vel[i].dot(n)
            if vn < 0.0:
                vel[i] = vel[i] - 2.0 * vn * n
                hit[i] = 1

# ============================================================
# Rendu image Taichi
# ============================================================

@ti.kernel
def clear_image():
    # AJOUTÉ
    for i, j in image:
        image[i, j] = BG_COLOR

@ti.kernel
def draw_curve():
    # AJOUTÉ
    thickness = 2.0 / H
    for i, j in image:
        x = (i + 0.5) / W
        y = (j + 0.5) / H
        yc = curve_y(x)
        d = ti.abs(y - yc)
        if d < thickness:
            alpha = 1.0 - d / thickness
            image[i, j] = image[i, j] * (1.0 - alpha) + CURVE_COLOR * alpha

@ti.kernel
def draw_particles():
    # AJOUTÉ
    r_px = int(PARTICLE_RADIUS) + 1
    for p_idx in range(N):
        cx = int(pos[p_idx].x * W)
        cy = int(pos[p_idx].y * H)
        color = PARTICLE_COLOR
        if hit[p_idx] == 1:
            color = HIT_COLOR

        for dx, dy in ti.ndrange((-r_px, r_px + 1), (-r_px, r_px + 1)):
            x = cx + dx
            y = cy + dy
            if 0 <= x < W and 0 <= y < H:
                dist2 = dx * dx + dy * dy
                if dist2 <= PARTICLE_RADIUS * PARTICLE_RADIUS:
                    image[x, y] = color

# ============================================================
# Audio synthétique avec pygame
# ============================================================

pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
pygame.init()
pygame.mixer.init()
pygame.mixer.set_num_channels(32)

def adsr_envelope(n_samples, sample_rate, attack, decay, sustain_level, release):
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

    else:
        detune = 0.997
        wave = (
            0.55 * np.sin(2.0 * np.pi * freq * t) +
            0.40 * np.sin(2.0 * np.pi * (freq * detune) * t + 0.3) +
            0.22 * np.sin(2.0 * np.pi * 2.0 * freq * t) +
            0.10 * np.sin(2.0 * np.pi * 3.0 * freq * t)
        )
        env = adsr_envelope(n, sample_rate, 0.004, 0.06, 0.55, 0.10)

    wave = np.tanh(1.4 * wave)
    mono = (volume * wave * env).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    audio = np.clip(stereo * 32767.0, -32767, 32767).astype(np.int16)

    return pygame.sndarray.make_sound(audio)

tone_bank = [make_tone(f) for f in NOTES_HZ]

def play_note(note_index):
    sound = tone_bank[note_index % len(tone_bank)]
    sound.play()

# ============================================================
# Profilage : accumulateurs
# ============================================================

frame_count = 0
t_step_acc = 0.0
t_copy_acc = 0.0
t_audio_acc = 0.0
t_draw_acc = 0.0
t_total_acc = 0.0

# ============================================================
# Boucle principale
# ============================================================

init_particles()

# MODIFIÉ : fast_gui=True pour bénéficier de la voie rapide avec set_image
gui = ti.GUI(
    "Taichi - courbe analytique + particules musicales",
    res=(W, H),
    fast_gui=True
)

while gui.running:
    t0_total = time.perf_counter()

    if gui.get_event(ti.GUI.ESCAPE):
        break

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------
    t0 = time.perf_counter()
    for _ in range(SUBSTEPS):
        step()
    ti.sync()
    t_step_acc += time.perf_counter() - t0

    # --------------------------------------------------------
    # Copies GPU -> CPU (gardées uniquement pour l'audio)
    # --------------------------------------------------------
    t0 = time.perf_counter()
    h = hit.to_numpy()          # MODIFIÉ : on ne copie plus CartPoss pour le rendu
    p = pos.to_numpy()          # gardé ici seulement pour mapper note <- x
    t_copy_acc += time.perf_counter() - t0

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------
    t0 = time.perf_counter()
    if ENABLE_AUDIO:
        impact_indices = np.where(h == 1)[0]
        for i in impact_indices:
            note_index = int(np.clip(p[i, 0] * len(NOTES_HZ), 0, len(NOTES_HZ) - 1))
            play_note(note_index)
    t_audio_acc += time.perf_counter() - t0

    # --------------------------------------------------------
    # Rendu Taichi -> image -> set_image
    # --------------------------------------------------------
    t0 = time.perf_counter()
    if ENABLE_RENDER:
        clear_image()
        draw_curve()
        draw_particles()
        gui.set_image(image)
        gui.show()
    t_draw_acc += time.perf_counter() - t0

    # --------------------------------------------------------
    # Statistiques
    # --------------------------------------------------------
    dt_total = time.perf_counter() - t0_total
    t_total_acc += dt_total
    frame_count += 1

    if PROFILE and frame_count % PROFILE_EVERY == 0:
        avg_step = 1000.0 * t_step_acc / PROFILE_EVERY
        avg_copy = 1000.0 * t_copy_acc / PROFILE_EVERY
        avg_audio = 1000.0 * t_audio_acc / PROFILE_EVERY
        avg_draw = 1000.0 * t_draw_acc / PROFILE_EVERY
        avg_total = 1000.0 * t_total_acc / PROFILE_EVERY
        fps = PROFILE_EVERY / t_total_acc

        print("--------------------------------------------------")
        print(f"Frames profilées : {PROFILE_EVERY}")
        print(f"step() + sync      : {avg_step:8.3f} ms/frame")
        print(f"to_numpy()         : {avg_copy:8.3f} ms/frame")
        print(f"audio              : {avg_audio:8.3f} ms/frame")
        print(f"draw + show        : {avg_draw:8.3f} ms/frame")
        print(f"total              : {avg_total:8.3f} ms/frame")
        print(f"FPS                : {fps:8.2f}")
        print("--------------------------------------------------")

        t_step_acc = 0.0
        t_copy_acc = 0.0
        t_audio_acc = 0.0
        t_draw_acc = 0.0
        t_total_acc = 0.0

pygame.mixer.quit()
pygame.quit()