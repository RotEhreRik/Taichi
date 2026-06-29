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
N = 16
DT = 1.0 / 240.0
SUBSTEPS = 2
GRAVITY = 1.2
PARTICLE_RADIUS = 4.0

# ============================================================
# Paramètres audio
# ============================================================

SAMPLE_RATE = 44100
AUDIO_DURATION = 0.22
AUDIO_VOLUME = 0.10
# TIMBRE = "warm_synth"
TIMBRE = "pluck"

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

ROOT_NOTE_HZ = 130.81                    # AJOUTÉ : Do3
CHORD_INTERVALS = [0, 4, 7, 11]         # AJOUTÉ : accord majeur 7
NB_OCTAVES = 4                           # AJOUTÉ

def build_chord_scale(root_hz, intervals, nb_octaves):
    # AJOUTÉ
    notes = []
    for octv in range(nb_octaves):
        for interval in intervals:
            semitones = interval + 12 * octv
            freq = root_hz * (2.0 ** (semitones / 12.0))
            notes.append(freq)
    return notes

NOTES_HZ = build_chord_scale(ROOT_NOTE_HZ, CHORD_INTERVALS, NB_OCTAVES)  # MODIFIÉ

NOTE_COUNT = len(NOTES_HZ)   # AJOUTÉ

# ============================================================
# Champs Taichi
# ============================================================

pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
old_pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N)

hit = ti.field(dtype=ti.i32, shape=N)
impact_event = ti.field(dtype=ti.i32, shape=N)  # AJOUTÉ
particle_note_index = ti.field(dtype=ti.i32, shape=N)   # AJOUTÉ

image = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

# ============================================================
# Couleurs
# ============================================================

BG_COLOR = ti.Vector([16 / 255, 20 / 255, 23 / 255])
CURVE_COLOR = ti.Vector([102 / 255, 204 / 255, 255 / 255])
PARTICLE_COLOR = ti.Vector([1.0, 224 / 255, 138 / 255])
HIT_COLOR = ti.Vector([1.0, 112 / 255, 112 / 255])

# ============================================================
# Courbe analytique : y = f(x)
# ============================================================

C_CONST = 0.25
B_CONST = -6.00
A_CONST = 40.00


@ti.func
def curve_y(x):
    return A_CONST * (x-0.5) ** 4 + B_CONST * (x-0.5) ** 2 + C_CONST


@ti.func
def curve_dy(x):
    return 4 * A_CONST * (x-0.5) ** 3 + 2 * B_CONST * (x-0.5)


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
        old_pos[i] = pos[i]
        vel[i] = ti.Vector([
            0.25 * (ti.random() - 0.5),
            -0.15 * ti.random()
        ])
        hit[i] = 0
        impact_event[i] = 0  # AJOUTÉ
        particle_note_index[i] = i % len(NOTES_HZ)  # AJOUTÉ


# ============================================================
# Simulation
# ============================================================

@ti.kernel
def step():
    for i in range(N):
        hit[i] = 0
        old_pos[i] = pos[i]

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

        old_yc = curve_y(old_pos[i].x)
        new_yc = curve_y(pos[i].x)

        crossed = (old_pos[i].y > old_yc) and (pos[i].y <= new_yc)

        if crossed:
            n = curve_normal(pos[i].x)

            pos[i].y = new_yc + 1e-4

            vn = vel[i].dot(n)
            if vn < 0.0:
                vel[i] = vel[i] - 2.0 * vn * n

            hit[i] = 1
            impact_event[i] = 1  # AJOUTÉ : événement audio persistant

        elif pos[i].y <= new_yc:
            pos[i].y = new_yc + 1e-4


@ti.kernel
def clear_impact_events():
    # AJOUTÉ
    for i in range(N):
        impact_event[i] = 0


# ============================================================
# Rendu image Taichi
# ============================================================

@ti.kernel
def clear_image():
    for i, j in image:
        image[i, j] = BG_COLOR


@ti.kernel
def draw_curve():
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

# pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=8192)  # MODIFIÉ
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
        # env = adsr_envelope(N, sample_rate, 0.004, 0.06, 0.55, 0.10)
        env = adsr_envelope(n, sample_rate, 0.008, 0.07, 0.50, 0.12)  # MODIFIÉ
    wave = np.tanh(1.4 * wave)
    mono = (volume * wave * env).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    audio = np.clip(stereo * 32767.0, -32767, 32767).astype(np.int16)

    return pygame.sndarray.make_sound(audio)


tone_bank = [make_tone(f) for f in NOTES_HZ]


def play_note(note_index):
    sound = tone_bank[note_index % len(tone_bank)]
    ch = pygame.mixer.find_channel()
    if ch is not None:
        ch.play(sound)


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
    # Copies GPU -> CPU
    # --------------------------------------------------------
    t0 = time.perf_counter()
    h = hit.to_numpy()
    e = impact_event.to_numpy()  # AJOUTÉ
    p = pos.to_numpy()
    t_copy_acc += time.perf_counter() - t0
    particle_notes_np = particle_note_index.to_numpy()  # AJOUTÉ

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------
    t0 = time.perf_counter()
    if ENABLE_AUDIO:
        impact_indices = np.where(e == 1)[0]  # MODIFIÉ
        for i in impact_indices:
            note_index = int(particle_notes_np[i])  # MODIFIÉ
            play_note(note_index)
        clear_impact_events()  # AJOUTÉ
    t_audio_acc += time.perf_counter() - t0

    # --------------------------------------------------------
    # Rendu
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
