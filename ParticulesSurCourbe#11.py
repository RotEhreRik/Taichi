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
N_NOTE = 15
N_CHORD = 1
DT = 1.0 / 240.0
SUBSTEPS = 2
GRAVITY = 1.2
PARTICLE_RADIUS = 4.0
CHORD_PARTICLE_RADIUS = 6.0

# ============================================================
# Paramètres du set "accord"
# ============================================================

CHORD_SPEED_SCALE = 0.35
CHORD_GRAVITY_SCALE = 0.45
CHORD_COLOR_SPEED = 0.08
CHORD_PARTICLES_PLAY_SOUND = False

# ============================================================
# Paramètres HUD / debug
# ============================================================

SHOW_HUD = False
PAUSED = False
ENERGY_HISTORY_SIZE = 240
PARTICLE_MASS = 1.0
SHOW_ENERGY_GRAPH = False   # AJOUTÉ : laisser False pour préserver les FPS

# ============================================================
# Paramètres audio
# ============================================================

SAMPLE_RATE = 44100
AUDIO_DURATION = 0.22
AUDIO_VOLUME = 0.10
TIMBRE = "pluck"

# ============================================================
# Profilage / options de test
# ============================================================

PROFILE = True
PROFILE_EVERY = 240
ENABLE_RENDER = True
ENABLE_AUDIO = True

# ============================================================
# Bibliothèque harmonique
# ============================================================

ROOT_NOTE_HZ = 130.81
NB_OCTAVES = 4

CHORD_LIBRARY = [
    ("maj7", [0, 4, 7, 11]),
    ("min7", [0, 3, 7, 10]),
    ("sus2", [0, 2, 7, 12]),
    ("add9", [0, 4, 7, 14]),
    ("pentatonic", [0, 2, 4, 7, 9]),
]

def build_chord_scale(root_hz, intervals, nb_octaves):
    notes = []
    for octv in range(nb_octaves):
        for interval in intervals:
            semitones = interval + 12 * octv
            freq = root_hz * (2.0 ** (semitones / 12.0))
            notes.append(freq)
    return notes

def build_all_chord_scales(root_hz, chord_library, nb_octaves):
    scales = []
    for chord_name, intervals in chord_library:
        notes = build_chord_scale(root_hz, intervals, nb_octaves)
        scales.append((chord_name, notes))
    return scales

ALL_CHORD_SCALES = build_all_chord_scales(ROOT_NOTE_HZ, CHORD_LIBRARY, NB_OCTAVES)
current_chord_index = 0
current_chord_name, NOTES_HZ = ALL_CHORD_SCALES[current_chord_index]
NOTE_COUNT = len(NOTES_HZ)

# ============================================================
# Champs Taichi : set notes
# ============================================================

note_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_NOTE)
note_old_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_NOTE)
note_vel = ti.Vector.field(2, dtype=ti.f32, shape=N_NOTE)

note_hit = ti.field(dtype=ti.i32, shape=N_NOTE)
note_impact_event = ti.field(dtype=ti.i32, shape=N_NOTE)
note_particle_index = ti.field(dtype=ti.i32, shape=N_NOTE)
note_particle_color = ti.Vector.field(3, dtype=ti.f32, shape=N_NOTE)

energy_kin_field = ti.field(dtype=ti.f32, shape=())
energy_pot_field = ti.field(dtype=ti.f32, shape=())
energy_tot_field = ti.field(dtype=ti.f32, shape=())

# ============================================================
# Champs Taichi : set accords
# ============================================================

chord_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_CHORD)
chord_old_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_CHORD)
chord_vel = ti.Vector.field(2, dtype=ti.f32, shape=N_CHORD)

chord_hit = ti.field(dtype=ti.i32, shape=N_CHORD)
chord_impact_event = ti.field(dtype=ti.i32, shape=N_CHORD)
chord_change_event = ti.field(dtype=ti.i32, shape=())

# ============================================================
# Courbe analytique principale : set notes
# ============================================================

C_CONST = 0.25
B_CONST = -6.00
A_CONST = 40.00

@ti.func
def curve_y(x):
    return A_CONST * (x - 0.5) ** 4 + B_CONST * (x - 0.5) ** 2 + C_CONST

@ti.func
def curve_dy(x):
    return 4 * A_CONST * (x - 0.5) ** 3 + 2 * B_CONST * (x - 0.5)

@ti.func
def curve_normal(x):
    n = ti.Vector([-curve_dy(x), 1.0])
    return n.normalized()

# ============================================================
# Courbe analytique secondaire : set accords
# ============================================================

@ti.func
def curve2_y(x):
    return 0.72 + 0.05 * ti.sin(10.0 * x)

@ti.func
def curve2_dy(x):
    return 0.50 * ti.cos(10.0 * x)

@ti.func
def curve2_normal(x):
    n = ti.Vector([-curve2_dy(x), 1.0])
    return n.normalized()

# ============================================================
# Versions Python des courbes pour le rendu direct GUI
# ============================================================

def curve_y_py(x):
    return A_CONST * (x - 0.5) ** 4 + B_CONST * (x - 0.5) ** 2 + C_CONST

def curve2_y_py(x):
    return 0.72 + 0.05 * math.sin(10.0 * x)

def build_curve_segments(func_y, n_segments=600):
    xs = np.linspace(0.0, 1.0, n_segments + 1, dtype=np.float32)
    ys = np.array([func_y(float(x)) for x in xs], dtype=np.float32)
    begin = np.stack([xs[:-1], ys[:-1]], axis=1)
    end = np.stack([xs[1:], ys[1:]], axis=1)
    return begin, end

curve1_begin, curve1_end = build_curve_segments(curve_y_py, n_segments=600)
curve2_begin, curve2_end = build_curve_segments(curve2_y_py, n_segments=600)

# ============================================================
# Coloration
# ============================================================

@ti.func
def clamp01(x):
    return ti.max(0.0, ti.min(1.0, x))

@ti.func
def hsv_to_rgb(h, s, v):
    h = h - ti.floor(h)
    c = v * s
    hp = h * 6.0
    x = c * (1.0 - ti.abs(hp % 2.0 - 1.0))

    r1, g1, b1 = 0.0, 0.0, 0.0

    if hp < 1.0:
        r1, g1, b1 = c, x, 0.0
    elif hp < 2.0:
        r1, g1, b1 = x, c, 0.0
    elif hp < 3.0:
        r1, g1, b1 = 0.0, c, x
    elif hp < 4.0:
        r1, g1, b1 = 0.0, x, c
    elif hp < 5.0:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x

    m = v - c
    return ti.Vector([
        clamp01(r1 + m),
        clamp01(g1 + m),
        clamp01(b1 + m)
    ])

def hsv_to_rgb_cpu(h, s, v):
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6

    if i == 0:
        return np.array([v, t, p], dtype=np.float32)
    elif i == 1:
        return np.array([q, v, p], dtype=np.float32)
    elif i == 2:
        return np.array([p, v, t], dtype=np.float32)
    elif i == 3:
        return np.array([p, q, v], dtype=np.float32)
    elif i == 4:
        return np.array([t, p, v], dtype=np.float32)
    else:
        return np.array([v, p, q], dtype=np.float32)

# ============================================================
# Initialisation
# ============================================================

@ti.kernel
def init_note_particles():
    for i in range(N_NOTE):
        x = 0.08 + 0.84 * (i / max(1, N_NOTE - 1))
        # y = 0.78 + 0.08 * ti.random()
        y = 0.78
        note_pos[i] = ti.Vector([x, y])
        note_old_pos[i] = note_pos[i]
        # note_vel[i] = ti.Vector([
        #     0.25 * (ti.random() - 0.5),
        #     -0.15 * ti.random()
        # ])
        note_vel[i] = ti.Vector([0.0, 0.0])
        note_hit[i] = 0
        note_impact_event[i] = 0
        note_particle_index[i] = 0

        hue = i / max(1, N_NOTE)
        note_particle_color[i] = hsv_to_rgb(hue, 0.85, 1.00)

@ti.kernel
def init_chord_particles():
    for i in range(N_CHORD):
        chord_pos[i] = ti.Vector([0.15 + 0.20 * i, 0.92])
        chord_old_pos[i] = chord_pos[i]
        chord_vel[i] = ti.Vector([
            CHORD_SPEED_SCALE * 0.08 * (ti.random() - 0.5),
            -CHORD_SPEED_SCALE * 0.05 * ti.random()
        ])
        chord_hit[i] = 0
        chord_impact_event[i] = 0

    chord_change_event[None] = 0

# ============================================================
# Simulation
# ============================================================

@ti.kernel
def step_note_particles():
    for i in range(N_NOTE):
        note_hit[i] = 0
        note_old_pos[i] = note_pos[i]

        note_vel[i].y -= GRAVITY * DT
        note_pos[i] += note_vel[i] * DT

        if note_pos[i].x < 0.0:
            note_pos[i].x = 0.0
            note_vel[i].x = -note_vel[i].x

        if note_pos[i].x > 1.0:
            note_pos[i].x = 1.0
            note_vel[i].x = -note_vel[i].x

        if note_pos[i].y > 1.0:
            note_pos[i].y = 1.0
            note_vel[i].y = -note_vel[i].y

        old_yc = curve_y(note_old_pos[i].x)
        new_yc = curve_y(note_pos[i].x)

        crossed = (note_old_pos[i].y > old_yc) and (note_pos[i].y <= new_yc)

        if crossed:
            n = curve_normal(note_pos[i].x)
            note_pos[i].y = new_yc + 1e-4

            vn = note_vel[i].dot(n)
            if vn < 0.0:
                note_vel[i] = note_vel[i] - 2.0 * vn * n

            note_hit[i] = 1
            note_impact_event[i] = 1

        elif note_pos[i].y <= new_yc:
            note_pos[i].y = new_yc + 1e-4

@ti.kernel
def step_chord_particles():
    for i in range(N_CHORD):
        chord_hit[i] = 0
        chord_old_pos[i] = chord_pos[i]

        chord_vel[i].y -= (GRAVITY * CHORD_GRAVITY_SCALE) * DT
        chord_pos[i] += chord_vel[i] * DT

        if chord_pos[i].x < 0.0:
            chord_pos[i].x = 0.0
            chord_vel[i].x = -chord_vel[i].x

        if chord_pos[i].x > 1.0:
            chord_pos[i].x = 1.0
            chord_vel[i].x = -chord_vel[i].x

        if chord_pos[i].y > 1.0:
            chord_pos[i].y = 1.0
            chord_vel[i].y = -chord_vel[i].y

        old_yc = curve2_y(chord_old_pos[i].x)
        new_yc = curve2_y(chord_pos[i].x)

        crossed = (chord_old_pos[i].y > old_yc) and (chord_pos[i].y <= new_yc)

        if crossed:
            n = curve2_normal(chord_pos[i].x)
            chord_pos[i].y = new_yc + 1e-4

            vn = chord_vel[i].dot(n)
            if vn < 0.0:
                chord_vel[i] = chord_vel[i] - 2.0 * vn * n

            chord_hit[i] = 1
            chord_impact_event[i] = 1
            chord_change_event[None] = 1

        elif chord_pos[i].y <= new_yc:
            chord_pos[i].y = new_yc + 1e-4

@ti.kernel
def clear_impact_events():
    for i in range(N_NOTE):
        note_impact_event[i] = 0
    for i in range(N_CHORD):
        chord_impact_event[i] = 0

@ti.kernel
def compute_energy():
    e_kin = 0.0
    e_pot = 0.0

    for i in range(N_NOTE):
        v2 = note_vel[i].dot(note_vel[i])
        e_kin += 0.5 * PARTICLE_MASS * v2
        e_pot += PARTICLE_MASS * GRAVITY * note_pos[i].y

    for i in range(N_CHORD):
        v2 = chord_vel[i].dot(chord_vel[i])
        e_kin += 0.5 * PARTICLE_MASS * v2
        e_pot += PARTICLE_MASS * GRAVITY * chord_pos[i].y

    energy_kin_field[None] = e_kin
    energy_pot_field[None] = e_pot
    energy_tot_field[None] = e_kin + e_pot

# ============================================================
# Audio synthétique avec pygame
# ============================================================

pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=8192)
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
        env = adsr_envelope(n, sample_rate, 0.008, 0.07, 0.50, 0.12)

    wave = np.tanh(1.4 * wave)
    mono = (volume * wave * env).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    audio = np.clip(stereo * 32767.0, -32767, 32767).astype(np.int16)

    return pygame.sndarray.make_sound(audio)

tone_bank = []

def play_note(note_index):
    sound = tone_bank[note_index % len(tone_bank)]
    ch = pygame.mixer.find_channel()
    if ch is not None:
        ch.play(sound)

def apply_chord(chord_index):
    global current_chord_index, current_chord_name, NOTES_HZ, NOTE_COUNT, tone_bank

    current_chord_index = chord_index % len(ALL_CHORD_SCALES)
    current_chord_name, NOTES_HZ = ALL_CHORD_SCALES[current_chord_index]
    NOTE_COUNT = len(NOTES_HZ)
    tone_bank = [make_tone(f) for f in NOTES_HZ]

    note_indices = np.zeros(N_NOTE, dtype=np.int32)
    for i in range(N_NOTE):
        note_indices[i] = i % NOTE_COUNT

    note_particle_index.from_numpy(note_indices)

# ============================================================
# Énergie / instrumentation
# ============================================================

energy_history = []
energy_total = 0.0
energy_kin = 0.0
energy_pot = 0.0
energy_ref = None

def draw_energy_history(gui, values, x0=0.02, y0=0.62, w=0.26, h=0.12, color=0x66FFAA):
    if len(values) < 2:
        return

    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    pts = []
    n = len(values)
    for i, v in enumerate(values):
        x = x0 + w * (i / (n - 1))
        y = y0 + h * (1.0 - (v - vmin) / (vmax - vmin))
        pts.append([x, y])

    pts = np.array(pts, dtype=np.float32)
    gui.lines(pts[:-1], pts[1:], radius=1.0, color=color)

def reset_simulation_state():
    global energy_history, energy_total, energy_kin, energy_pot, energy_ref
    init_note_particles()
    init_chord_particles()
    apply_chord(0)
    energy_history = []
    energy_total = 0.0
    energy_kin = 0.0
    energy_pot = 0.0
    energy_ref = None

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

reset_simulation_state()

gui = ti.GUI(
    "Taichi - courbe analytique + particules musicales",
    res=(W, H),
    fast_gui=False
)

start_time = time.perf_counter()

while gui.running:
    t0_total = time.perf_counter()

    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
        elif gui.event.key == ti.GUI.SPACE:
            PAUSED = not PAUSED
        elif gui.event.key == 'h':
            SHOW_HUD = not SHOW_HUD
        elif gui.event.key == 'r':
            reset_simulation_state()
        elif gui.event.key == '-':
            GRAVITY = max(0.0, GRAVITY - 0.05)
        elif gui.event.key == '=':
            GRAVITY += 0.05
        elif gui.event.key == '[':
            CHORD_COLOR_SPEED = max(0.0, CHORD_COLOR_SPEED - 0.01)
        elif gui.event.key == ']':
            CHORD_COLOR_SPEED += 0.01
        elif gui.event.key == 'g':
            SHOW_ENERGY_GRAPH = not SHOW_ENERGY_GRAPH

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------
    t0 = time.perf_counter()
    if not PAUSED:
        for _ in range(SUBSTEPS):
            step_note_particles()
            step_chord_particles()
        ti.sync()
    t_step_acc += time.perf_counter() - t0

    # --------------------------------------------------------
    # Copies GPU -> CPU
    # --------------------------------------------------------
    t0 = time.perf_counter()
    note_events_np = note_impact_event.to_numpy()
    note_indices_np = note_particle_index.to_numpy()

    chord_evt = chord_change_event[None]

    compute_energy()
    energy_kin = float(energy_kin_field[None])
    energy_pot = float(energy_pot_field[None])
    energy_total = float(energy_tot_field[None])

    note_pos_np = note_pos.to_numpy()
    chord_pos_np = chord_pos.to_numpy()
    note_hit_np = note_hit.to_numpy()
    chord_hit_np = chord_hit.to_numpy()
    note_colors_np = note_particle_color.to_numpy()
    t_copy_acc += time.perf_counter() - t0

    # --------------------------------------------------------
    # Changement d'accord
    # --------------------------------------------------------
    if chord_evt == 1:
        apply_chord(current_chord_index + 1)
        chord_change_event[None] = 0

    if energy_ref is None:
        energy_ref = energy_total

    energy_history.append(energy_total)
    if len(energy_history) > ENERGY_HISTORY_SIZE:
        energy_history.pop(0)

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------
    t0 = time.perf_counter()
    if ENABLE_AUDIO and not PAUSED:
        impact_indices = np.where(note_events_np == 1)[0]
        for i in impact_indices:
            note_index = int(note_indices_np[i])
            play_note(note_index)

        if CHORD_PARTICLES_PLAY_SOUND:
            chord_events_np = chord_impact_event.to_numpy()
            chord_impacts = np.where(chord_events_np == 1)[0]
            for _ in chord_impacts:
                play_note(0)

        clear_impact_events()
    t_audio_acc += time.perf_counter() - t0

    # --------------------------------------------------------
    # Rendu direct GUI
    # --------------------------------------------------------
    t0 = time.perf_counter()
    if ENABLE_RENDER:
        elapsed = time.perf_counter() - start_time
        chord_hue = (CHORD_COLOR_SPEED * elapsed) % 1.0

        gui.clear(0x101417)

        gui.lines(begin=curve1_begin, end=curve1_end, radius=2, color=0x66CCFF)
        gui.lines(begin=curve2_begin, end=curve2_end, radius=2, color=0xFFB45A)

        note_colors_draw = note_colors_np.copy()
        note_hit_mask = (note_hit_np == 1)
        if np.any(note_hit_mask):
            note_colors_draw[note_hit_mask] = 0.35 * note_colors_draw[note_hit_mask] + 0.65 * 1.0

        gui.circles(note_pos_np, radius=PARTICLE_RADIUS, color=note_colors_draw)

        chord_rgb = hsv_to_rgb_cpu(chord_hue, 0.95, 1.0)
        chord_colors_draw = np.tile(chord_rgb[None, :], (N_CHORD, 1))
        chord_hit_mask = (chord_hit_np == 1)
        if np.any(chord_hit_mask):
            chord_colors_draw[chord_hit_mask] = 0.35 * chord_colors_draw[chord_hit_mask] + 0.65 * 1.0

        gui.circles(chord_pos_np, radius=CHORD_PARTICLE_RADIUS, color=chord_colors_draw)

        if SHOW_HUD:
            drift_abs = energy_total - energy_ref
            drift_rel = 0.0 if abs(energy_ref) < 1e-12 else drift_abs / energy_ref

            gui.text(f"Accord : {current_chord_name}", pos=(0.02, 0.97), color=0xFFFFFF)
            gui.text(f"Pause : {PAUSED}", pos=(0.02, 0.94), color=0xFFFFFF)
            gui.text(f"E totale : {energy_total:10.6f}", pos=(0.02, 0.91), color=0xFFFFFF)
            gui.text(f"Derive rel : {100.0 * drift_rel:+8.4f} %", pos=(0.02, 0.88), color=0xFFE080)
            gui.text(f"g={GRAVITY:.3f}  dt={DT:.6f}  substeps={SUBSTEPS}", pos=(0.02, 0.85), color=0xC0E0FF)
            gui.text("SPACE pause | r reset | h HUD | g graphe", pos=(0.02, 0.82), color=0xA0FFA0)

            if SHOW_ENERGY_GRAPH:
                draw_energy_history(gui, energy_history)

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
        print(f"Frames profilees : {PROFILE_EVERY}")
        print(f"Accord courant   : {current_chord_name}")
        print(f"E totale         : {energy_total:12.6f}")
        print(f"step() + sync    : {avg_step:8.3f} ms/frame")
        print(f"to_numpy()       : {avg_copy:8.3f} ms/frame")
        print(f"audio            : {avg_audio:8.3f} ms/frame")
        print(f"draw + show      : {avg_draw:8.3f} ms/frame")
        print(f"total            : {avg_total:8.3f} ms/frame")
        print(f"FPS              : {fps:8.2f}")
        print("--------------------------------------------------")

        t_step_acc = 0.0
        t_copy_acc = 0.0
        t_audio_acc = 0.0
        t_draw_acc = 0.0
        t_total_acc = 0.0

pygame.mixer.quit()
pygame.quit()