# https://www.youtube.com/watch?v=nCg3aXn5F3M

import taichi as ti

# ============================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================

USE_GPU = True

# Commencez petit, puis montez progressivement :
# 500, 1000, 2000, 5000, 10000...
N = 300
NTRACE = 3

MSOL = 1e3
MTER = MSOL / 1e1
MLUN = MTER / 64
MDEB = MTER / 1000

DT = 0.5 / MSOL

DAMPING = 0.999
DAMPING = 1e-12
WINDOW_RES = (1000, 1000)
FPS_LIMIT = 1200

# Force répulsive "type charge identique"
# G_ATTRACTION = 0.00002
G_ATTRACTION = 0.00004
SOFTENING = 1e-5

# Rayon d'affichage
PARTICLE_RADIUS = 0.002
AGREG_RADIUS2 = 0.0040**2
KFROT = 1e-6

# >>> AJOUT : paramètres de traîne
TRAIL_PRIM_LENGTH = 300
TRAIL_SECD_LENGTH = 4
TRAIL_GAP = 5
TRAIL_LINE_WIDTH = 0.0008

# >>> AJOUT : nombre total de sommets utilisés pour tous les segments de traîne
NB_TRAIL_PRIM_VERTICES = 2 * N * (TRAIL_PRIM_LENGTH - 1)
NB_TRAIL_SECD_VERTICES = 2 * N * (TRAIL_SECD_LENGTH - 1)

# ============================================================
# INITIALISATION TAICHI
# ============================================================

ti.init(arch=ti.gpu if USE_GPU else ti.cpu)

# ============================================================
# CHAMPS TAICHI
# ============================================================

positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
previous_positions = ti.Vector.field(2, dtype=ti.f32, shape=N)  # >>> CONSERVÉ
velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)
accels = ti.Vector.field(2, dtype=ti.f32, shape=N)
densities = ti.field(dtype=ti.f32, shape=N)  # >>> MODIF : scalaire cohérent
masses = ti.field(dtype=ti.f32, shape=N)
colors = ti.Vector.field(3, dtype=ti.f32, shape=N)
radii = ti.field(dtype=ti.f32, shape=N)

# >>> AJOUT : historique des CartPoss
history = ti.Vector.field(2, dtype=ti.f32, shape=(TRAIL_PRIM_LENGTH, N))

# >>> AJOUT : buffer de sommets pour les segments de traîne
trail_prim_vertices = ti.Vector.field(2, dtype=ti.f32, shape=NB_TRAIL_PRIM_VERTICES)
trail_secd_vertices = ti.Vector.field(2, dtype=ti.f32, shape=NB_TRAIL_SECD_VERTICES)

# ============================================================
# INITIALISATION
# ============================================================





@ti.kernel
def init_particles():
    # SOLEIL
    positions[0] = ti.Vector([0.5, 0.5])
    previous_positions[0] = positions[0]
    velocities[0] = ti.Vector([0.0, 0.0])
    accels[0] = ti.Vector([0.0, 0.0])
    # masses[0] = 1.0e6
    masses[0] = MSOL
    colors[0] = ti.Vector([1.0, 1.0, 0.0])
    radii[0] = PARTICLE_RADIUS * 5
    densities[0] = 0.0

    # v1=0
    # v2=0

    # TERRE
    if N>=2:
        positions[1] = ti.Vector([0.75, 0.5])
        previous_positions[1] = positions[1]
        r1 = positions[1]-positions[0]
        v1 = ti.sqrt(G_ATTRACTION * masses[0] / r1.norm())
        velocities[1] = ti.Vector([0.0, v1])
        accels[1] = ti.Vector([0.0, 0.0])
        # masses[0] = 1.0e6
        masses[1] = MTER
        colors[1] = ti.Vector([0.0, 0.0, 1.0])
        radii[1] = PARTICLE_RADIUS * 2
        densities[1] = 0.0

    # LUNE
    if N>=3:
        positions[2] = ti.Vector([0.77, 0.5])
        previous_positions[2] = positions[2]
        r2 = positions[2]-positions[1]
        v2 = ti.sqrt(G_ATTRACTION * masses[1] / r2.norm())
        v1 = velocities[1][1]
        velocities[2] = ti.Vector([0.0, v2+v1])
        accels[2] = ti.Vector([0.0, 0.0])
        # masses[0] = 1.0e6
        masses[2] = MLUN
        colors[2] = ti.Vector([0.5, 0.5, 0.5])
        radii[2] = PARTICLE_RADIUS * 1
        densities[2] = 0.0

    for i in range(3, N):
        x = 0.15 + 0.7 * ti.random(dtype=ti.f32)
        # x=0.5
        y = 0.15 + 0.7 * ti.random(dtype=ti.f32)
        # y = 0.5
        positions[i] = ti.Vector([x, y])
        previous_positions[i] = positions[i]

        # vx = (ti.random(dtype=ti.f32) - 0.5) * 4.5
        # vy = (ti.random(dtype=ti.f32) - 0.5) * 4.5
        # CircAngles[i] = ti.Vector([vx, vy])
        ri = positions[i]-positions[0]
        ai = ti.atan2(ri[0], ri[1])
        vi = ti.sqrt(G_ATTRACTION * masses[0] / ri.norm())
        velocities[i] = ti.Vector([-vi*ti.cos(ai), vi*ti.sin(ai)])
        # CircAngles[i] = ti.Vector([vi*ti.sin(ai), vi*ti.cos(ai)])

        accels[i] = ti.Vector([0.0, 0.0])

        masses[i] = MDEB

        colors[i] = ti.Vector([
            0.4 + 0.6 * ti.random(dtype=ti.f32),
            0.5 + 0.5 * ti.random(dtype=ti.f32),
            0.7 + 0.3 * ti.random(dtype=ti.f32)
        ])

        radii[i] = PARTICLE_RADIUS
        densities[i] = 0.0





# ============================================================
# DIVERS
# ============================================================


# >>> AJOUT
@ti.kernel
def reset_history():
    for t, i in history:
        history[t, i] = positions[i]

# >>> AJOUT
@ti.kernel
def store_history(slot: ti.i32):
    for i in range(N):
        history[slot, i] = positions[i]

# >>> AJOUT
@ti.kernel
def build_prim_trail_vertices(history_head: ti.i32):
    # for i in range(N):
    for i in range(NTRACE):
        for age in range(TRAIL_PRIM_LENGTH - 1):
            slot0 = (history_head - 1 - age) % TRAIL_PRIM_LENGTH
            slot1 = (history_head - 2 - age) % TRAIL_PRIM_LENGTH

            base = 2 * (i * (TRAIL_PRIM_LENGTH - 1) + age)
            trail_prim_vertices[base + 0] = history[slot1, i]
            trail_prim_vertices[base + 1] = history[slot0, i]

@ti.kernel
def build_secd_trail_vertices(history_head: ti.i32):
    # for i in range(N):
    for i in range(NTRACE, N):
        for age in range(TRAIL_SECD_LENGTH - 1):
            slot0 = (history_head - 1 - age) % TRAIL_PRIM_LENGTH
            slot1 = (history_head - 2 - age) % TRAIL_PRIM_LENGTH

            base = 2 * (i * (TRAIL_SECD_LENGTH - 1) + age)
            trail_secd_vertices[base + 0] = history[slot1, i]
            trail_secd_vertices[base + 1] = history[slot0, i]
# ============================================================
# CALCUL DES FORCES : VERSION NAÏVE TOUT-À-TOUT
# ============================================================

@ti.kernel
def clear_accels():
    for i in range(N):
        accels[i] = ti.Vector([0.0, 0.0])

@ti.kernel
def compute_accels_naive():
    for i in range(N):
        pi = positions[i]
        mi = masses[i]
        vi = velocities[i]
        density = 0.0

        for j in range(i):
            pj = positions[j]
            mj = masses[j]

            # Vecteur de j vers i
            r = pi - pj

            # Distance avec régularisation pour éviter les divisions explosives
            dist2 = r.dot(r) + SOFTENING

            # density += 1 / dist2
            dist = ti.sqrt(dist2)
            density += (mi + mj) / (dist2 * dist)

        for j in range(i):
            pj = positions[j]
            mj = masses[j]

            # Vecteur de j vers i
            r = pi - pj

            # Distance avec régularisation pour éviter les divisions explosives
            dist2 = r.dot(r) + SOFTENING
            dist = ti.sqrt(dist2)

            # Répulsion de type 1 / r^2
            # Forme vectorielle : k * r / |r|^3
            # ai = - G_ATTRACTION * r / (dist2 * dist)
            kij = - G_ATTRACTION * r / (dist2 * dist)
            ai = kij * mj
            aj = kij * mi
            # ai = - G_ATTRACTION / density * mi * mj * r / (dist2 * dist)

            # ai /= (1e-5 * density)

            # Action-réaction
            accels[i] += ai
            accels[j] -= aj

        densities[i] = density

        # PolarAccs[i] += - KFROT * density * vi

def compute_energies():
    for i in range(N):
        pi = positions[i]
        mi = masses[i]
        vi = velocities[i]
        density = 0.0

        for j in range(i):
            pj = positions[j]
            mj = masses[j]

            # Vecteur de j vers i
            r = pi - pj

            # Distance avec régularisation pour éviter les divisions explosives
            dist2 = r.dot(r) + SOFTENING

            # density += 1 / dist2
            dist = ti.sqrt(dist2)
            density += (mi + mj) / (dist2 * dist)

        for j in range(i):
            pj = positions[j]
            mj = masses[j]

            # Vecteur de j vers i
            r = pi - pj

            # Distance avec régularisation pour éviter les divisions explosives
            dist2 = r.dot(r) + SOFTENING
            dist = ti.sqrt(dist2)

            # Répulsion de type 1 / r^2
            # Forme vectorielle : k * r / |r|^3
            # f = - G_ATTRACTION * r / (dist2 * dist)
            f = - G_ATTRACTION * mi * mj * r / (dist2 * dist)
            # f = - G_ATTRACTION / density * mi * mj * r / (dist2 * dist)

            # f /= (1e-5 * density)

            # Action-réaction
            accels[i] += f
            accels[j] -= f

        densities[i] = density

        accels[i] += - KFROT * density * vi

def compute_colisions():
    for i in range(N):
        pi = positions[i]
        mi = masses[i]
        vi = velocities[i]
        density = 0.0

        for j in range(i):
            pj = positions[j]
            mj = masses[j]

            # Vecteur de j vers i
            r = pi - pj

            # Distance avec régularisation pour éviter les divisions explosives
            dist2 = r.dot(r) + SOFTENING

            # density += 1 / dist2
            dist = ti.sqrt(dist2)
            density += (mi + mj) / (dist2 * dist)

        for j in range(i):
            pj = positions[j]
            mj = masses[j]

            # Vecteur de j vers i
            r = pi - pj

            # Distance avec régularisation pour éviter les divisions explosives
            dist2 = r.dot(r) + SOFTENING
            dist = ti.sqrt(dist2)

            # Répulsion de type 1 / r^2
            # Forme vectorielle : k * r / |r|^3
            # f = - G_ATTRACTION * r / (dist2 * dist)
            f = - G_ATTRACTION * mi * mj * r / (dist2 * dist)
            # f = - G_ATTRACTION / density * mi * mj * r / (dist2 * dist)

            # f /= (1e-5 * density)

            # Action-réaction
            accels[i] += f
            accels[j] -= f

        densities[i] = density

        accels[i] += - KFROT * density * vi
@ti.kernel
def apply_accels():
    for i in range(1, N):
        v = velocities[i]
        p = positions[i]
        d = densities[i]
        m = masses[i]
        previous_positions[i] = p

        # v = (1 - DAMPING * d) * v + DT * PolarAccs[i] / m
        # p = p + DT * 0.5*v
        # v = (1 - DAMPING) * v + DT * PolarAccs[i] / m
        v = v + DT * accels[i]
        p = p + DT * v
        # v = (1 - DAMPING) * v + DT * PolarAccs[i] / m
        velocities[i] = v
        positions[i] = p

@ti.kernel
def apply_accels_before():
    for i in range(1, N):
        velocities[i]+= 0.5 * DT * accels[i]
        positions[i] += DT * velocities[i]



@ti.kernel
def apply_accels_after():
    for i in range(1, N):
        velocities[i]+= 0.5 * DT * accels[i]


@ti.kernel
def agregate():
    for i in range(N):
        pi = positions[i]
        mi = masses[i]
        vi = velocities[i]

        for j in range(i):
            pj = positions[j]
            mj = masses[j]
            vj = velocities[j]

            # Vecteur de j vers i
            r = pi - pj

            # Distance avec régularisation pour éviter les divisions explosives
            dist2 = r.dot(r) + SOFTENING

            if dist2 < AGREG_RADIUS2:
                v = (mi * vi + mj * vj) / (mi + mj)
                p = (mi * pi + mj * pj) / (mi + mj)

                positions[i] = p
                positions[j] = p
                velocities[i] = v
                velocities[j] = v

@ti.kernel
def integrate():
    SLOWER_PRI = 1.0
    SLOWER_SEC = 1.0

    for i in range(1, N):
        v = velocities[i]
        p = positions[i]

        if p[0] < 0.0:
            # p[0] = 0.0
            p[0] = -p[0]
            v[0] = -v[0] / SLOWER_PRI
            v[1] = +v[1] / SLOWER_SEC
        elif p[0] > 1.0:
            # p[0] = 1.0
            p[0] = 2.0 - p[0]
            v[0] = -v[0] / SLOWER_PRI
            v[1] = +v[1] / SLOWER_SEC

        if p[1] < 0.0:
            # p[1] = 0.0
            p[1] = -p[1]
            v[0] = +v[0] / SLOWER_SEC
            v[1] = -v[1] / SLOWER_PRI
        elif p[1] > 1.0:
            # p[1] = 1.0
            p[1] = 2.0 - p[1]
            v[0] = +v[0] / SLOWER_SEC
            v[1] = -v[1] / SLOWER_PRI

        velocities[i] = v
        positions[i] = p

# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

import os
import sys
try:
    script_name = os.path.basename(__file__)
except NameError:
    script_name = os.path.basename(sys.argv[0])

def main():
    init_particles()
    reset_history()

    history_head = 0
    frame_counter = 0

    window = ti.ui.Window(
        name=f"Taichi naive all-to-all ({N} particules) | {script_name} |",
        res=WINDOW_RES,
        fps_limit=FPS_LIMIT
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    with gui.sub_window("Infos", 0.02, 0.02, 0.34, 0.24):
        gui.text(f"N = {N}")
        gui.text("Mode : interactions completes naives")
        gui.text("Complexite : O(N^2)")
        gui.text(f"TRAIL_LENGTH = {TRAIL_PRIM_LENGTH}")
        gui.text(f"TRAIL_GAP = {TRAIL_GAP}")
        gui.text("Touche R : reinitialiser")

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == 'r':
                init_particles()
                reset_history()
                history_head = 0
                frame_counter = 0

        clear_accels()
        # apply_accels_before()
        compute_accels_naive()
        apply_accels()
        # apply_accels_after()

        # agregate()
        integrate()

        if frame_counter % TRAIL_GAP == 0:
            store_history(history_head)
            history_head = (history_head + 1) % TRAIL_PRIM_LENGTH

        frame_counter += 1

        # canvas.set_background_color((0.02, 0.02, 0.03))

        build_prim_trail_vertices(history_head)  # >>> AJOUT
        canvas.lines(
            trail_prim_vertices,
            width=TRAIL_LINE_WIDTH,
            color=(0.60, 0.78, 1.00),
            # per_vertex_color=Colors,
            # per_line_color=Colors,
        )

        build_secd_trail_vertices(history_head)  # >>> AJOUT
        canvas.lines(
            trail_secd_vertices,
            width=TRAIL_LINE_WIDTH,
            color=(0.60, 0.78, 1.00),
            # per_vertex_color=Colors,
            # per_line_color=Colors,
        )


        canvas.circles(positions, radius=PARTICLE_RADIUS, per_vertex_color=colors)


        window.show()

if __name__ == "__main__":
    main()