import math
import taichi as ti

# ============================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================

USE_GPU = True
N = 200_000          # Essayez 50_000, 100_000, 200_000, 500_000...
DT = 0.002
SUBSTEPS = 1
WINDOW_RES = (1000, 1000)
FPS_LIMIT = 120

# ============================================================
# INITIALISATION TAICHI
# ============================================================

ti.init(arch=ti.gpu if USE_GPU else ti.cpu)

# ============================================================
# DONNÉES
# ============================================================

pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
col = ti.Vector.field(3, dtype=ti.f32, shape=N)

# Paramètres globaux pilotables
time_value = ti.field(dtype=ti.f32, shape=())
mouse_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
mouse_strength = ti.field(dtype=ti.f32, shape=())

# ============================================================
# INITIALISATION
# ============================================================

@ti.kernel
def init_particles():
    for i in range(N):
        x = ti.random(dtype=ti.f32)
        y = ti.random(dtype=ti.f32)

        pos[i] = ti.Vector([x, y])

        # Petite vitesse initiale aléatoire
        vx = (ti.random(dtype=ti.f32) - 0.5) * 0.2 *4
        vy = (ti.random(dtype=ti.f32) - 0.5) * 0.2 *4
        vel[i] = ti.Vector([vx, vy])

        # Dégradé de couleur selon la position
        col[i] = ti.Vector([
            0.2 + 0.8 * x,
            0.3 + 0.7 * y,
            0.9 - 0.5 * x
        ])

# ============================================================
# SIMULATION
# ============================================================

@ti.kernel
def update():
    center = ti.Vector([0.5, 0.5])
    t = time_value[None]
    mpos = mouse_pos[None]
    mstrength = mouse_strength[None]

    for i in range(N):
        p = pos[i]
        v = vel[i]

        # ----------------------------------------------------
        # 1) Attraction douce vers le centre
        # ----------------------------------------------------
        d_center = center - p
        r2_center = d_center.dot(d_center) + 1e-4
        force_center = 0.0008 * d_center / ti.sqrt(r2_center)

        # ----------------------------------------------------
        # 2) Champ tourbillonnant autour du centre
        # ----------------------------------------------------
        swirl = ti.Vector([-d_center[1], d_center[0]]) * 0.9

        # ----------------------------------------------------
        # 3) Pulsation temporelle légère
        # ----------------------------------------------------
        pulsation = 0.4 + 0.3 * ti.sin(2.0 * t)

        # ----------------------------------------------------
        # 4) Interaction souris optionnelle
        #    clic gauche : attraction
        #    clic droit  : répulsion
        # ----------------------------------------------------
        d_mouse = mpos - p
        r2_mouse = d_mouse.dot(d_mouse) + 1e-4
        force_mouse = mstrength * d_mouse / (r2_mouse * ti.sqrt(r2_mouse))

        # ----------------------------------------------------
        # 5) Accélération totale
        # ----------------------------------------------------
        a = force_center + pulsation * 0.0012 * swirl + 0.00003 * force_mouse

        # Amortissement léger
        v = 0.999 * v + DT * a
        p = p + DT * v

        # ----------------------------------------------------
        # 6) Rebonds sur les bords
        # ----------------------------------------------------
        if p[0] < 0.0:
            p[0] = 0.0
            v[0] = -v[0]
        elif p[0] > 1.0:
            p[0] = 1.0
            v[0] = -v[0]

        if p[1] < 0.0:
            p[1] = 0.0
            v[1] = -v[1]
        elif p[1] > 1.0:
            p[1] = 1.0
            v[1] = -v[1]

        pos[i] = p
        vel[i] = v

# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():
    init_particles()

    window = ti.ui.Window(
        name=f"Taichi - Stress test particules ({N})",
        res=WINDOW_RES,
        fps_limit=FPS_LIMIT
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    t = 0.0
    radius = 0.0018

    while window.running:
        # Gestion des entrées
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == 'r':
                init_particles()

        cursor = window.get_cursor_pos()
        mouse_pos[None] = ti.Vector([cursor[0], cursor[1]])

        strength = 0.0
        if window.is_pressed(ti.ui.LMB):
            strength = 100.0
        elif window.is_pressed(ti.ui.RMB):
            strength = -100.0
        mouse_strength[None] = strength

        # Avance simulation
        for _ in range(SUBSTEPS):
            time_value[None] = t
            update()
            t += DT

        # Rendu
        canvas.set_background_color((0.03, 0.03, 0.05))
        canvas.circles(pos, radius=radius, per_vertex_color=col)

        # GUI d'information
        with gui.sub_window("Parametres", 0.02, 0.02, 0.24, 0.16):
            gui.text(f"Particules : {N}")
            gui.text(f"Backend GPU demande : {USE_GPU}")
            gui.text("Souris gauche : attraction")
            gui.text("Souris droite : repulsion")
            gui.text("Touche R : reinitialisation")

        window.show()

if __name__ == "__main__":
    main()