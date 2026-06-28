import taichi as ti

# ============================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================

USE_GPU = True

# Commencez petit, puis montez progressivement :
# 500, 1000, 2000, 5000, 10000...
N = 1

DT = 0.0015
DAMPING = 0.99999999999
WINDOW_RES = (1000, 1000)
FPS_LIMIT = 120

# Force répulsive "type charge identique"
# G_ATTRACTION = 0.00002
K_REPULSION = 0.02
SOFTENING = 1e-5

# Rayon d'affichage
PARTICLE_RADIUS = 0.003

# ============================================================
# INITIALISATION TAICHI
# ============================================================

ti.init(arch=ti.gpu if USE_GPU else ti.cpu)

# ============================================================
# CHAMPS TAICHI
# ============================================================

pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
force = ti.Vector.field(2, dtype=ti.f32, shape=N)
col = ti.Vector.field(3, dtype=ti.f32, shape=N)


# ============================================================
# INITIALISATION
# ============================================================

@ti.kernel
def init_particles():
    for i in range(N):
        x = 0.15 + 0.7 * ti.random(dtype=ti.f32)
        y = 0.15 + 0.7 * ti.random(dtype=ti.f32)

        pos[i] = ti.Vector([x, y])

        vx = (ti.random(dtype=ti.f32) - 0.5) * 0.05 * 300
        vy = (ti.random(dtype=ti.f32) - 0.5) * 0.05 * 300
        vel[i] = ti.Vector([vx, vy])

        force[i] = ti.Vector([0.0, 0.0])

        col[i] = ti.Vector([
            0.4 + 0.6 * ti.random(dtype=ti.f32),
            0.5 + 0.5 * ti.random(dtype=ti.f32),
            0.7 + 0.3 * ti.random(dtype=ti.f32)
        ])


# ============================================================
# CALCUL DES FORCES : VERSION NAÏVE TOUT-À-TOUT
# ============================================================

@ti.kernel
def clear_forces():
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])


@ti.kernel
def compute_forces_naive():
    for i in range(N):
        pi = pos[i]

        for j in range(i):
            pj = pos[j]

            # Vecteur de j vers i
            r = pi - pj

            # Distance avec régularisation pour éviter les divisions explosives
            dist2 = r.dot(r) + SOFTENING
            dist = ti.sqrt(dist2)

            # Répulsion de type 1 / r^2
            # Forme vectorielle : k * r / |r|^3
            f = K_REPULSION * r / (dist2 * dist)

            # Action-réaction
            force[i] += f
            force[j] -= f


# ============================================================
# INTÉGRATION
# ============================================================

@ti.kernel
def integrate():
    for i in range(N):
        v = vel[i]
        p = pos[i]

        v = DAMPING * v + DT * force[i]
        p = p + DT * v

        # Rebonds sur les bords
        if p[0] < 0.0:
            p[0] = 0.0
            # p[0] = -p[0]
            v[0] = -v[0]
        elif p[0] > 1.0:
            # p[0] = 1.0
            p[0] = 1.0 - p[0]
            v[0] = -v[0]

        if p[1] < 0.0:
            # p[1] = 0.0
            p[1] = -p[1]
            v[1] = -v[1]
        elif p[1] > 1.0:
            # p[1] = 1.0
            p[1] = 1.0 - p[1]
            v[1] = -v[1]

        vel[i] = v
        pos[i] = p


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():
    init_particles()

    window = ti.ui.Window(
        name=f"Taichi naive all-to-all ({N} particules)",
        res=WINDOW_RES,
        fps_limit=FPS_LIMIT
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == 'r':
                init_particles()

        clear_forces()
        compute_forces_naive()
        integrate()

        canvas.set_background_color((0.02, 0.02, 0.03))
        canvas.circles(pos, radius=PARTICLE_RADIUS, per_vertex_color=col)

        with gui.sub_window("Infos", 0.02, 0.02, 0.28, 0.16):
            gui.text(f"N = {N}")
            gui.text("Mode : interactions completes naives")
            gui.text("Complexite : O(N^2)")
            gui.text("Touche R : reinitialiser")

        window.show()


if __name__ == "__main__":
    main()
