import taichi as ti

# ============================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================

USE_GPU = True

# Commencez petit, puis montez progressivement :
# 500, 1000, 2000, 5000, 10000...
N = 500

DT = 0.00015
# DAMPING = 0.999
DAMPING = 1e-8
WINDOW_RES = (1000, 1000)
FPS_LIMIT = 120

# Force répulsive "type charge identique"
# G_ATTRACTION = 0.00002
K_ATTRACTION = 0.00004
SOFTENING = 1e-5

# Rayon d'affichage
PARTICLE_RADIUS = 0.001
AGREG_RADIUS2 = 0.0040**2

# ============================================================
# INITIALISATION TAICHI
# ============================================================

ti.init(arch=ti.gpu if USE_GPU else ti.cpu)

# ============================================================
# CHAMPS TAICHI
# ============================================================

positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)
forces = ti.Vector.field(2, dtype=ti.f32, shape=N)
densities = ti.Vector.field(2, dtype=ti.f32, shape=N)
masses = ti.field(dtype=ti.f32, shape=N)
colors = ti.Vector.field(3, dtype=ti.f32, shape=N)
radii = ti.field(dtype=ti.f32, shape=N)


# ============================================================
# INITIALISATION
# ============================================================

@ti.kernel
def init_particles():
    positions[0] = ti.Vector([0.5, 0.5])
    velocities[0] = ti.Vector([0.0, 0.0])
    forces[0] = ti.Vector([0.0, 0.0])
    # masses[0] = 1.0e6
    masses[0] = 1000.0
    colors[0] = ti.Vector([1.0, 1.0, 0.0])
    radii[0] = PARTICLE_RADIUS * 5

    for i in range(1, N):
        x = 0.15 + 0.7 * ti.random(dtype=ti.f32)
        y = 0.15 + 0.7 * ti.random(dtype=ti.f32)

        positions[i] = ti.Vector([x, y])

        vx = (ti.random(dtype=ti.f32) - 0.5) * 0.5
        vy = (ti.random(dtype=ti.f32) - 0.5) * 0.5
        velocities[i] = ti.Vector([vx, vy])

        forces[i] = ti.Vector([0.0, 0.0])

        masses[i] = 1.0

        colors[i] = ti.Vector([
            0.4 + 0.6 * ti.random(dtype=ti.f32),
            0.5 + 0.5 * ti.random(dtype=ti.f32),
            0.7 + 0.3 * ti.random(dtype=ti.f32)
        ])

        radii[i] = PARTICLE_RADIUS


# ============================================================
# CALCUL DES FORCES : VERSION NAÏVE TOUT-À-TOUT
# ============================================================

@ti.kernel
def clear_forces():
    for i in range(N):
        forces[i] = ti.Vector([0.0, 0.0])


@ti.kernel
def compute_forces_naive():
    for i in range(N):
        pi = positions[i]
        mi = masses[i]

        density = 1.0;

        for j in range(i):
            pj = positions[j]
            mj = masses[j]

            # Vecteur de j vers i
            r = pi - pj

            # Distance avec régularisation pour éviter les divisions explosives
            dist2 = r.dot(r) + SOFTENING

            # density += 1 / dist2
            density += mi*mj / dist2

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
            f = - K_ATTRACTION * mi * mj * r / (dist2 * dist)
            # f = - G_ATTRACTION / density * mi * mj * r / (dist2 * dist)

            # f /= (1e-5 * density)

            # Action-réaction
            forces[i] += f
            forces[j] -= f
            densities[j] = density

@ti.kernel
def agregate():
    for i in range(1,N):
        v = velocities[i]
        p = positions[i]
        d = densities[i]
        m = masses[i]

        # v = (1 - DAMPING * d) * v + DT * PolarAccs[i] / m
        # p = p + DT * 0.5*v
        v = (1 - DAMPING) * v + DT * forces[i] / m
        p = p + DT * v
        # v = (1 - DAMPING) * v + DT * PolarAccs[i] / m
        velocities[i] = v
        positions[i] = p

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

            if dist2<AGREG_RADIUS2:

                velocities[i] = (mi*vi+mj*vj)/(mi+mj)
                velocities[j] = velocities[i]

# ============================================================
# INTÉGRATION
# ============================================================

@ti.kernel
def integrate():


    SLOWER_PRI = 1.0
    SLOWER_SEC = -1.0
    for i in range(1, N):
        v = velocities[i]
        p = positions[i]
        # Rebonds sur les bords
        if p[0] < 0.0:
            # p[0] = 0.0
            p[0] = -p[0]
            v[0] = -v[0]/SLOWER_PRI
            v[1] = -v[1]/SLOWER_SEC
        elif p[0] > 1.0:
            p[0] = 1.0
            # p[0] = 1.0 - p[0]
            v[0] = -v[0]/SLOWER_PRI
            v[1] = -v[1]/SLOWER_SEC

        if p[1] < 0.0:
            p[1] = 0.0
            # p[1] = -p[1]
            v[0] = -v[0]/SLOWER_SEC
            v[1] = -v[1]/SLOWER_PRI
        elif p[1] > 1.0:
            p[1] = 1.0
            # p[1] = 1.0 - p[1]
            v[0] = -v[0]/SLOWER_SEC
            v[1] = -v[1]/SLOWER_PRI

        velocities[i] = v
        positions[i] = p


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
        agregate()
        integrate()

        canvas.set_background_color((0.02, 0.02, 0.03))
        canvas.circles(positions, radius=PARTICLE_RADIUS, per_vertex_color=colors)

        with gui.sub_window("Infos", 0.02, 0.02, 0.28, 0.16):
            gui.text(f"N = {N}")
            gui.text("Mode : interactions completes naives")
            gui.text("Complexite : O(N^2)")
            gui.text("Touche R : reinitialiser")

        window.show()


if __name__ == "__main__":
    main()
