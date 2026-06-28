import taichi as ti

# from sympy.physics.units import acceleration
# from sympy.plotting.pygletplot.plot_modes import Polar

# ============================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================

USE_GPU = True

# Commencez petit, puis montez progressivement :
# 500, 1000, 2000, 5000, 10000...
N = 20

DT = 0.0015
DAMPING = 0
WINDOW_RES = (1000, 1000)
FPS_LIMIT = 120
DT = 1/FPS_LIMIT

# Force répulsive "type charge identique"
# G_ATTRACTION = 0.00002
G_ATTRACTION = 9.81
SOFTENING = 1e-5

# Rayon d'affichage
PARTICLE_RADIUS = 0.003

MAX_T = 20

MIN_LENGTH = 0.3
MAX_LENGTH = 0.8

INIT_ANGLE_DEG = 30.
PI = 3.1415926

# ============================================================
# INITIALISATION TAICHI
# ============================================================

ti.init(arch=ti.gpu if USE_GPU else ti.cpu)

# ============================================================
# CHAMPS TAICHI
# ============================================================

CartPoss = ti.Vector.field(2, dtype=ti.f32, shape=N)
PolarPoss = ti.Vector.field(2, dtype=ti.f32, shape=N)
PolarVels = ti.Vector.field(2, dtype=ti.f32, shape=N)
PolarAccs = ti.Vector.field(2, dtype=ti.f32, shape=N)
Colors = ti.Vector.field(3, dtype=ti.f32, shape=N)


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


@ti.func
def get_const(T_max, l_max):
    # k = ti.sqrt((G_ATTRACTION / l_max) * (T_max / (2 * PI)) ** 2) - 2.0
    k = ti.sqrt(G_ATTRACTION / l_max) * (T_max / (2 * PI)) - 2.0
    return ti.cast(k, ti.f32)


@ti.func
def get_length(i, T_max, k):
    # l = i / N
    l = G_ATTRACTION * (T_max / (2 * PI * (k + i + 1))) ** 2
    return ti.cast(l, ti.f32)


# ============================================================
# INITIALISATION
# ============================================================

@ti.kernel
def init_particles():
    k = get_const(MAX_T, MAX_LENGTH)
    for i in range(N):
        pr = get_length(i, MAX_T, k)
        pa = INIT_ANGLE_DEG * PI / 180
        PolarPoss[i] = ti.Vector([pr, pa])
        CartPoss[i] = ti.Vector(Polar2Cart(pr, pa))

        vr = 0.0
        va = 0.0
        PolarVels[i] = ti.Vector([0.0, 0.0])

        PolarAccs[i] = ti.Vector([0.0, 0.0])

        hue = pr
        Colors[i] = hsv_to_rgb(hue, 0.85, 1.00)

        # Colors[i] = ti.Vector([
        #     0.4 + 0.6 * ti.random(dtype=ti.f32),
        #     0.5 + 0.5 * ti.random(dtype=ti.f32),
        #     0.7 + 0.3 * ti.random(dtype=ti.f32)
        # ])


@ti.func
def Polar2Cart(pr, pa):
    x = 0.50 - 0.80 * pr * ti.sin(pa)
    y = 0.90 - 0.80 * pr * ti.cos(pa)
    return [x, y]


# ============================================================
# CALCUL DES FORCES : VERSION NAÏVE TOUT-À-TOUT
# ============================================================

@ti.kernel
def clear_accelerations():
    for i in range(N):
        PolarAccs[i] = ti.Vector([0.0, 0.0])


@ti.kernel
def compute_accelerations():
    for i in range(N):
        p = PolarPoss[i]
        r = p[0]
        a = p[1]
        ar = 0.0
        aa = - G_ATTRACTION * ti.sin(a) / r
        PolarAccs[i] = ti.Vector([ar, aa])


# ============================================================
# INTÉGRATION
# ============================================================

@ti.kernel
def apply_accelerations():
    for i in range(N):
        ppos = PolarPoss[i]
        pr = ppos[0]
        pa = ppos[1]

        pvel = PolarVels[i]
        vr = 0.0
        va = pvel[1]

        pacc = PolarAccs[i]
        ar = 0.0
        aa = pacc[1]

        va += DT * aa
        pa += DT * va

        PolarVels[i] = ti.Vector([vr, va])
        PolarPoss[i] = ti.Vector([pr, pa])
        CartPoss[i] = ti.Vector(Polar2Cart(pr, pa))


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

        clear_accelerations()
        compute_accelerations()
        apply_accelerations()

        canvas.set_background_color((0.02, 0.02, 0.03))
        canvas.circles(CartPoss, radius=PARTICLE_RADIUS, per_vertex_color=Colors)

        with gui.sub_window("Infos", 0.02, 0.02, 0.28, 0.16):
            gui.text(f"N = {N}")
            gui.text("Mode : pendules de différentes longueurs")
            gui.text("Touche R : reinitialiser")

        window.show()


if __name__ == "__main__":
    main()
