import taichi as ti

ti.init(arch=ti.gpu)

N = 320
X_SIZE = N * 2
Y_SIZE = N
ITER_MAX = 100

# ============================================================
# Coloration
# ============================================================


pixels = ti.field(dtype=float, shape =(X_SIZE, Y_SIZE))


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


# ============================================================
# Calculs
# ============================================================

@ti.func
def complex_sqr(z):
    return ti.Vector ([z[0]**2 - z[1]**2 , z[1] * z[0] * 2])

@ti.kernel
# def paint(t: float):
def paint(cx: float, cy: float, t: float):
    for i, j in pixels: # Parallized over all pixels
        # c = ti.Vector ([-0.8* ti.sin(t), 0.2 * ti.cos(t)])
        # c = ti.Vector([cx, cy])
        c = ti.Vector ([cx -0.01* ti.sin(t), cy +0.01 * ti.cos(t)])
        z = ti.Vector ([i / N - 1, j / N - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < ITER_MAX:
            z = complex_sqr(z) + c
            iterations += 1
            pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(N * 2, N))



# ============================================================
# Principal
# ============================================================



for i in range (1000000):
    # paint(i * 0.03)
    mx, my = gui.get_cursor_pos()
    cx = (mx - 0.5) * 2.0
    cy = (my - 0.5) * 2.0
    paint(cx, cy, i*0.1)
    gui.set_image(pixels)
    gui.show()