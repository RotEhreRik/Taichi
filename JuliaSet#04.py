import taichi as ti

ti.init(arch=ti.gpu)

N = 1024
FACTOR = 1
X_SIZE = N * FACTOR
Y_SIZE = N
ITER_MAX = 255

# ============================================================
# Coloration
# ============================================================


# pixels = ti.field(dtype=float, shape =(X_SIZE, Y_SIZE))
pixels = ti.Vector.field(3, dtype=float, shape=(X_SIZE, Y_SIZE))


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
def complex_absimag(z):
    return ti.Vector([z[0] , ti.abs(z[1])])


@ti.func
def complex_mul(z1, z2):
    return ti.Vector([z1[0] *z2[0] - z1[1] *z2[1], z1[0] * z2[1] + z1[1] * z2[0]])

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


@ti.func
def complex_cub(z):
    return ti.Vector([z[0] ** 3 - 3 * z[0] * z[1] ** 2, 3 * z[0] ** 2 * z[1] - z[1] ** 3])


@ti.kernel
# def paint(t: float):
def paint(cx: float, cy: float, t: float):
    for i, j in pixels:  # Parallized over all pixels
        if i >= 10 and i < X_SIZE - 10 and j >= 10 and j < 20:
            # h = 0.02 * iterations + 0.1 * t
            h = (i - 10) / (N - 10)
            h = 2 * (i - 10) / (N - 10)
            s = 1.0
            v = 1.0
            pixels[i, j] = hsv_to_rgb(h, s, v)
        else:
            # c = ti.Vector ([-0.8* ti.sin(t), 0.2 * ti.cos(t)])
            # c = ti.Vector([cx, cy])
            c = ti.Vector([cx - 0.01 * ti.sin(t), cy + 0.01 * ti.cos(t)])
            z = ti.Vector([i / X_SIZE - 0.5 * FACTOR, j / Y_SIZE - 0.5]) * 2
            iterations = 0
            while z.norm() < 20 and iterations < ITER_MAX:
                # z = complex_sqr(z) + c
                # z = complex_sqr(complex_absimag(z)) + c
                # z = complex_mul(z, complex_absimag(z)) + c
                # z = complex_mul(z,z) + c
                # z = complex_cub(z) + c
                # z = complex_mul(z, complex_mul(z, z)) + c
                # z = complex_sqr(complex_cub(z)) + c
                z = complex_mul(complex_cub(z),complex_sqr(z)) + c
                iterations += 1
                # pixels[i, j] = 1 - iterations * 0.02
                if iterations == ITER_MAX:
                    pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])
                else:
                    # h = 0.02 * iterations + 0.1 * t
                    # h = 2 * iterations / ITER_MAX + 0.1 * t
                    h = iterations / ITER_MAX + 0.1 * t
                    s = 1.0
                    v = 1.0
                    pixels[i, j] = hsv_to_rgb(h, s, v)


gui = ti.GUI("Julia Set", res=(X_SIZE, Y_SIZE))

# ============================================================
# Principal
# ============================================================


for i in range(1000000):
    # paint(i * 0.03)
    mx, my = gui.get_cursor_pos()
    cx = (mx - 0.5) * 2.0
    cy = (my - 0.5) * 2.0
    paint(cx, cy, i * 0.01)
    gui.set_image(pixels)
    gui.show()
