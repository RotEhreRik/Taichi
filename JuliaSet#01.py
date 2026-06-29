import taichi as ti

ti.init(arch=ti.gpu)

N = 320
X_SIZE = N * 2
Y_SIZE = N
ITER_MAX = 100

pixels = ti.field(dtype=float, shape =(X_SIZE, Y_SIZE))

@ti.func
def complex_sqr(z):
    return ti.Vector ([z[0]**2 - z[1]**2 , z[1] * z[0] * 2])

@ti.kernel
def paint(t: float):
    for i, j in pixels: # Parallized over all pixels
        c = ti.Vector ([-0.8* ti.sin(t), 0.2 * ti.cos(t)])
        z = ti.Vector ([i / N - 1, j / N - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < ITER_MAX:
            z = complex_sqr(z) + c
            iterations += 1
            pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(N * 2, N))

for i in range (1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()