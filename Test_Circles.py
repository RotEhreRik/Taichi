import taichi as ti
import numpy as np

USE_GPU = True

# ti.init()
ti.init(arch=ti.gpu if USE_GPU else ti.cpu)

num_circles = 50
pos = np.random.random((num_circles, 2))  # Random CartPoss
# radii = np.ones(num_circles) * 5          # Uniform radius
# radii = ti.random(dtype=ti.f32) *5
radii = np.random.random((num_circles, 1))  # Random CartPoss

# Palette indices for random color assignment
indices = np.random.randint(0, 3, size=(num_circles,))
palette = [0x068587, 0xED553B, 0xEEEEF0]

gui = ti.GUI('Circles', res=(400, 400))
while gui.running:
    gui.circles(pos, radius=radii, per_vertex_radius=radii, palette=palette, palette_indices=indices)
    gui.show()