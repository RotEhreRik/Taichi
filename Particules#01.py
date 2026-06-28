import taichi as ti

# Initialisation Taichi :
# - ti.gpu si vous voulez privilégier le GPU
# - vous pouvez remplacer par ti.cpu pour un premier test très robuste
ti.init(arch=ti.gpu)

# Nombre de particules
N = 2

# Paramètres numériques
dt = 0.001
k_coulomb = 0.02
softening = 1e-4

# Champs Taichi
pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
force = ti.Vector.field(2, dtype=ti.f32, shape=N)
charge = ti.field(dtype=ti.f32, shape=N)
mass = ti.field(dtype=ti.f32, shape=N)

@ti.kernel
def init_particles():
    pos[0] = ti.Vector([0.35, 0.5])
    pos[1] = ti.Vector([0.65, 0.5])

    vel[0] = ti.Vector([0.0, 0.04])
    vel[1] = ti.Vector([0.0, -0.04])

    charge[0] = 1.0
    charge[1] = -1.0

    mass[0] = 1.0
    mass[1] = 1.0

@ti.kernel
def compute_forces():
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])

    # Interaction entre les deux particules
    r = pos[1] - pos[0]
    dist2 = r.dot(r) + softening
    dist = ti.sqrt(dist2)

    # Force coulombienne vectorielle
    # F12 = k * q1*q2 * r / |r|^3
    f = - k_coulomb * charge[0] * charge[1] * r / (dist2 * dist)

    force[0] += f
    force[1] -= f

@ti.kernel
def integrate():
    for i in range(N):
        vel[i] += dt * force[i] / mass[i]
        pos[i] += dt * vel[i]

        # Rebond simple sur les bords de la fenêtre [0,1]x[0,1]
        for d in ti.static(range(2)):
            if pos[i][d] < 0.02:
                pos[i][d] = 0.02
                vel[i][d] = -vel[i][d]
            elif pos[i][d] > 0.98:
                pos[i][d] = 0.98
                vel[i][d] = -vel[i][d]

def main():
    init_particles()

    gui = ti.GUI("Deux particules coulombiennes", res=(800, 800))

    while gui.running:
        for _ in range(10):
            compute_forces()
            integrate()

        positions = pos.to_numpy()

        gui.clear(0x111111)
        gui.circles(positions, radius=12, color=0xFFCC33)
        gui.line(begin=positions[0], end=positions[1], radius=2, color=0x4477FF)
        gui.show()

if __name__ == "__main__":
    main()