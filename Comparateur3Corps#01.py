import taichi as ti
import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================
# Comparaison de trois intégrateurs pour un problème à 3 corps
# - Euler semi-implicite (souvent utilisé en temps réel)
# - Velocity Verlet
# - Leapfrog KDK
# Suivi des énergies : cinétique, potentielle, mécanique
# ============================================================


USE_GPU = True
ti.init(arch=ti.gpu if USE_GPU else ti.cpu)
# ti.init(arch=ti.cpu, default_fp=ti.f64)

N = 3
METHODS = 3
G = 1.0
SOFTENING = 1e-3
DT = 1e-3
STEPS = 20000
SAMPLES = 2001

masses_np = np.array([1000.0, 10.0, 1.0], dtype=np.float64)
init_pos_np = np.array([
    [0.0, 0.0],
    [8.0, 0.0],
    [14.0, 0.0],
], dtype=np.float64)

v2 = math.sqrt(G * masses_np[0] / np.linalg.norm(init_pos_np[1] - init_pos_np[0]))
v3 = math.sqrt(G * masses_np[0] / np.linalg.norm(init_pos_np[2] - init_pos_np[0]))
init_vel_np = np.array([
    [0.0, 0.0],
    [0.0, 0.96 * v2],
    [0.0, 0.90 * v3],
], dtype=np.float64)

momentum = np.sum(masses_np[:, None] * init_vel_np, axis=0)
init_vel_np[0] -= momentum / masses_np[0]

masses = ti.field(dtype=ti.f64, shape=N)
pos = ti.Vector.field(2, dtype=ti.f64, shape=(METHODS, N))
vel = ti.Vector.field(2, dtype=ti.f64, shape=(METHODS, N))
acc = ti.Vector.field(2, dtype=ti.f64, shape=(METHODS, N))
vel_half = ti.Vector.field(2, dtype=ti.f64, shape=N)

for i in range(N):
    masses[i] = masses_np[i]

@ti.func
def grav_acc(p_i, p_j, m_j):
    r = p_j - p_i
    d2 = r.dot(r) + SOFTENING * SOFTENING
    inv_d = 1.0 / ti.sqrt(d2)
    inv_d3 = inv_d * inv_d * inv_d
    return G * m_j * r * inv_d3

@ti.kernel
def initialize():
    for method, i in ti.ndrange(METHODS, N):
        pos[method, i] = ti.Vector([init_pos_np[i, 0], init_pos_np[i, 1]])
        vel[method, i] = ti.Vector([init_vel_np[i, 0], init_vel_np[i, 1]])
        acc[method, i] = ti.Vector([0.0, 0.0])
    for i in range(N):
        vel_half[i] = ti.Vector([0.0, 0.0])

@ti.kernel
def compute_acc_for_method(method: ti.i32):
    for i in range(N):
        a = ti.Vector([0.0, 0.0])
        for j in range(N):
            if i != j:
                a += grav_acc(pos[method, i], pos[method, j], masses[j])
        acc[method, i] = a

@ti.kernel
def prepare_leapfrog_half(dt: ti.f64):
    for i in range(N):
        vel_half[i] = vel[2, i] + 0.5 * dt * acc[2, i]

@ti.kernel
def step_euler(dt: ti.f64):
    for i in range(N):
        vel[0, i] = vel[0, i] + dt * acc[0, i]
        pos[0, i] = pos[0, i] + dt * vel[0, i]

@ti.kernel
def verlet_drift(dt: ti.f64):
    for i in range(N):
        pos[1, i] = pos[1, i] + dt * vel[1, i] + 0.5 * dt * dt * acc[1, i]

@ti.kernel
def verlet_kick(old_ax: ti.types.ndarray(), old_ay: ti.types.ndarray(), dt: ti.f64):
    for i in range(N):
        old_a = ti.Vector([old_ax[i], old_ay[i]])
        vel[1, i] = vel[1, i] + 0.5 * dt * (old_a + acc[1, i])

@ti.kernel
def leapfrog_drift(dt: ti.f64):
    for i in range(N):
        pos[2, i] = pos[2, i] + dt * vel_half[i]

@ti.kernel
def leapfrog_kick(dt: ti.f64):
    for i in range(N):
        vel_half[i] = vel_half[i] + dt * acc[2, i]
        vel[2, i] = vel_half[i] - 0.5 * dt * acc[2, i]

@ti.kernel
def export_pos(method: ti.i32, out: ti.types.ndarray()):
    for i, k in ti.ndrange(N, 2):
        out[i, k] = pos[method, i][k]

@ti.kernel
def export_acc_verlet(out: ti.types.ndarray()):
    for i, k in ti.ndrange(N, 2):
        out[i, k] = acc[1, i][k]

@ti.kernel
def energy_of_method(method: ti.i32, out: ti.types.ndarray()):
    ksum = 0.0
    psum = 0.0
    for i in range(N):
        ksum += 0.5 * masses[i] * vel[method, i].dot(vel[method, i])
    for i in range(N):
        for j in range(i + 1, N):
            r = pos[method, j] - pos[method, i]
            d = ti.sqrt(r.dot(r) + SOFTENING * SOFTENING)
            psum += -G * masses[i] * masses[j] / d
    out[0] = ksum
    out[1] = psum
    out[2] = ksum + psum

initialize()
for method in range(METHODS):
    compute_acc_for_method(method)
prepare_leapfrog_half(DT)

sample_times = np.linspace(0.0, DT * STEPS, SAMPLES)
kin = np.zeros((METHODS, SAMPLES), dtype=np.float64)
pot = np.zeros((METHODS, SAMPLES), dtype=np.float64)
mech = np.zeros((METHODS, SAMPLES), dtype=np.float64)
traj = np.zeros((METHODS, SAMPLES, N, 2), dtype=np.float64)

pos_buf = np.zeros((N, 2), dtype=np.float64)
acc_buf = np.zeros((N, 2), dtype=np.float64)
energy_buf = np.zeros(3, dtype=np.float64)

sample_every = max(1, STEPS // (SAMPLES - 1))

method_names = ["Euler", "Verlet", "Leapfrog"]
colors = ["#d62728", "#1f77b4", "#2ca02c"]


def record(sample_id):
    for m in range(METHODS):
        export_pos(m, pos_buf)
        traj[m, sample_id] = pos_buf
        energy_of_method(m, energy_buf)
        kin[m, sample_id] = energy_buf[0]
        pot[m, sample_id] = energy_buf[1]
        mech[m, sample_id] = energy_buf[2]

record(0)
sample_id = 1

for step in range(1, STEPS + 1):
    step_euler(DT)
    compute_acc_for_method(0)

    export_acc_verlet(acc_buf)
    verlet_drift(DT)
    compute_acc_for_method(1)
    verlet_kick(acc_buf[:, 0], acc_buf[:, 1], DT)

    leapfrog_drift(DT)
    compute_acc_for_method(2)
    leapfrog_kick(DT)

    if step % sample_every == 0 and sample_id < SAMPLES:
        record(sample_id)
        sample_id += 1

while sample_id < SAMPLES:
    record(sample_id)
    sample_id += 1

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
ax_traj = axes[0, 0]
ax_k = axes[0, 1]
ax_u = axes[1, 0]
ax_e = axes[1, 1]

for m, (name, color) in enumerate(zip(method_names, colors)):
    for b in range(N):
        ax_traj.plot(traj[m, :, b, 0], traj[m, :, b, 1], color=color, alpha=0.25 + 0.2 * b, lw=1.0)
    ax_k.plot(sample_times, kin[m], label=name, color=color)
    ax_u.plot(sample_times, pot[m], label=name, color=color)
    ax_e.plot(sample_times, mech[m], label=name, color=color)

ax_traj.set_title("Trajectoires des 3 corps")
ax_traj.set_xlabel("x")
ax_traj.set_ylabel("y")
ax_traj.axis("equal")
ax_traj.grid(True, alpha=0.3)

ax_k.set_title("Énergie cinétique totale")
ax_k.set_xlabel("Temps")
ax_k.set_ylabel("K")
ax_k.grid(True, alpha=0.3)
ax_k.legend()

ax_u.set_title("Énergie potentielle totale")
ax_u.set_xlabel("Temps")
ax_u.set_ylabel("U")
ax_u.grid(True, alpha=0.3)
ax_u.legend()

ax_e.set_title("Énergie mécanique totale")
ax_e.set_xlabel("Temps")
ax_e.set_ylabel("E = K + U")
ax_e.grid(True, alpha=0.3)
ax_e.legend()

fig.suptitle("Comparaison Taichi : Euler vs Verlet vs Leapfrog sur un système gravitationnel à 3 corps")
fig.savefig("output/three_body_taichi_compare.png", dpi=180)
plt.close(fig)

with open("output/three_body_taichi_compare.csv", "w", encoding="utf-8") as f:
    f.write("time,method,kinetic,potential,mechanical\n")
    for s, t in enumerate(sample_times):
        for m, name in enumerate(method_names):
            f.write(f"{t},{name},{kin[m, s]},{pot[m, s]},{mech[m, s]}\n")

print("OK")