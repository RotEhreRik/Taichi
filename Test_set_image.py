import taichi as ti

ti.init(arch=ti.gpu)  # utilisation du GPU si disponible

# ------------------------------
# Paramètres de la fenêtre
# ------------------------------
W, H = 800, 600

window = ti.ui.Window("Mémoire vidéo persistante", res=(W, H))
canvas = window.get_canvas()

# ------------------------------
# Framebuffer persistant
# ------------------------------
# fb[x, y] = couleur RGB en float32 dans [0, 1]
fb = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

# ------------------------------
# État de la particule
# ------------------------------
pos = ti.Vector.field(2, dtype=ti.f32, shape=())
vel = ti.Vector.field(2, dtype=ti.f32, shape=())

@ti.kernel
def init_state():
    pos[None] = ti.Vector([0.5, 0.5])    # position en coordonnées normalisées [0,1]
    vel[None] = ti.Vector([0.003, 0.002])

@ti.kernel
def clear_fb():
    # Si vous voulez effacer tout l'écran (optionnel, utilisé au début seulement)
    for x, y in fb:
        fb[x, y] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def fade_fb():
    # Si vous voulez effacer tout l'écran (optionnel, utilisé au début seulement)
    for x, y in fb:
        fb[x, y] = fb[x, y]/1.01

@ti.kernel
def update_and_draw():
    # Met à jour la position de la particule
    p = pos[None]
    v = vel[None]

    p += v

    # rebonds sur les bords
    if p.x < 0.0 or p.x > 1.0:
        v.x = -v.x
    if p.y < 0.0 or p.y > 1.0:
        v.y = -v.y

    pos[None] = p
    vel[None] = v

    # convertit la position normalisée [0,1] en coordonnées pixel
    x = int(p.x * W)
    y = int(p.y * H)

    # écrit un point lumineux dans le framebuffer
    if 0 <= x < W and 0 <= y < H:
        fb[x, y] = ti.Vector([1.0, 1.0, 0.0])  # jaune

# ------------------------------
# Programme principal
# ------------------------------
init_state()
clear_fb()  # on efface une fois au début

while window.running:
    # mise à jour de la simulation et dessin dans fb
    fade_fb()
    update_and_draw()

    # on envoie fb vers la fenêtre à chaque frame
    canvas.set_image(fb)
    window.show()