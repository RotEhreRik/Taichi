import taichi as ti

# ============================================================
# Configuration générale
# ============================================================

ti.init(arch=ti.gpu)

N = 512 # taille logique de la grille
BLOCK_SIZE = 16 # taille d'un bloc sparse
POINTER_SIZE = N // BLOCK_SIZE

# >>> LIGNE AJOUTÉE : facteur de zoom entier pour un affichage net
SCALE = 1

# >>> LIGNE AJOUTÉE : taille réelle de la fenêtre
WINDOW_W = N * SCALE
# >>> LIGNE AJOUTÉE : taille réelle de la fenêtre
WINDOW_H = N * SCALE

WRAPPED = True


# Deux buffers : état courant et état suivant
current = ti.field(dtype=ti.i32)
next_grid = ti.field(dtype=ti.i32)

# Image d'affichage
pixels = ti.field(dtype=ti.f32, shape=(N, N))

display_pixels = ti.field(dtype=ti.f32, shape=(WINDOW_W, WINDOW_H))

# ============================================================
# Définition de la structure sparse
# pointer -> bitmasked -> cellules
# ============================================================

root = ti.root.pointer(ti.ij, (POINTER_SIZE, POINTER_SIZE))
block = root.bitmasked(ti.ij, (BLOCK_SIZE, BLOCK_SIZE))
block.place(current, next_grid)

# ============================================================
# Outils
# ============================================================

@ti.func
def inside(i, j):
    return 0 <= i < N and 0 <= j < N


@ti.func
def count_neighbors(i, j):
    s = 0
    for di, dj in ti.ndrange((-1, 2), (-1, 2)):
        if di != 0 or dj != 0:
            ni = i + di
            nj = j + dj
            if WRAPPED:
                ni = ni % N
                nj = nj % N
                s += current[ni, nj]
            else:
                if inside(ni, nj):
                    s += current[ni, nj]
    return s

# ============================================================
# Initialisation
# ============================================================

@ti.kernel
def clear_all():
    for i, j in current:
        current[i, j] = 0
    for i, j in next_grid:
        next_grid[i, j] = 0


@ti.kernel
def seed_glider(x: ti.i32, y: ti.i32):
    current[x + 1, y + 0] = 1
    current[x + 2, y + 1] = 1
    current[x + 0, y + 2] = 1
    current[x + 1, y + 2] = 1
    current[x + 2, y + 2] = 1


@ti.kernel
def seed_r_pentomino(x: ti.i32, y: ti.i32):
    current[x + 1, y + 0] = 1
    current[x + 2, y + 0] = 1
    current[x + 0, y + 1] = 1
    current[x + 1, y + 1] = 1
    current[x + 1, y + 2] = 1

def seed_random(n: ti.i32):
    for i in range(n):
        x = N*(0.1+0.8*ti.random(dtype=ti.f32))
        x = ti.cast(x, ti.i32)
        y = N*(0.1+0.8*ti.random(dtype=ti.f32))
        y = ti.cast(y, ti.i32)
        seed_r_pentomino(x, y)



# ============================================================
# Préparation des zones candidates
# ============================================================

@ti.kernel
def prepare_candidates():
    for i, j in current:
        for di, dj in ti.ndrange((-1, 2), (-1, 2)):
            ni = i + di
            nj = j + dj
            if inside(ni, nj):
                next_grid[ni, nj] = 0

# ============================================================
# Une génération du jeu de la vie
# ============================================================

@ti.kernel
def step():
    for i, j in next_grid:
        n = count_neighbors(i, j)
        alive = current[i, j]

        if alive == 1:
            if n == 2 or n == 3:
                next_grid[i, j] = 1
            else:
                next_grid[i, j] = 0
        else:
            if n == 3:
                next_grid[i, j] = 1
            else:
                next_grid[i, j] = 0


@ti.kernel
def copy_back():
    for i, j in next_grid:
        current[i, j] = next_grid[i, j]

# ============================================================
# Affichage
# ============================================================

@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = 0.0

    for i, j in current:
        if current[i, j] != 0:
            pixels[i, j] = 1.0

@ti.kernel
def upscale_pixels():
    for x, y in display_pixels:
        i = x // SCALE
        j = y // SCALE
        display_pixels[x, y] = pixels[i, j]





# ============================================================
# Programme principal
# ============================================================

def main():
    window = ti.ui.Window("Jeu de la vie sparse GPU - Taichi", (WINDOW_W, WINDOW_H))
    canvas = window.get_canvas()

    clear_all()

    # seed_glider(20, 20)
    # seed_r_pentomino(50,20)
    # seed_r_pentomino(200,220)
    # seed_r_pentomino(350,20)

    seed_random(10)

    while window.running:
        prepare_candidates()
        step()
        copy_back()
        render()

        upscale_pixels()

        canvas.set_image(display_pixels)
        window.show()


if __name__ == "__main__":
    main()