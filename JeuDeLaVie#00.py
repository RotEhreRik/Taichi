import taichi as ti

# ============================================================
# Configuration générale
# ============================================================

ti.init(arch=ti.gpu)

N = 128                 # taille logique de la grille
BLOCK_SIZE = 16          # taille d'un bloc sparse
POINTER_SIZE = N // BLOCK_SIZE

# Deux buffers : état courant et état suivant
current = ti.field(dtype=ti.i32)
next_grid = ti.field(dtype=ti.i32)

# Image d'affichage
pixels = ti.field(dtype=ti.f32, shape=(N, N))

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


# ============================================================
# Préparation des zones candidates
# On active dans next_grid les cellules voisines des cellules actives
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
                # next_grid[i, j] = 1
                next_grid[i, j] = 0
            else:
                next_grid[i, j] = 0
        else:
            if n == 3:
                # next_grid[i, j] = 1
                next_grid[i, j] = 0
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
        pixels[i, j] = 1.0


# ============================================================
# Programme principal
# ============================================================

def main():
    gui = ti.GUI("Jeu de la vie sparse GPU - Taichi", res=(N, N))

    clear_all()

    # Quelques motifs de départ
    seed_glider(20, 20)
    # seed_glider(40, 30)
    # seed_r_pentomino(200, 200)
    # seed_r_pentomino(260, 220)

    while gui.running:
        prepare_candidates()
        step()
        copy_back()
        render()

        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()