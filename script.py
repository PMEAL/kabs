import numpy as np
import taichi as ti
import porespy as ps

from kabs import compute_permeability, solve_flow

ti.init(arch=ti.cpu)

# im = np.loadtxt("img_ftb131.txt")
# im = np.reshape(im, (131, 131, 131), order="F")
# im[im > 0] = 1
im = ps.generators.cylinders([200, 200, 200], r=10)

PREFIX  = "LB_SinglePhase"
N_STEPS = 15000

solve_flow(im, direction="x", n_steps=N_STEPS, output_prefix=PREFIX)

compute_permeability(f"{PREFIX}_{N_STEPS}.vtr", direction="x")
