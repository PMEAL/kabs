import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK


__all__ = [
    "SinglePhaseSolver",
]


bc_defs = {
    'periodic': 0,
    'pressure': 1,
    'velocity': 2,
}


@ti.data_oriented
class SinglePhaseSolver:
    def __init__(self, im, sparse_storage=False):
        self.enable_projection = True
        self.sparse_storage = sparse_storage

        nx, ny, nz = im.shape
        self.nx, self.ny, self.nz = nx, ny, nz
        self.solid = ti.field(ti.i8, shape=(nx, ny, nz))
        self.solid.from_numpy(im)

        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
        self.niu = 0.16667
        self.max_v = ti.field(ti.f32, shape=())

        # Boundary condition mode:
        # 0 = periodic
        # 1 = fixed pressure
        # 2 = fixed velocity
        # boundary pressure value (rho)
        # boundary velocity value for vx, vy, vz

        # Boundary x-axis left side
        self.bc_x_left = 0
        self.rho_bcxl = 1.0
        self.vx_bcxl = 0.0
        self.vy_bcxl = 0.0
        self.vz_bcxl = 0.0

        # Boundary x-axis right side
        self.bc_x_right = 0
        self.rho_bcxr = 1.0
        self.vx_bcxr = 0.0
        self.vy_bcxr = 0.0
        self.vz_bcxr = 0.0

        # Boundary y-axis left side
        self.bc_y_left = 0
        self.rho_bcyl = 1.0
        self.vx_bcyl = 0.0
        self.vy_bcyl = 0.0
        self.vz_bcyl = 0.0

        # Boundary y-axis right side
        self.bc_y_right = 0
        self.rho_bcyr = 1.0
        self.vx_bcyr = 0.0
        self.vy_bcyr = 0.0
        self.vz_bcyr = 0.0

        # Boundary z-axis left side
        self.bc_z_left = 0
        self.rho_bczl = 1.0
        self.vx_bczl = 0.0
        self.vy_bczl = 0.0
        self.vz_bczl = 0.0

        # Boundary z-axis right side
        self.bc_z_right = 0
        self.rho_bczr = 1.0
        self.vx_bczr = 0.0
        self.vy_bczr = 0.0
        self.vz_bczr = 0.0

        if sparse_storage == False:
            self.f = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz))
            self.F = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz))
            self.rho = ti.field(ti.f32, shape=(nx, ny, nz))
            self.v = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))
        else:
            self.f = ti.Vector.field(19, ti.f32)
            self.F = ti.Vector.field(19, ti.f32)
            self.rho = ti.field(ti.f32)
            self.v = ti.Vector.field(3, ti.f32)
            n_mem_partition = 3

            cell1 = ti.root.pointer(
                ti.ijk,
                (
                    nx // n_mem_partition + 1,
                    ny // n_mem_partition + 1,
                    nz // n_mem_partition + 1,
                ),
            )
            cell1.dense(
                ti.ijk, (n_mem_partition, n_mem_partition, n_mem_partition)
            ).place(self.rho, self.v, self.f, self.F)

        self.e = ti.Vector.field(3, ti.i32, shape=(19))
        self.S_dig = ti.Vector.field(19, ti.f32, shape=())
        self.e_f = ti.Vector.field(3, ti.f32, shape=(19))
        self.w = ti.field(ti.f32, shape=(19))
        self.ext_f = ti.Vector.field(3, ti.f32, shape=())

        self.M = ti.Matrix.field(19, 19, ti.f32, shape=())
        self.inv_M = ti.Matrix.field(19, 19, ti.f32, shape=())

        M_np = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, -2, -2, -2, -2, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, -2, 2, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1],
                [0, 0, 0, -2, 2, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1],
                [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, -2, 2, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
                [0, 2, 2, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2],
                [0, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2],
                [0, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0],
                [0, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, -1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, -1, 1, 1, -1],
            ]
        )
        inv_M_np = np.linalg.inv(M_np)

        self.LR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]

        self.M[None] = ti.Matrix(M_np)

        self.inv_M[None] = ti.Matrix(inv_M_np)

        self.x = np.linspace(0, nx, nx)
        self.y = np.linspace(0, ny, ny)
        self.z = np.linspace(0, nz, nz)

    def init_simulation(self):
        self.bc_vel_x_left = [self.vx_bcxl, self.vy_bcxl, self.vz_bcxl]
        self.bc_vel_x_right = [self.vx_bcxr, self.vy_bcxr, self.vz_bcxr]
        self.bc_vel_y_left = [self.vx_bcyl, self.vy_bcyl, self.vz_bcyl]
        self.bc_vel_y_right = [self.vx_bcyr, self.vy_bcyr, self.vz_bcyr]
        self.bc_vel_z_left = [self.vx_bczl, self.vy_bczl, self.vz_bczl]
        self.bc_vel_z_right = [self.vx_bczr, self.vy_bczr, self.vz_bczr]

        self.tau_f = self.niu / 3.0 + 0.5
        self.s_v = 1.0 / self.tau_f
        self.s_other = 8.0 * (2.0 - self.s_v) / (8.0 - self.s_v)

        self.S_dig[None] = ti.Vector(
            [
                0,
                self.s_v,
                self.s_v,
                0,
                self.s_other,
                0,
                self.s_other,
                0,
                self.s_other,
                self.s_v,
                self.s_v,
                self.s_v,
                self.s_v,
                self.s_v,
                self.s_v,
                self.s_v,
                self.s_other,
                self.s_other,
                self.s_other,
            ]
        )

        self.ext_f[None][0] = self.fx
        self.ext_f[None][1] = self.fy
        self.ext_f[None][2] = self.fz
        if (abs(self.fx) > 0) or (abs(self.fy) > 0) or (abs(self.fz) > 0):
            self.force_flag = 1
        else:
            self.force_flag = 0
        ti.static(self.inv_M)
        ti.static(self.M)
        ti.static(self.S_dig)
        self.static_init()
        self.init()

    @ti.func
    def feq(self, k, rho_local, u):
        eu = self.e[k].dot(u)
        uv = u.dot(u)
        feqout = self.w[k] * rho_local * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
        return feqout

    @ti.kernel
    def init(self):
        for i, j, k in self.solid:
            if (self.sparse_storage == False) or (self.solid[i, j, k] == 0):
                self.rho[i, j, k] = 1.0
                self.v[i, j, k] = ti.Vector([0, 0, 0])
                for s in ti.static(range(19)):
                    self.f[i, j, k][s] = self.feq(s, 1.0, self.v[i, j, k])
                    self.F[i, j, k][s] = self.feq(s, 1.0, self.v[i, j, k])

    @ti.kernel
    def static_init(self):
        if ti.static(self.enable_projection):  # No runtime overhead
            self.e[0] = ti.Vector([0, 0, 0])
            self.e[1] = ti.Vector([1, 0, 0])
            self.e[2] = ti.Vector([-1, 0, 0])
            self.e[3] = ti.Vector([0, 1, 0])
            self.e[4] = ti.Vector([0, -1, 0])
            self.e[5] = ti.Vector([0, 0, 1])
            self.e[6] = ti.Vector([0, 0, -1])
            self.e[7] = ti.Vector([1, 1, 0])
            self.e[8] = ti.Vector([-1, -1, 0])
            self.e[9] = ti.Vector([1, -1, 0])
            self.e[10] = ti.Vector([-1, 1, 0])
            self.e[11] = ti.Vector([1, 0, 1])
            self.e[12] = ti.Vector([-1, 0, -1])
            self.e[13] = ti.Vector([1, 0, -1])
            self.e[14] = ti.Vector([-1, 0, 1])
            self.e[15] = ti.Vector([0, 1, 1])
            self.e[16] = ti.Vector([0, -1, -1])
            self.e[17] = ti.Vector([0, 1, -1])
            self.e[18] = ti.Vector([0, -1, 1])

            self.e_f[0] = ti.Vector([0, 0, 0])
            self.e_f[1] = ti.Vector([1, 0, 0])
            self.e_f[2] = ti.Vector([-1, 0, 0])
            self.e_f[3] = ti.Vector([0, 1, 0])
            self.e_f[4] = ti.Vector([0, -1, 0])
            self.e_f[5] = ti.Vector([0, 0, 1])
            self.e_f[6] = ti.Vector([0, 0, -1])
            self.e_f[7] = ti.Vector([1, 1, 0])
            self.e_f[8] = ti.Vector([-1, -1, 0])
            self.e_f[9] = ti.Vector([1, -1, 0])
            self.e_f[10] = ti.Vector([-1, 1, 0])
            self.e_f[11] = ti.Vector([1, 0, 1])
            self.e_f[12] = ti.Vector([-1, 0, -1])
            self.e_f[13] = ti.Vector([1, 0, -1])
            self.e_f[14] = ti.Vector([-1, 0, 1])
            self.e_f[15] = ti.Vector([0, 1, 1])
            self.e_f[16] = ti.Vector([0, -1, -1])
            self.e_f[17] = ti.Vector([0, 1, -1])
            self.e_f[18] = ti.Vector([0, -1, 1])

            self.w[0] = 1.0 / 3.0
            self.w[1] = 1.0 / 18.0
            self.w[2] = 1.0 / 18.0
            self.w[3] = 1.0 / 18.0
            self.w[4] = 1.0 / 18.0
            self.w[5] = 1.0 / 18.0
            self.w[6] = 1.0 / 18.0
            self.w[7] = 1.0 / 36.0
            self.w[8] = 1.0 / 36.0
            self.w[9] = 1.0 / 36.0
            self.w[10] = 1.0 / 36.0
            self.w[11] = 1.0 / 36.0
            self.w[12] = 1.0 / 36.0
            self.w[13] = 1.0 / 36.0
            self.w[14] = 1.0 / 36.0
            self.w[15] = 1.0 / 36.0
            self.w[16] = 1.0 / 36.0
            self.w[17] = 1.0 / 36.0
            self.w[18] = 1.0 / 36.0

    @ti.func
    def meq_vec(self, rho_local, u):
        out = ti.Vector([0.0]*19)
        out[0] = rho_local
        out[3] = u[0]
        out[5] = u[1]
        out[7] = u[2]
        out[1] = u.dot(u)
        out[9] = 2 * u.x * u.x - u.y * u.y - u.z * u.z
        out[11] = u.y * u.y - u.z * u.z
        out[13] = u.x * u.y
        out[14] = u.y * u.z
        out[15] = u.x * u.z
        return out

    @ti.func
    def calc_local_force(self, i, j, k):
        f = ti.Vector([self.fx, self.fy, self.fz])
        return f

    @ti.kernel
    def collision(self):
        for i, j, k in self.rho:
            if self.solid[i, j, k] == 0 and (i < self.nx)*(j < self.ny)*(k < self.nz):
                m_temp = self.M[None] @ self.F[i, j, k]
                meq = self.meq_vec(self.rho[i, j, k], self.v[i, j, k])
                m_temp -= self.S_dig[None] * (m_temp - meq)
                f = self.calc_local_force(i, j, k)
                if ti.static(self.force_flag == 1):
                    for s in ti.static(range(19)):
                        f_guo = 0.0
                        for idx in ti.static(range(19)):
                            f_guo += (
                                self.w[idx]
                                * (
                                    (self.e_f[idx] - self.v[i, j, k]).dot(f) / 3.0
                                    + (
                                        self.e_f[idx].dot(self.v[i, j, k])
                                        * (self.e_f[idx].dot(f))
                                    )
                                    / 9.0
                                )
                                * self.M[None][s, idx]
                            )
                        m_temp[s] += (1 - 0.5 * self.S_dig[None][s]) * f_guo

                self.f[i, j, k] = ti.Vector(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
                self.f[i, j, k] += self.inv_M[None] @ m_temp

    @ti.func
    def periodic_index(self, i):
        iout = i
        if i[0] < 0:
            iout[0] = self.nx - 1
        if i[0] > self.nx - 1:
            iout[0] = 0
        if i[1] < 0:
            iout[1] = self.ny - 1
        if i[1] > self.ny - 1:
            iout[1] = 0
        if i[2] < 0:
            iout[2] = self.nz - 1
        if i[2] > self.nz - 1:
            iout[2] = 0

        return iout

    @ti.kernel
    def streaming1(self):
        for i in ti.grouped(self.rho):
            if self.solid[i] == 0 and (i.x < self.nx)*(i.y < self.ny)*(i.z < self.nz):
                for s in ti.static(range(19)):
                    ip = self.periodic_index(i + self.e[s])
                    if self.solid[ip] == 0:
                        self.F[ip][s] = self.f[i][s]
                    else:
                        self.F[i][self.LR[s]] = self.f[i][s]

    @ti.kernel
    def boundary_condition(self):
        if ti.static(self.bc_x_left == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[0, j, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[1, j, k] > 0:
                            self.F[0, j, k][s] = self.feq(
                                s, self.rho_bcxl, self.v[1, j, k]
                            )
                        else:
                            self.F[0, j, k][s] = self.feq(
                                s, self.rho_bcxl, self.v[0, j, k]
                            )

        if ti.static(self.bc_x_left == 2):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[0, j, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[0, j, k][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_x_left)
                        )

        if ti.static(self.bc_x_right == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[self.nx - 1, j, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[self.nx - 2, j, k] > 0:
                            self.F[self.nx - 1, j, k][s] = self.feq(
                                s, self.rho_bcxr, self.v[self.nx - 2, j, k]
                            )
                        else:
                            self.F[self.nx - 1, j, k][s] = self.feq(
                                s, self.rho_bcxr, self.v[self.nx - 1, j, k]
                            )

        if ti.static(self.bc_x_right == 2):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[self.nx - 1, j, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[self.nx - 1, j, k][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_x_right)
                        )

        # Direction Y
        if ti.static(self.bc_y_left == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, 0, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, 1, k] > 0:
                            self.F[i, 0, k][s] = self.feq(
                                s, self.rho_bcyl, self.v[i, 1, k]
                            )
                        else:
                            self.F[i, 0, k][s] = self.feq(
                                s, self.rho_bcyl, self.v[i, 0, k]
                            )

        if ti.static(self.bc_y_left == 2):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, 0, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, 0, k][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_y_left)
                        )

        if ti.static(self.bc_y_right == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, self.ny - 1, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, self.ny - 2, k] > 0:
                            self.F[i, self.ny - 1, k][s] = self.feq(
                                s, self.rho_bcyr, self.v[i, self.ny - 2, k]
                            )
                        else:
                            self.F[i, self.ny - 1, k][s] = self.feq(
                                s, self.rho_bcyr, self.v[i, self.ny - 1, k]
                            )

        if ti.static(self.bc_y_right == 2):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, self.ny - 1, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, self.ny - 1, k][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_y_right)
                        )

        # Z direction
        if ti.static(self.bc_z_left == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, 0] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, j, 1] > 0:
                            self.F[i, j, 0][s] = self.feq(
                                s, self.rho_bczl, self.v[i, j, 1]
                            )
                        else:
                            self.F[i, j, 0][s] = self.feq(
                                s, self.rho_bczl, self.v[i, j, 0]
                            )

        if ti.static(self.bc_z_left == 2):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, 0] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, j, 0][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_z_left)
                        )

        if ti.static(self.bc_z_right == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, self.nz - 1] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, j, self.nz - 2] > 0:
                            self.F[i, j, self.nz - 1][s] = self.feq(
                                s, self.rho_bczr, self.v[i, j, self.nz - 2]
                            )
                        else:
                            self.F[i, j, self.nz - 1][s] = self.feq(
                                s, self.rho_bczr, self.v[i, j, self.nz - 1]
                            )

        if ti.static(self.bc_z_right == 2):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, self.nz - 1] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, j, self.nz - 1][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_z_right)
                        )

    @ti.kernel
    def streaming3(self):
        for i in ti.grouped(self.rho):
            if self.solid[i] == 0 and (i.x < self.nx)*(i.y < self.ny)*(i.z < self.nz):
                self.rho[i] = 0
                self.v[i] = ti.Vector([0, 0, 0])
                self.f[i] = self.F[i]
                self.rho[i] += self.f[i].sum()

                for s in ti.static(range(19)):
                    self.v[i] += self.e_f[s] * self.f[i][s]

                f = self.calc_local_force(i.x, i.y, i.z)

                self.v[i] /= self.rho[i]
                self.v[i] += (f / 2) / self.rho[i]

            else:
                self.rho[i] = 1.0
                self.v[i] = ti.Vector([0, 0, 0])

    def get_max_v(self):
        self.max_v[None] = -1e10
        self.calc_max_v()
        return self.max_v[None]

    @ti.kernel
    def calc_max_v(self):
        for idx in ti.grouped(self.rho):
            ti.atomic_max(self.max_v[None], self.v[idx].norm())

    def set_bc_vel_x1(self, vel):
        self.bc_x_right = 2
        self.vx_bcxr = vel[0]
        self.vy_bcxr = vel[1]
        self.vz_bcxr = vel[2]

    def set_bc_vel_x0(self, vel):
        self.bc_x_left = 2
        self.vx_bcxl = vel[0]
        self.vy_bcxl = vel[1]
        self.vz_bcxl = vel[2]

    def set_bc_vel_y1(self, vel):
        self.bc_y_right = 2
        self.vx_bcyr = vel[0]
        self.vy_bcyr = vel[1]
        self.vz_bcyr = vel[2]

    def set_bc_vel_y0(self, vel):
        self.bc_y_left = 2
        self.vx_bcyl = vel[0]
        self.vy_bcyl = vel[1]
        self.vz_bcyl = vel[2]

    def set_bc_vel_z1(self, vel):
        self.bc_z_right = 2
        self.vx_bczr = vel[0]
        self.vy_bczr = vel[1]
        self.vz_bczr = vel[2]

    def set_bc_vel_z0(self, vel):
        self.bc_z_left = 2
        self.vx_bczl = vel[0]
        self.vy_bczl = vel[1]
        self.vz_bczl = vel[2]

    def set_bc_rho_x0(self, rho):
        self.bc_x_left = 1
        self.rho_bcxl = rho

    def set_bc_rho_x1(self, rho):
        self.bc_x_right = 1
        self.rho_bcxr = rho

    def set_bc_rho_y0(self, rho):
        self.bc_y_left = 1
        self.rho_bcyl = rho

    def set_bc_rho_y1(self, rho):
        self.bc_y_right = 1
        self.rho_bcyr = rho

    def set_bc_rho_z0(self, rho):
        self.bc_z_left = 1
        self.rho_bczl = rho

    def set_bc_rho_z1(self, rho):
        self.bc_z_right = 1
        self.rho_bczr = rho

    def set_viscosity(self, niu):
        self.niu = niu

    def set_force(self, force):
        self.fx = force[0]
        self.fy = force[1]
        self.fz = force[2]

    def export_VTK(self, n, prefix="LB_SingelPhase"):
        gridToVTK(
            f"./{prefix}_{n}",
            self.x,
            self.y,
            self.z,
            pointData={
                "Solid": np.ascontiguousarray(self.solid.to_numpy()),
                "rho": np.ascontiguousarray(self.rho.to_numpy()),
                "velocity": (
                    np.ascontiguousarray(
                        self.v.to_numpy()[..., 0]
                    ),
                    np.ascontiguousarray(
                        self.v.to_numpy()[..., 1]
                    ),
                    np.ascontiguousarray(
                        self.v.to_numpy()[..., 2]
                    ),
                ),
            },
        )

    def step(self):
        self.collision()
        self.streaming1()
        self.boundary_condition()
        self.streaming3()
