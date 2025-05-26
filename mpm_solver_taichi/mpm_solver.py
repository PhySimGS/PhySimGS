import numpy
import numpy as np
import taichi as ti

dim, n_grid, steps, max_dt, dt = 3, 256, 25, 1e-5, 1e-5

n_particles = 8000000
old_n_particles = n_particles
dx, inv_dx = 1 / n_grid, float(n_grid)
p_rho = 1
show_n_particles = 205000
d = 3  # Dimension updated to 3

E_array, nu_array = ti.field(dtype=ti.f64, shape=n_particles), ti.field(dtype=ti.f64,
                                                                       shape=n_particles)  # Young's modulus and Poisson's ratio
p_vol_array = ti.field(dtype=ti.f64, shape=n_particles)  # particle volume
density_array = ti.field(dtype=ti.f64, shape=n_particles)  # density
global_mu = 0.95

x = ti.Vector.field(3, dtype=ti.f64, shape=n_particles)  # position
x_freeze = ti.Vector.field(3, dtype=float, shape=n_particles)  # position
temp_x = ti.Vector.field(3, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(3, dtype=ti.f64, shape=n_particles)  # velocity
C = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n_particles)  # deformation gradient
R = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n_particles)  # rotation matrix
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=ti.f64, shape=n_particles)  # plastic deformation
Cov = ti.field(dtype=ti.f64, shape=6 * n_particles)  # affine velocity field
init_Cov = ti.field(dtype=ti.f64, shape=6 * n_particles)  # affine velocity field

grid_v = ti.Vector.field(3, dtype=ti.f64, shape=(n_grid, n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=ti.f64, shape=(n_grid, n_grid, n_grid))  # grid node mass
grid_f = ti.Vector.field(3, dtype=ti.f64, shape=(n_grid, n_grid, n_grid))  # 3D grid forces

# grid_collider = ti.field(dtype=int, shape=(n_grid, n_grid, n_grid))  # grid node mass
trans_offset = ti.Vector.field(3, dtype=ti.f64, shape=(1))
move_offset = ti.Vector.field(3, dtype=ti.f64, shape=(1))
trans_scale = ti.field(dtype=ti.f64, shape=(1))
points_labels = ti.field(dtype=int, shape=n_particles)
phi_s = ti.field(dtype=ti.f64, shape=n_particles)  # Cohesion and saturation
c_C0 = ti.field(dtype=ti.f64, shape=n_particles)  # Initial cohesion
vc_s = ti.field(dtype=ti.f64, shape=n_particles)  # Volume change tracker
alpha_s = ti.field(dtype=ti.f64, shape=n_particles)  # Yield surface size
q = ti.field(dtype=ti.f64, shape=n_particles)  # Hardening state

gravity = ti.Vector([0, -9.8, 0])  # 3D gravity vector
bound = 3
ground = ti.field(dtype=int, shape=(1))
background = ti.field(dtype=int, shape=(n_grid, n_grid, n_grid))

p_vol, s_rho = (dx * 0.5) ** 3, 400  # Volume updated for 3D
s_mass = p_vol * s_rho
E_s, nu_s = 3.537e5, 0.1  # Sand's Young's modulus and Poisson's ratio
E_e, nu_e = 1e5, 0.2  # Elastic material's Young's modulus and Poisson's ratio
mu_b = 0.75  # Friction coefficient
ELASTIC = 1
SAND = 3
state = ti.field(dtype=int, shape=n_particles)
pi = 3.14159265358979
h0, h1, h2, h3 = 35, 9, 0.2, 10
# cpic

x_rp = ti.Vector.field(3, dtype=float, shape=1600)
x_mask = ti.field( dtype=int, shape=n_particles)
force_pos = ti.Vector.field(3, dtype=float, shape=(1))

blade_move_vector = ti.Vector.field(3, dtype=float, shape=(1))
blade_normal = ti.Vector.field(3, dtype=float, shape=(1))
rebase_velo_point = ti.Vector.field(3, dtype=float, shape=(1))

# visualize
x_collider = ti.Vector.field(3, dtype=ti.types.f64, shape=6)

max_vel = 0.0


# Project function updated for 3D
@ti.func
def project(e0, p):
    mu_s, lambda_s = E_s / (2 * (1 + nu_s)), E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s))  # 沙子的拉梅参数
    e = e0 + vc_s[p] / d * ti.Matrix.identity(ti.f64, 3)
    e += c_C0[p] / (d * alpha_s[p]) * ti.Matrix.identity(ti.f64, 3)  # 无水时 phi_s[p] = 0
    ehat = e - e.trace() / d * ti.Matrix.identity(ti.f64, 3)
    Fnorm = ti.sqrt(ehat[0, 0] ** 2 + ehat[1, 1] ** 2 + ehat[2, 2] ** 2)
    yp = Fnorm + (d * lambda_s + 2 * mu_s) / (2 * mu_s) * e.trace() * alpha_s[p]
    new_e = ti.Matrix.zero(ti.f64, 3, 3)
    delta_q = 0.0
    if Fnorm <= 0 or e.trace() > 0:  # Case II
        new_e = ti.Matrix.zero(ti.f64, 3, 3)
        delta_q = ti.sqrt(e[0, 0] ** 2 + e[1, 1] ** 2 + e[2, 2] ** 2)
        state[p] = 0
    elif yp <= 0:  # Case I
        new_e = e0
        delta_q = 0
        state[p] = 1
    else:  # Case III
        new_e = e - yp / Fnorm * ehat
        delta_q = yp
        state[p] = 2
    return new_e, delta_q


@ti.func
def hardening(dq, p):
    q[p] += dq
    phi = h0 + (h1 * q[p] - h3) * ti.exp(-h2 * q[p])
    phi = phi / 180 * pi
    sin_phi = ti.sin(phi)
    alpha_s[p] = ti.sqrt(2 / 3) * (2 * sin_phi) / (3 - sin_phi)

@ti.func
def get_new_dt():
    max_vel = 0.0
    for p in range(n_particles):
        max_vel = max(max_vel, v[p].norm())
    new_dt = min(max_dt, 0.1 * dx / (max_vel + 1e-6))
    return new_dt

@ti.kernel
def get_new_dt_kernel()-> ti.f32:
    max_vel = 0.0
    for p in range(n_particles):
        max_vel = max(max_vel, v[p].norm())
    new_dt = min(max_dt, 0.1 * dx / (max_vel + 1e-6))
    return new_dt

@ti.kernel
def substep():

    dt = max_dt

    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
        grid_f[i, j, k] = [0, 0, 0]

    for p in range(n_particles):

        base = (x[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[2] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2 or base[2] >= n_grid - 2:
            continue
        fx = x[p] * inv_dx - base.cast(ti.f64)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        U, sig, V = ti.svd(F[p])
        update_rot(F[p], p)
        compute_cov_from_F(F[p], p)
        stress, affine = ti.Matrix.zero(ti.f64, 3, 3), ti.Matrix.zero(ti.f64, 3, 3)
        if material[p] == SAND:
            mu_s, lambda_s = E_s / (2 * (1 + nu_s)), E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s))  # 沙子的拉梅参数
            inv_sig = sig.inverse()
            e = ti.Matrix([[ti.log(sig[0, 0]), 0, 0], [0, ti.log(sig[1, 1]), 0], [0, 0, ti.log(sig[2, 2])]])
            stress = U @ (2 * mu_s * inv_sig @ e + lambda_s * e.trace() * inv_sig) @ V.transpose()
            stress = (-p_vol * 4 * inv_dx * inv_dx) * stress @ F[p].transpose()
            affine = s_mass * C[p]
        else:  # ELASTIC
            h = 1.0
            mu_0, lambda_0 = E_e / (2 * (1 + nu_e)), E_e * nu_e / ((1 + nu_e) * (1 - 2 * nu_e))
            mu, la = mu_0 * h, lambda_0 * h
            J = 1.0
            for d in ti.static(range(3)):
                J *= sig[d, d]
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(ti.f64, 3) * la * J * (J - 1)
            stress = (-dt * p_vol * 4 * inv_dx**3) * stress
            affine = stress + s_mass * C[p]
        # stress = ti.Matrix.zero(ti.f64, 3, 3) if stress.norm() > 0 else stress
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # 3x3x3 neighborhood
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(ti.f64) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (s_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * s_mass
            grid_f[base + offset] += weight * stress @ dpos

    # 网格更新和边界条件（保持不变）
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
            grid_v[i, j, k] += dt * (gravity + grid_f[i, j, k] / grid_m[i, j, k])
        normal = ti.Vector.zero(ti.f64, 3)
        if grid_m[i, j, k] > 0:
            if i < 3 and grid_v[i, j, k][0] < 0: normal = ti.Vector([1, 0, 0])
            if i > n_grid - 3 and grid_v[i, j, k][0] > 0: normal = ti.Vector([-1, 0, 0])
            if j < 3 and grid_v[i, j, k][1] < 0: normal = ti.Vector([0, 1, 0])
            if j > n_grid - 3 and grid_v[i, j, k][1] > 0: normal = ti.Vector([0, -1, 0])
            if k < 3 and grid_v[i, j, k][2] < 0: normal = ti.Vector([0, 0, 1])
            if k > n_grid - 3 and grid_v[i, j, k][2] > 0: normal = ti.Vector([0, 0, -1])
            if j <= int(ground[0]) and grid_v[i, j, k][1] < 0: normal = ti.Vector([0, 1, 0])
        # normal = ti.Vector([0, 1, 0])
        if normal.norm() > 0:  # Apply 3D friction
            s = grid_v[i, j, k].dot(normal)
            if s <= 0:
                v_normal = s * normal
                v_tangent = grid_v[i, j, k] - v_normal
                vt = v_tangent.norm()
                if vt > 1e-12:
                    grid_v[i, j, k] = v_tangent - min(vt, -mu_b * s) * (v_tangent / vt)
                else:
                    grid_v[i, j, k] = v_tangent

    # 网格到粒子（G2P，保持不变）
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[2] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2 or base[2] >= n_grid - 2:
            continue
        fx = x[p] * inv_dx - base.cast(ti.f64)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f64, 3)
        new_C = ti.Matrix.zero(ti.f64, 3, 3)
        phi_s[p] = 0.0  # No water, saturation is 0
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = ti.Vector([i, j, k]).cast(ti.f64) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        F[p] = (ti.Matrix.identity(ti.f64, 3) + dt * new_C) @ F[p]
        # 避免速度出现nan，如果nan则将速度置为0
        # if new_v.norm() > 1e3:
        #     new_v = ti.Vector.zero(ti.f64, 3)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]
        if material[p] == SAND:
            U, sig, V = ti.svd(F[p])
            e = ti.Matrix([[ti.log(sig[0, 0]), 0, 0], [0, ti.log(sig[1, 1]), 0], [0, 0, ti.log(sig[2, 2])]])
            new_e, dq = project(e, p)
            hardening(dq, p)
            new_F = U @ ti.Matrix([[ti.exp(new_e[0, 0]), 0, 0], [0, ti.exp(new_e[1, 1]), 0], [0, 0, ti.exp(new_e[2, 2])]]) @ V.transpose()
            vc_s[p] += -ti.log(new_F.determinant()) + ti.log(F[p].determinant())
            F[p] = new_F


@ti.kernel
def clear_v():
    for p in v:
        v[p] = ti.Vector([0, 0, 0])
        F[p] = ti.Matrix.identity(float, 3)
        C[p] = ti.Matrix.zero(float, 3, 3)
        Jp[p] = 1

@ti.func
def cal_det(matrix: ti.template()):
    return matrix[0, 0] * matrix[1, 1] * matrix[2, 2] + matrix[0, 1] * matrix[1, 2] * matrix[2, 0] + matrix[0, 2] * \
           matrix[1, 0] * matrix[2, 1] - matrix[0, 2] * matrix[1, 1] * matrix[2, 0] - matrix[0, 1] * matrix[1, 0] * \
           matrix[2, 2] - matrix[0, 0] * matrix[1, 2] * matrix[2, 1]

@ti.func
def update_rot(_F:ti.template(),p:int):
    # U, s, Vt = np.linalg.svd(_F.to_numpy())
    U, sig, V = ti.svd(_F)
    # V=Vt.transpose()
    det_u = cal_det(U)
    det_v = cal_det(V)

    if det_u < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if det_v < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    R[p] = U @ V.transpose()
    R[p] = R[p].transpose()
    return V

def update_cov():
    _F = F.to_numpy()
    length = int(len(init_Cov)/6)
    for p in range(length):
        init_cov = ti.Matrix.zero(ti.f64, 3, 3)
        init_cov[0, 0] = init_Cov[p * 6]
        init_cov[0, 1] = init_Cov[p * 6 + 1]
        init_cov[0, 2] = init_Cov[p * 6 + 2]
        init_cov[1, 0] = init_Cov[p * 6 + 1]
        init_cov[1, 1] = init_Cov[p * 6 + 3]
        init_cov[1, 2] = init_Cov[p * 6 + 4]
        init_cov[2, 0] = init_Cov[p * 6 + 2]
        init_cov[2, 1] = init_Cov[p * 6 + 4]
        init_cov[2, 2] = init_Cov[p * 6 + 5]

        cov = _F[p] * init_cov * _F[p].transpose()

        Cov[p * 6] = cov[0, 0]
        Cov[p * 6 + 1] = cov[0, 1]
        Cov[p * 6 + 2] = cov[0, 2]
        Cov[p * 6 + 3] = cov[1, 1]
        Cov[p * 6 + 4] = cov[1, 2]
        Cov[p * 6 + 5] = cov[2, 2]

@ti.kernel
def clear():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
    for p in x:
        x[p] = [0, 0, 0]
        v[p] = [0, 0, 0]
        F[p] = ti.Matrix.identity(float, 3)
        C[p] = ti.Matrix.zero(float, 3, 3)
        Jp[p] = 1
        material[p] = 0
        E_array[p] = 0
        nu_array[p] = 0
        c_C0[p] = -0.01
        alpha_s[p] = 0.267765
        for i in range(6):
            Cov[p * 6 + i] = 0


@ti.kernel
def clear_intermediate_results():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
    for p in x:
        v[p] = [0, 0, 0]
        F[p] = ti.Matrix.identity(float, 3)
        C[p] = ti.Matrix.zero(float, 3, 3)
        Jp[p] = 1
        for i in range(6):
            Cov[p * 6 + i] = 0

@ti.kernel
def set_x(_x: ti.types.ndarray(), max_values: ti.types.ndarray(), min_values: ti.types.ndarray(),
          _offset: ti.types.ndarray()):
    for p in x_rp:
        x_rp[p] = ti.Vector([0, 0, 0])
    # for p in x:
    #     x[p]= [0,0,0]
    # for p in range(_x.shape[0]):
    #     x[p] = ti.Vector([_x[p,0], _x[p,1], _x[p,2]])
    min_x, min_y, min_z = min_values[0], min_values[1], min_values[2]
    max_x, max_y, max_z = max_values[0], max_values[1], max_values[2]

    offset = ti.Vector([min_x, min_y, min_z])
    trans_offset[0] = offset
    move_offset[0] = ti.Vector([_offset[0], _offset[1], _offset[2]])
    scale = 1 / (ti.max(max_x - min_x, ti.max(max_y - min_y, max_z - min_z)))
    trans_scale[0] = scale / 1.2

    for p in range(_x.shape[0]):
        x[p] = (x[p] - offset) * trans_scale[0] + move_offset[0]
        # x[p] = [0,0,0]


@ti.kernel
def set_v(_v: ti.types.ndarray()):
    for p in v:
        v[p] = ti.Vector([_v[p, 0], _v[p, 1], _v[p, 2]])


@ti.kernel
def set_additional_v_points(pos: ti.types.ndarray()):
    pass
    grid_pos = (ti.Vector([pos[0], pos[1], pos[2]]) * inv_dx - 0.5).cast(int)
    force_pos[0] = ti.Vector([pos[0], pos[1], pos[2]])
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int) - grid_pos
        if abs(base[0]) < 3 and abs(base[1]) < 3 and abs(base[2]) < 30:
            x_mask[p] = 1
        else:
            x_mask[p] = 0


@ti.kernel
def update_v_force(new_v: ti.types.ndarray()):
    v_new = ti.Vector([new_v[0], new_v[1], new_v[2]])
    for p in x:
        if x_mask[p] == 1:
            v[p][0] = v[p][0] + ti.f32(v_new[0])
            v[p][1] = v[p][1] + ti.f32(v_new[1])
            v[p][2] = v[p][2] + ti.f32(v_new[2])
            # v[p] += v_new


@ti.kernel
def set_material(_material: ti.types.ndarray()):
    for p in material:
        # material[p] = _material[p]
        material[p] = 1

@ti.kernel
def _set_material_with_labels(_material: ti.int32, label: ti.int32):
    for p in material:
        if points_labels[p] == label or label == -1:
            material[p] = _material
def set_material_with_labels(label_to_material=None):
    material.from_numpy(np.ones(old_n_particles, dtype=np.int32))
    if label_to_material is None:
        _set_material_with_labels(1, -1)
    else:
        for label, _material in label_to_material.items():
            _set_material_with_labels(_material, label)

@ti.kernel
def set_material_grid(_material: ti.types.ndarray()):
    for p in material:
        base = (x[p] * inv_dx - 0.5).cast(int)
        material[p] = _material[base[0], base[1], base[2]]


@ti.kernel
def set_gravity(_gravity: ti.template()):
    pass
    # gravity[0] = _gravity[0]
    # gravity[1] = _gravity[1]
    # gravity[2] = _gravity[2]


@ti.kernel
def set_E_array(_E_array: ti.types.ndarray()):
    for p in E_array:
        E_array[p] = _E_array[p, 0]


@ti.kernel
def set_nu_array(_nu_array: ti.types.ndarray()):
    for p in nu_array:
        nu_array[p] = _nu_array[p, 0]


@ti.kernel
def set_p_vol_array(_p_vol_array: ti.types.ndarray()):
    for p in p_vol_array:
        # p_vol_array[p] = _p_vol_array[p]
        p_vol_array[p] = (dx * 0.5) ** 3

@ti.kernel
def set_density_array(_density_array: ti.types.ndarray()):
    for p in density_array:
        density_array[p] = _density_array[p,0]

def set_Cov(_Cov):
    reshape_cov = _Cov.reshape(-1)
    # 填补为 0
    fill_cov = np.zeros(6 * old_n_particles - len(reshape_cov))
    reshape_cov = np.concatenate((reshape_cov, fill_cov), axis=0)
    init_Cov.from_numpy(reshape_cov)


@ti.kernel
def set_ground(_ground: ti.template()):
    ground[0] = int(_ground * inv_dx)


grid_background = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))

@ti.kernel
def set_background():
    # 在调用这个函数前就手动传递了temp_x
    for I in ti.grouped(background):
        background[I] = 0
    for p in x_freeze:
        offset = (((x_freeze[p] - trans_offset[0]) * trans_scale[0] + move_offset[0]) * inv_dx - 0.5).cast(int)

        if not(offset[0] < n_grid and offset[1] < n_grid and offset[2] < n_grid and
               offset[0] >= 0 and offset[1] >= 0 and offset[2] >= 0):
            continue
        fx = ((x_freeze[p] - trans_offset[0]) * trans_scale[0] + move_offset[0]) * inv_dx - offset.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_background[offset + ti.Vector([i, j, k])] += weight

        # if offset[0] < n_grid and offset[1] < n_grid and offset[2] < n_grid and \
        #         offset[0] >= 0 and offset[1] >= 0 and offset[2] >= 0:
        #     background[offset[0], offset[1], offset[2]] = 1
        # background[] = 1
        x_freeze[p] = (x_freeze[p] - trans_offset[0]) * trans_scale[0] + move_offset[0]
        x_freeze[p] = ((x_freeze[p] - move_offset[0]) / trans_scale[0]) + trans_offset[0]

    for I in ti.grouped(grid_background):
        if grid_background[I] > 1.5:
            background[I] = 1
        else:
            background[I] = 0

@ti.kernel
def _apply_displacement(_offset: ti.types.ndarray(), label:ti.template()):
    for p in x:
        if label == -1 or points_labels[p] == label:
            x[p] = x[p] + ti.Vector([_offset[0], _offset[1], _offset[2]])

def apply_displacement(_offset, move_labels=None):
    if move_labels is None:
        _apply_displacement(_offset, -1)
    else:
        for label in move_labels:
            _apply_displacement(_offset, label)

def set_blade(p1, p2, p3, p4, blade_velo, direction=None, n_points_u=40, n_points_v=40):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    v3 = np.array(p4) - np.array(p1)

    # 计算叉积
    cross_product = np.cross(v2, v3)
    normal_vector = cross_product

    # 计算混合积
    mixed_product = np.dot(v1, cross_product)

    # 判断是否共面
    if not np.isclose(mixed_product, 0):
        print('Error, can\'t init cpic blade. Exit.')
        exit()

    _direction = direction
    if _direction == None:
        _direction = v1
    # normalize
    _direction = _direction / np.linalg.norm(_direction)
    _blade_move_vector = np.array(_direction) * blade_velo
    _blade_normal = normal_vector
    _rebasing_point = np.array(p1)

    u_values = np.linspace(0, 1, n_points_u)
    v_values = np.linspace(0, 1, n_points_v)
    points = []
    _p1, _p2, _p3, _p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
    for u in u_values:
        for v in v_values:
            Puv = (1 - u) * (1 - v) * _p1 + u * (1 - v) * _p2 + (1 - u) * v * _p3 + u * v * _p4
            points.append(Puv)
    points = np.array(points)
    init_blade(points, _blade_normal, _blade_move_vector, _rebasing_point)
    x_rp.from_numpy(points)
    pass

def set_points_labels(_points_labels):
    points_labels.from_numpy(_points_labels)


@ti.kernel
def init_blade(points: ti.types.ndarray(),
               _blade_normal: ti.types.ndarray(),
               _blade_move_vector: ti.types.ndarray(),
               _rebasing_point: ti.types.ndarray()):
    blade_normal[0] = ti.Vector([_blade_normal[0], _blade_normal[1], _blade_normal[2]])
    blade_move_vector[0] = ti.Vector([_blade_move_vector[0], _blade_move_vector[1], _blade_move_vector[2]])
    rebase_velo_point[0] = ti.Vector([_rebasing_point[0], _rebasing_point[1], _rebasing_point[2]])
    pass


@ti.kernel
def update_blade():
    for p in x_rp:
        x_rp[p] += blade_move_vector[0]

def get_x():
    x_from_mpm_to_outside()
    x_numpy = temp_x.to_numpy()[:show_n_particles, ...]
    # x_from_outside_to_mpm()
    return x_numpy

def get_v():
    return v.to_numpy()[:show_n_particles, ...]


def get_F():
    return F.to_numpy()[:show_n_particles, ...]

def get_Cov():
    # compute_cov_from_F()
    return Cov.to_numpy()[:(show_n_particles * 6), ...]

def get_R():
    return R.to_numpy()[:show_n_particles, ...]

def get_C():
    return C.to_numpy()[:show_n_particles, ...]

@ti.kernel
def x_from_mpm_to_outside():
    for p in x:
        temp_x[p] = ((x[p] - move_offset[0]) / trans_scale[0]) + trans_offset[0]

@ti.kernel
def x_from_outside_to_mpm():
    for p in x:
        x[p] = (x[p] - trans_offset[0]) * trans_scale[0] + move_offset[0]

def single_x_from_mpm_to_outside(_x):
    _move_offset = (move_offset.to_numpy())[0]
    _trans_scale = (trans_scale.to_numpy())[0]
    _trans_offset = (trans_offset.to_numpy())[0]
    return ((_x-_move_offset)/trans_scale)+trans_offset

def single_x_from_outside_to_mpm(_x):
    _move_offset = (move_offset.to_numpy())[0]
    _trans_scale = (trans_scale.to_numpy())[0]
    _trans_offset = (trans_offset.to_numpy())[0]
    return (_x-_trans_offset)*_trans_scale + _move_offset

@ti.func
def compute_cov_from_F(_F: ti.template(), p: int):
    init_cov = ti.Matrix.zero(ti.f64, 3, 3)
    init_cov[0, 0] = init_Cov[p * 6]
    init_cov[0, 1] = init_Cov[p * 6 + 1]
    init_cov[0, 2] = init_Cov[p * 6 + 2]
    init_cov[1, 0] = init_Cov[p * 6 + 1]
    init_cov[1, 1] = init_Cov[p * 6 + 3]
    init_cov[1, 2] = init_Cov[p * 6 + 4]
    init_cov[2, 0] = init_Cov[p * 6 + 2]
    init_cov[2, 1] = init_Cov[p * 6 + 4]
    init_cov[2, 2] = init_Cov[p * 6 + 5]

    cov = _F @ init_cov @ _F.transpose()

    Cov[p * 6] = cov[0, 0]
    Cov[p * 6 + 1] = cov[0, 1]
    Cov[p * 6 + 2] = cov[0, 2]
    Cov[p * 6 + 3] = cov[1, 1]
    Cov[p * 6 + 4] = cov[1, 2]
    Cov[p * 6 + 5] = cov[2, 2]


# gui = ti.ui.Window(name='MPM', res=(1080, 720), pos=(150, 150))
# scene = ti.ui.Scene()
# camera = ti.ui.Camera()
# camera.position(0.5, 0.5, 2)  # x, y, z
# camera.lookat(0.5, 1, -6)
# # camera.position(-1,0.5,0.5)
# # camera.lookat(6,0.5,0.5)
# camera.projection_mode(ti.ui.ProjectionMode.Perspective)
# scene.set_camera(camera)
# canvas = gui.get_canvas()
# canvas.scene(scene)


def draw_gui():
    pass
    # scene.set_camera(camera)
    # scene.point_light(pos=(0.5, 0.5, 0.5), color=(1, 1, 1))
    # scene.point_light(pos=(0.5, 0.2, 0), color=(1, 1, 1))
    # # scene.point_light(pos=(0.5, 0, 0.5), color=(0.5, 0.5, 0.5))
    # # scene.point_light(pos=(1, 0.5, 0.5), color=(0.5, 0.5, 0.5))
    # # scene.point_light(pos=(0, 0.5, 0.5), color=(0.5, 0.5, 0.5))
    # # scene.point_light(pos=(0.5, 0.5, 0.5), color=(0.5, 0.5, 0.5))
    # # scene.point_light(pos=(0.3, 0.3, -0.3), color=(0.5, 0.5, 0.5))
    # # scene.ambient_light((0.8, 0.8, 0.8))
    # scene.particles(x, 0.004, (1, 0, 0))
    # scene.particles(x_rp, 0.005, (0, 1, 0))
    # scene.particles(force_pos, 0.02, (0, 0, 1))
    # scene.particles(x_collider, 0.01, (1, 0, 1))
    # pos = x.to_numpy()
    # canvas.scene(scene)
    # gui.show()
