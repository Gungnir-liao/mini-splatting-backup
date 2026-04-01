import numpy as np
import json
import math
from pyquaternion import Quaternion

def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def build_projection_matrix_js(fovRadian, ratio, zn, zf, p):
    """
    仿照你 JS 的 getProjectionMatrix 输出（row-major）
    注意：最终发送给 JS 时要以 column-major(flatten 'F') 输出。
    """
    yScale = 1.0 / math.tan(fovRadian / 2.0)
    xScale = yScale / ratio
    dx = 2.0 * p['x'] - 1.0
    dy = 2.0 * p['y'] - 1.0

    m = np.zeros((4, 4), dtype=float)
    # 按 row-major 填入（同 JS 中 Matrix4.set 的顺序）
    m[0, 0] = xScale; m[0, 1] = 0.0;   m[0, 2] = dx;                     m[0, 3] = 0.0
    m[1, 0] = 0.0;    m[1, 1] = yScale; m[1, 2] = dy;                     m[1, 3] = 0.0
    m[2, 0] = 0.0;    m[2, 1] = 0.0;    m[2, 2] = (zn + zf) / (zn - zf);  m[2, 3] = (2 * zn * zf) / (zn - zf)
    m[3, 0] = 0.0;    m[3, 1] = 0.0;    m[3, 2] = -1.0;                    m[3, 3] = 0.0
    return m

def build_view_matrix_via_compose(Eye, center, up):
    """
    严格复刻你 JS 中的过程：
    1) 计算 xAxis,yAxis,zAxis (与 JS 保持一模一样的顺序/方向)
    2) 构造旋转矩阵 R，使得其列向量为 (xAxis, yAxis, zAxis)
    3) 构造平移矩阵 T (将 position 放到最后一列，row-major)
    4) M = T @ R_homogeneous
    5) view = inv(M)
    6) 返回 column-major flatten 列表 (与 Three.js Matrix4.elements 等价)
    """
    Eye = np.array(Eye, dtype=float)
    center = np.array(center, dtype=float)
    up = np.array(up, dtype=float)

    # JS 里计算的 zAxis,xAxis,yAxis（注意顺序与标注）
    zAxis = normalize(Eye - center)                      # Eye - center (matches JS: zAxis = Eye - center)
    xAxis = normalize(np.cross(normalize(up), zAxis))    # up.normalize().cross(zAxis)
    yAxis = normalize(np.cross(zAxis, xAxis))            # zAxis.cross(xAxis)

    # 构造 4x4 旋转矩阵 R (row-major) —— JS 的 rotation.set(...) 所构造的矩阵
    # rotation.set(x.x, y.x, z.x, 0, x.y, y.y, z.y, 0, x.z, y.z, z.z, 0, 0,0,0,1)
    R = np.eye(4, dtype=float)
    R[0, 0], R[0, 1], R[0, 2] = xAxis[0], yAxis[0], zAxis[0]
    R[1, 0], R[1, 1], R[1, 2] = xAxis[1], yAxis[1], zAxis[1]
    R[2, 0], R[2, 1], R[2, 2] = xAxis[2], yAxis[2], zAxis[2]
    # R[0:3,3] are zeros, bottom row is [0,0,0,1] already

    # 平移矩阵 T (row-major): 将 position 放在最后一列
    T = np.eye(4, dtype=float)
    T[0, 3] = Eye[0]
    T[1, 3] = Eye[1]
    T[2, 3] = Eye[2]

    # Compose: M = T @ R  (scale=I)
    M = T @ R

    # viewMatrix = inv(M)
    view_row = np.linalg.inv(M)

    # Three.js 的 Matrix4.elements 是 column-major -> flatten order='F'
    view_colmajor_list = view_row.flatten(order='F').tolist()
    return view_row, view_colmajor_list

# ---------------- 主类 （保留你原逻辑，仅替换视图/投影的构造与输出） ----------------

class ViewportGenerator:
    def __init__(self):
        self._orbit = {
            'radius': 3.0,
            'theta': 0.0,
            'phi': 0.0,
            'roll': 0.0,
            'center': np.array([-0.113792, 1.26448, 0.421548]),
            'xAxis': np.array([-0.44662, -0.10908, 0.88805]),
            'yAxis': np.array([-0.110094, -0.978298, -0.175533]),
            'zAxis': np.array([-0.887924, 0.176165, -0.424919]),
            'move': np.array([0.0, 0.0, 0.0])
        }

        self.cameraParam = {
            'width': 1920,
            'height': 1080,
            'fov': 60.0 / 180.0 * math.pi,
            'aspect': 1920.0 / 1080.0,
            'znear': 0.009,
            'zfar': 1100,
            'p': {'x': 0.5, 'y': 0.5}
        }

        self.maxPolarAngle = 0.0
        self.minPolarAngle = -0.5 * math.pi * 0.8
        self._rotateAxis = False
        self.roll = 0.0
        self.theta = 0.0
        self.phi = 0.0

    def generate_viewport_info(self, viewport):
        yaw, pitch, radius = viewport
        orbit_state = self._orbit.copy()
        orbit_state['theta'] = yaw
        orbit_state['phi'] = pitch
        orbit_state['radius'] = radius

        # 四元数旋转（保持原有顺序）
        qRoll = Quaternion(axis=orbit_state['zAxis'], angle=orbit_state['roll'])
        qTheta = Quaternion(axis=orbit_state['yAxis'], angle=orbit_state['theta'])
        qPhi = Quaternion(axis=orbit_state['xAxis'], angle=orbit_state['phi'])

        center = orbit_state['center']
        dir_vec = -orbit_state['zAxis']
        dir_rot = qTheta.rotate(qPhi.rotate(dir_vec))
        Eye = center + dir_rot * orbit_state['radius']
        up = qRoll.rotate(orbit_state['yAxis'])

        # ----- 关键：按 JS 的 compose->invert 流程构造 viewMatrix -----
        view_row_mat, view_colmajor = build_view_matrix_via_compose(Eye, center, up)

        # ----- 构造 projection（row-major），再与 view_row 相乘得到 viewProj_row -----
        proj_row = build_projection_matrix_js(
            self.cameraParam['fov'],
            self.cameraParam['aspect'],
            self.cameraParam['znear'],
            self.cameraParam['zfar'],
            self.cameraParam['p']
        )

        viewproj_row = proj_row @ view_row_mat
        # 输出时转换为 column-major 与 Three.js.elements 对齐
        viewproj_colmajor = viewproj_row.flatten(order='F').tolist()

        fovy = self.cameraParam['fov']
        fovx = 2 * math.atan(math.tan(fovy * 0.5) * self.cameraParam['aspect'])

        return {
            "yaw": math.degrees(yaw),
            "pitch": math.degrees(pitch),
            "radius": radius,
            "train": 1,
            "shs_python": 0,
            "rot_scale_python": 0,
            "scaling_modifier": 1.0,
            "resolution_x": self.cameraParam['width'],
            "resolution_y": self.cameraParam['height'],
            "fov_y": fovy,
            "fov_x": fovx,
            "z_far": self.cameraParam['zfar'],
            "z_near": self.cameraParam['znear'],
            "keep_alive": 1,
            # 这两个数组的顺序与 Three.js 的 Matrix4.elements 完全一致（column-major）
            "view_matrix": view_colmajor,
            "view_projection_matrix": viewproj_colmajor
        }

# ---------------- 采样主循环（与你原来保持一致） ----------------

def main_sampling_loop():
    generator = ViewportGenerator()

    R_LEVELS = np.linspace(2, 6, 16, endpoint=False)
    A_LEVELS_RAD = np.radians(np.linspace(0, 360, 36, endpoint=False))
    E_LEVELS_RAD = np.radians(np.linspace(-30, 90, 10, endpoint=False))

    all_infos = []
    for r in R_LEVELS:
        for yaw in A_LEVELS_RAD:
            for pitch in E_LEVELS_RAD:
                info = generator.generate_viewport_info([yaw, pitch, r])
                all_infos.append(info)

    with open("viewports.json", "w", encoding="utf-8") as f:
        json.dump(all_infos, f, indent=2)

    print("已生成 viewports.json，共", len(all_infos), "个视点")

if __name__ == "__main__":
    main_sampling_loop()
