#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate viewports_20260105-style camera samples for the per-view g(q) pipeline."""

import argparse
import json
import math

import numpy as np
from pyquaternion import Quaternion


def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def build_projection_matrix_js(fov_radian, ratio, z_near, z_far, principal_point):
    y_scale = 1.0 / math.tan(fov_radian / 2.0)
    x_scale = y_scale / ratio
    dx = 2.0 * principal_point["x"] - 1.0
    dy = 2.0 * principal_point["y"] - 1.0

    m = np.zeros((4, 4), dtype=float)
    m[0, 0] = x_scale
    m[0, 2] = dx
    m[1, 1] = y_scale
    m[1, 2] = dy
    m[2, 2] = (z_near + z_far) / (z_near - z_far)
    m[2, 3] = (2 * z_near * z_far) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


def build_view_matrix_via_compose(eye, center, up):
    eye = np.array(eye, dtype=float)
    center = np.array(center, dtype=float)
    up = np.array(up, dtype=float)

    z_axis = normalize(eye - center)
    x_axis = normalize(np.cross(normalize(up), z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    rotation = np.eye(4, dtype=float)
    rotation[0, 0], rotation[0, 1], rotation[0, 2] = x_axis[0], y_axis[0], z_axis[0]
    rotation[1, 0], rotation[1, 1], rotation[1, 2] = x_axis[1], y_axis[1], z_axis[1]
    rotation[2, 0], rotation[2, 1], rotation[2, 2] = x_axis[2], y_axis[2], z_axis[2]

    translation = np.eye(4, dtype=float)
    translation[0, 3] = eye[0]
    translation[1, 3] = eye[1]
    translation[2, 3] = eye[2]

    # model: camera-to-world (columns = camera right/up/forward in world, last col = eye)
    model = translation @ rotation
    view_row = np.linalg.inv(model)
    view_colmajor_list = view_row.flatten(order="F").tolist()
    # extrinsicMatrix stored row-major (matches JS viewer convention)
    extrinsic_rowmajor_list = model.flatten(order="C").tolist()
    return view_row, view_colmajor_list, extrinsic_rowmajor_list


class ViewportGenerator:
    def __init__(self):
        # Matches the JS viewer orbit settings used during real trace recording:
        #   center: Vector3(0,0,0)  xAxis: (1,0,0)  yAxis: (0,-1,0)  zAxis: (0,0,1)
        # yAxis pointing down (-Y) matches the 3DGS/WebGL camera convention.
        self._orbit = {
            "radius": 4.0,
            "theta": 0.0,
            "phi": 0.0,
            "roll": 0.0,
            "center": np.array([0.0, 0.0, 0.0]),
            "xAxis": np.array([1.0, 0.0, 0.0]),
            "yAxis": np.array([0.0, -1.0, 0.0]),
            "zAxis": np.array([0.0, 0.0, 1.0]),
            "move": np.array([0.0, 0.0, 0.0]),
        }

        self.camera_param = {
            "width": 1920,
            "height": 1080,
            "fov": 60.0 / 180.0 * math.pi,
            "aspect": 1920.0 / 1080.0,
            "znear": 0.009,
            "zfar": 1100,
            "p": {"x": 0.5, "y": 0.5},
        }

    def generate_viewport_info(self, viewport):
        yaw, pitch, radius = viewport
        orbit_state = self._orbit.copy()
        orbit_state["theta"] = yaw
        orbit_state["phi"] = pitch
        orbit_state["radius"] = radius

        q_roll = Quaternion(axis=orbit_state["zAxis"], angle=orbit_state["roll"])
        q_theta = Quaternion(axis=orbit_state["yAxis"], angle=orbit_state["theta"])
        q_phi = Quaternion(axis=orbit_state["xAxis"], angle=orbit_state["phi"])

        center = orbit_state["center"]
        dir_vec = -orbit_state["zAxis"]
        dir_rot = q_theta.rotate(q_phi.rotate(dir_vec))
        eye = center + dir_rot * orbit_state["radius"]
        up = q_roll.rotate(orbit_state["yAxis"])

        view_row_mat, view_colmajor, extrinsic_rowmajor = build_view_matrix_via_compose(eye, center, up)
        proj_row = build_projection_matrix_js(
            self.camera_param["fov"],
            self.camera_param["aspect"],
            self.camera_param["znear"],
            self.camera_param["zfar"],
            self.camera_param["p"],
        )
        viewproj_row = proj_row @ view_row_mat
        viewproj_colmajor = viewproj_row.flatten(order="F").tolist()

        fov_y = self.camera_param["fov"]
        fov_x = 2 * math.atan(math.tan(fov_y * 0.5) * self.camera_param["aspect"])

        return {
            "yaw": math.degrees(yaw),
            "pitch": math.degrees(pitch),
            "radius": radius,
            "position": eye.tolist(),
            "train": 1,
            "shs_python": 0,
            "rot_scale_python": 0,
            "scaling_modifier": 1.0,
            "resolution_x": self.camera_param["width"],
            "resolution_y": self.camera_param["height"],
            "fov_y": fov_y,
            "fov_x": fov_x,
            "z_far": self.camera_param["zfar"],
            "z_near": self.camera_param["znear"],
            "keep_alive": 1,
            "view_matrix": view_colmajor,
            "view_projection_matrix": viewproj_colmajor,
            "extrinsicMatrix": extrinsic_rowmajor,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate viewports_20260105-compatible viewport JSON.")
    parser.add_argument("--output_json", type=str, default="viewports_20260105.json")
    parser.add_argument("--num_r", type=int, default=32)
    parser.add_argument("--num_yaw", type=int, default=32)
    parser.add_argument("--num_pitch", type=int, default=32)
    parser.add_argument("--r_min", type=float, default=2.0)
    parser.add_argument("--r_max", type=float, default=8.0)
    args = parser.parse_args()

    generator = ViewportGenerator()

    r_levels = np.linspace(args.r_min, args.r_max, args.num_r, endpoint=False)
    yaw_levels_rad = np.radians(np.linspace(0, 360, args.num_yaw, endpoint=False))
    pitch_levels_rad = np.radians(np.linspace(-90, 90, args.num_pitch, endpoint=False))

    all_infos = []
    for radius in r_levels:
        for yaw in yaw_levels_rad:
            for pitch in pitch_levels_rad:
                all_infos.append(generator.generate_viewport_info([yaw, pitch, radius]))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_infos, f, indent=2)

    print(f"Generated {args.output_json} with {len(all_infos)} viewports.")


if __name__ == "__main__":
    main()
