"""Modelo MuJoCo: Unitree G1 con manos (MuJoCo Menagerie, base fija) + humano cinemático (mocap).

La base se fija porque la propuesta asume un controlador de cuerpo completo que estabiliza la
locomoción; aquí se estudia la capa de seguridad del brazo derecho (7 GDL) durante el saludo.
"""
import os

import mujoco
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "mujoco_menagerie", "unitree_g1")
G1_XML = os.path.abspath(os.path.join(ROOT, "scene_with_hands.xml"))

ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
FINGER_JOINTS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                 "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_index_0_joint",
                 "right_hand_index_1_joint"]
FINGER_CLOSED = np.array([0.0, 0.5, -0.9, 1.1, 1.2, 1.1, 1.2])
# Cuerpos que delimitan las cápsulas de seguridad del robot (aproximación conservadora de los eslabones)
CAPSULES = {                       # nombre: (cuerpo_a, cuerpo_b, radio)
    "upperarm": ("right_shoulder_roll_link", "right_elbow_link", 0.050),
    "forearm": ("right_elbow_link", "right_wrist_roll_link", 0.045),
    "wrist": ("right_wrist_roll_link", "right_wrist_pitch_link", 0.042),   # sin el eslabón de la palma
}
PALM_BODY = "right_wrist_yaw_link"
HUMAN_GEOMS = ("h_torso", "h_head", "h_uarm", "h_farm", "h_hand")


def build(human=True):
    spec = mujoco.MjSpec.from_file(G1_XML)
    pelvis = spec.body("pelvis")
    for j in list(pelvis.joints):
        spec.delete(j)
    for k in list(spec.keys):
        spec.delete(k)
    pelvis.pos = [0.0, 0.0, 0.793]
    # base fija y humano cinemático: se desactivan los contactos (evita el roce brazo-torso del modelo de pie)
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    # centro de la palma: 65 % del camino entre la muñeca y la base de los dedos medio/índice
    wy = spec.body(PALM_BODY)
    base = 0.5 * (np.array(spec.body("right_hand_middle_0_link").pos) + np.array(spec.body("right_hand_index_0_link").pos))
    wy.add_site(name="palm", pos=list(0.65 * base), size=[0.012, 0, 0], rgba=[0.16, 0.47, 0.84, 1], group=2)
    if human:
        wb = spec.worldbody
        col = [0.82, 0.70, 0.60, 1.0]
        shirt = [0.55, 0.62, 0.72, 1.0]

        def mocap(name, gtype, size, rgba):
            b = wb.add_body(name=name, mocap=True)
            b.add_geom(name=name, type=gtype, size=size, rgba=rgba, contype=0, conaffinity=0, group=1)
            return b
        mocap("h_torso", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.16, 0.24, 0], shirt)
        mocap("h_head", mujoco.mjtGeom.mjGEOM_SPHERE, [0.10, 0, 0], col)
        mocap("h_uarm", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.050, 0.15, 0], shirt)
        mocap("h_farm", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.040, 0.135, 0], col)
        mocap("h_hand", mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.045, 0.025, 0.06], col)
        mocap("h_legs", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.13, 0.40, 0], [0.30, 0.32, 0.38, 1.0])
    # brazo derecho con actuadores de par (control por par calculado, como en un WBC); límites de par del G1
    for name in ARM_JOINTS:
        act = spec.actuator(name)
        act.gainprm[:3] = [1.0, 0.0, 0.0]
        act.biasprm[:3] = [0.0, 0.0, 0.0]
        lim = 5.0 if ("wrist_pitch" in name or "wrist_yaw" in name) else 25.0
        act.ctrlrange = [-lim, lim]
        act.ctrllimited = 1
    m = spec.compile()
    return m


class G1Index:
    """Índices útiles del modelo compilado."""

    def __init__(self, m):
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        A = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.arm_q = np.array([m.jnt_qposadr[J(n)] for n in ARM_JOINTS])
        self.arm_v = np.array([m.jnt_dofadr[J(n)] for n in ARM_JOINTS])
        self.arm_act = np.array([A(n) for n in ARM_JOINTS])
        self.arm_lo = np.array([m.jnt_range[J(n)][0] for n in ARM_JOINTS])
        self.arm_hi = np.array([m.jnt_range[J(n)][1] for n in ARM_JOINTS])
        self.fing_act = np.array([A(n) for n in FINGER_JOINTS])
        self.cap_bodies = {k: (B(a), B(b), r) for k, (a, b, r) in CAPSULES.items()}
        self.palm = B(PALM_BODY)
        self.palm_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "palm")
        self.mocap = {n: m.body_mocapid[B(n)] for n in HUMAN_GEOMS + ("h_legs",)}
        self.all_act = np.arange(m.nu)
        self.act_q = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])


def stand_qpos(m):
    """Postura de pie: articulaciones a cero salvo rodilla/cadera levemente flexionadas y brazos relajados."""
    q = np.zeros(m.nq)
    for n, v in [("left_hip_pitch_joint", -0.1), ("right_hip_pitch_joint", -0.1), ("left_knee_joint", 0.3),
                 ("right_knee_joint", 0.3), ("left_ankle_pitch_joint", -0.2), ("right_ankle_pitch_joint", -0.2),
                 ("left_shoulder_roll_joint", 0.25), ("right_shoulder_roll_joint", -0.30),
                 ("left_elbow_joint", 0.9), ("right_elbow_joint", 0.9), ("right_shoulder_pitch_joint", 0.25),
                 ("left_shoulder_pitch_joint", 0.25)]:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        q[m.jnt_qposadr[j]] = v
    return q
