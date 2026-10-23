"""Modelo MuJoCo: Unitree G1 con manos (base fija) + humano cinemático (mocap).

La base se fija porque la propuesta asume un controlador de cuerpo completo que estabiliza la
locomoción; aquí se estudia la capa de seguridad del brazo derecho (7 GDL) durante el saludo.

Robot (variable de entorno PABE_ROBOT, leída al importar; los procesos hijos la heredan):
  g1     MuJoCo Menagerie, G1 29 GDL + manos Dex3 (43 GDL), el del manuscrito (por defecto)
  revo2  URDF G1 29 GDL + manos BrainCo Revo2 (51 GDL) de PABE_REVO2_URDF; mismo brazo, misma cinemática
Cada robot tiene su propio MJB en caché y su propio directorio de resultados (RESULTS_DIR).
"""
import os

import mujoco
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "mujoco_menagerie", "unitree_g1")
G1_XML = os.path.abspath(os.path.join(ROOT, "scene_with_hands.xml"))
ROBOTS = ("g1", "revo2")
ROBOT = os.environ.get("PABE_ROBOT", "g1")
if ROBOT not in ROBOTS:
    raise ValueError(f"PABE_ROBOT={ROBOT!r}: debe ser uno de {ROBOTS}")
REVO2_URDF = os.environ.get("PABE_REVO2_URDF", os.path.expanduser(
    "~/gr00t_project/3D model/g1_with_revo2/g1_with_revo2_hands.urdf"))
RESULTS_DIR = "results/g1" if ROBOT == "g1" else f"results/g1_{ROBOT}"

ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
# mano derecha: articulaciones accionadas en el agarre, su cierre y los cuerpos de la base de los dedos medio/índice
HANDS = {
    "g1": dict(joints=["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                       "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_index_0_joint",
                       "right_hand_index_1_joint"],
               closed=np.array([0.0, 0.5, -0.9, 1.1, 1.2, 1.1, 1.2]),
               finger_base=("right_hand_middle_0_link", "right_hand_index_0_link"),
               palm_normal=(0.0, 1.0, 0.0)),
    # Revo2: las falanges distales siguen a las proximales (mimic del URDF -> igualdades de MuJoCo)
    "revo2": dict(joints=["right_thumb_metacarpal_joint", "right_thumb_proximal_joint", "right_index_proximal_joint",
                          "right_middle_proximal_joint", "right_ring_proximal_joint", "right_pinky_proximal_joint"],
                  closed=np.array([0.3, 0.6, 0.9, 0.9, 0.9, 0.9]),
                  finger_base=("right_middle_proximal_link", "right_index_proximal_link"),
                  palm_normal=(0.0, 0.0, -1.0)),     # montada con giro de 90° (rpy 0 pi/2 0)
}
FINGER_JOINTS = HANDS[ROBOT]["joints"]
FINGER_CLOSED = HANDS[ROBOT]["closed"]
# normal de la palma en el marco de la muñeca (lado hacia el que se cierran los dedos; medido en el modelo)
PALM_NORMAL = np.array(HANDS[ROBOT]["palm_normal"])
# Cuerpos que delimitan las cápsulas de seguridad del robot (aproximación conservadora de los eslabones)
CAPSULES = {                       # nombre: (cuerpo_a, cuerpo_b, radio)
    "upperarm": ("right_shoulder_roll_link", "right_elbow_link", 0.050),
    "forearm": ("right_elbow_link", "right_wrist_roll_link", 0.045),
    "wrist": ("right_wrist_roll_link", "right_wrist_pitch_link", 0.042),   # sin el eslabón de la palma
}
PALM_BODY = "right_wrist_yaw_link"
HUMAN_GEOMS = ("h_torso", "h_head", "h_uarm", "h_farm", "h_hand")


def _revo2_spec():
    """G1 + Revo2 desde el URDF, preparado como el G1 de Menagerie: pelvis como cuerpo raíz sin articulación,
    valores por defecto de las articulaciones del G1 de Menagerie, actuadores de posición en todas las
    articulaciones salvo las que siguen a otra (mimic), geometrías visual/colisión en los grupos 2/3 y la
    misma escena (suelo, cielo, luz)."""
    txt = open(REVO2_URDF).read()
    # el URDF declara meshdir="meshes" y sus rutas ya empiezan por meshes/: se resuelven desde su carpeta;
    # fusestatic=false conserva la pelvis (raíz) como cuerpo para poder fijar su altura
    if 'meshdir="meshes"' not in txt:
        raise ValueError(f"{REVO2_URDF}: se esperaba <compiler meshdir=\"meshes\" .../> en la etiqueta <mujoco>")
    txt = txt.replace('meshdir="meshes"', f'meshdir="{os.path.dirname(REVO2_URDF)}" fusestatic="false"', 1)
    spec = mujoco.MjSpec.from_string(txt)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for g in spec.geoms:
        g.group = 3 if (g.contype or g.conaffinity) else 2
    for j in spec.joints:
        j.armature, j.frictionloss = 0.01, 0.3
    followers = {e.name1 for e in spec.equalities if e.type == mujoco.mjtEq.mjEQ_JOINT}
    for j in spec.joints:
        if j.name not in followers:
            a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=500, dampratio=1, inheritrange=True)
    spec.add_texture(type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
                     markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    mat = spec.add_material(name="groundplane", texuniform=True, texrepeat=[5, 5], reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="groundplane")
    spec.worldbody.add_light(pos=[1, 0, 3.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    v = spec.visual
    v.headlight.diffuse, v.headlight.ambient, v.headlight.specular = [0.6] * 3, [0.1] * 3, [0.9] * 3
    v.rgba.haze = [0.15, 0.25, 0.35, 1]
    v.global_.azimuth, v.global_.elevation = 140, -20
    return spec


def _local_pos(spec, body, frame):
    """Posición del origen de `body` en el marco del cuerpo `frame` (posición de referencia del modelo)."""
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    b, f = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in (body, frame))
    return d.xmat[f].reshape(3, 3).T @ (d.xpos[b] - d.xpos[f])


def build(human=True, walk=False):
    """walk=True: variante de caminata (sólo visualización, ver walk_in.py): pelvis mocap y piernas humanas
    separadas para animar la marcha. No cambia nq, así que los registros de episodios se reproducen tal cual."""
    if ROBOT == "g1":
        spec = mujoco.MjSpec.from_file(G1_XML)
        pelvis = spec.body("pelvis")
        for j in list(pelvis.joints):
            spec.delete(j)
        for k in list(spec.keys):
            spec.delete(k)
    else:
        spec = _revo2_spec()
        pelvis = spec.body("pelvis")
    pelvis.pos = [0.0, 0.0, 0.793]
    if walk:
        pelvis.mocap = True
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    # base fija y humano cinemático: se desactivan los contactos (evita el roce brazo-torso del modelo de pie)
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    # centro de la palma: 65 % del camino entre la muñeca y la base de los dedos medio/índice
    wy = spec.body(PALM_BODY)
    fb = HANDS[ROBOT]["finger_base"]
    if ROBOT == "g1":      # bases de los dedos hijas directas de la muñeca
        base = 0.5 * (np.array(spec.body(fb[0]).pos) + np.array(spec.body(fb[1]).pos))
    else:                  # Revo2: montada en right_base_link (articulación fija con giro)
        base = 0.5 * (_local_pos(spec, fb[0], PALM_BODY) + _local_pos(spec, fb[1], PALM_BODY))
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
        pants = [0.30, 0.32, 0.38, 1.0]
        mocap("h_legs", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.13, 0.40, 0], [*pants[:3], 0.0] if walk else pants)
        if walk:
            mocap("h_rleg", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.065, 0.37, 0], pants)
            mocap("h_lleg", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.065, 0.37, 0], pants)
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
        self.torso = B("torso_link")
        self.palm_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "palm")
        self.mocap = {n: m.body_mocapid[B(n)] for n in HUMAN_GEOMS + ("h_legs",)}
        self.all_act = np.arange(m.nu)
        self.act_q = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])


def check_results_dir(out):
    """Impide mezclar resultados (y puntos de control) de robots distintos en un mismo directorio."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "robot.txt")
    if os.path.exists(path):
        prev = open(path).read().strip()
        if prev != ROBOT:
            raise RuntimeError(f"{out} contiene resultados de PABE_ROBOT={prev}, no de {ROBOT}; usa --out {RESULTS_DIR}")
    else:
        with open(path, "w") as f:
            f.write(ROBOT + "\n")


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
