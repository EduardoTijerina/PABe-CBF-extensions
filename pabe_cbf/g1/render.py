"""Renderizado MuJoCo (EGL) de episodios registrados: fotogramas para figuras y vídeo MP4."""
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from .. import params as P
from . import geometry as G
from .episode import get_model, PH_OFF
from .model import G1Index


def _camera(view="side"):
    cam = mujoco.MjvCamera()
    if view == "side":
        cam.lookat[:] = [0.42, 0.02, 0.95]
        cam.distance, cam.azimuth, cam.elevation = 1.55, -62.0, -12.0
    else:
        cam.lookat[:] = [0.40, 0.0, 0.98]
        cam.distance, cam.azimuth, cam.elevation = 1.35, -95.0, -20.0
    return cam


def _add_sphere(scn, pos, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def frames(record, idx, width=640, height=480, view="side", envelope=True):
    """Devuelve una lista de imágenes RGB para los índices pedidos de un episodio registrado."""
    mm = get_model()
    d = mujoco.MjData(mm)
    ix = G1Index(mm)
    from .episode import G1Episode
    r = mujoco.Renderer(mm, height, width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    imgs = []
    try:
        for i in idx:
            rec = record[min(i, len(record) - 1)]
            d.qpos[:] = rec["qpos"]
            G1Episode._set_mocap(None, mm, d, ix, rec["pose"])
            mujoco.mj_forward(mm, d)
            r.update_scene(d, camera=_camera(view), scene_option=opt)
            if envelope and rec["mode"] in ("A", "R") and not rec["engaged"]:
                rad = P.D_SEP_PALM + PH_OFF + P.EPS_D + rec["theta"][0]
                _add_sphere(r.scene, rec["hand"], rad, [0.16, 0.47, 0.84, 0.18])
            if rec["engaged"]:
                _add_sphere(r.scene, rec["hand"], 0.05, [0.92, 0.41, 0.20, 0.35])
            imgs.append(r.render().copy())
    finally:
        r.close()
    return imgs


def video(records, labels, path, fps=30, width=480, height=360, step=3):
    """Vídeo lado a lado (un panel por registro) con ffmpeg."""
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image, ImageDraw
    n = min(len(r) for r in records)
    idx = list(range(0, n, step))
    panels = [frames(r, idx, width, height) for r in records]
    tmp = path + "_frames"
    os.makedirs(tmp, exist_ok=True)
    for k, i in enumerate(idx):
        row = np.concatenate([p[k] for p in panels], axis=1)
        im = Image.fromarray(row)
        dr = ImageDraw.Draw(im)
        for j, (lab, rec) in enumerate(zip(labels, records)):
            x = rec[i]
            dr.text((j * width + 8, 6), f"{lab}  t={x['t']:.2f}s  modo {x['mode']}  |F|={abs(x['F']):.1f}N", fill=(20, 20, 20))
        im.save(os.path.join(tmp, f"{k:05d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", path], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
