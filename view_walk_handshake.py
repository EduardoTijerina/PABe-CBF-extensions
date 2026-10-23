"""Robot and person walk toward each other, then shake hands.
x_walk_hesitation (default): the approach and the mid-walk hesitation are part of the physical episode.
Any other family: a kinematic walk-in prologue is played before the unchanged episode.
Leg motion is always a kinematic animation (fixed-base simulation).
Usage: python view_walk_handshake.py [--method B1] [--family x_walk_hesitation] [--pid 5] [--ep 0]
       [--variant slowdown] [--models results/g1/models.pkl] [--speed 1.0] [--video out.mp4]"""
import argparse
import subprocess
import time

import mujoco
import numpy as np

from pabe_cbf.core import METHODS
from pabe_cbf.g1.episode import G1Episode
from pabe_cbf.g1.geometry import G1_GEOM
from pabe_cbf.g1.walk_in import WalkHandshake
from pabe_cbf.scenarios import Participant, Scenario

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="B1")
ap.add_argument("--family", default="x_walk_hesitation")
ap.add_argument("--pid", type=int, default=5)
ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--models", default=None)
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--video", default=None, help="render an MP4 instead of opening the viewer")
ap.add_argument("--approach", default=None, help="approach-system policy P0-P3 (default: P3 with --models, else P0)")
ap.add_argument("--variant", default=None, help="x_walk_hesitation only: pick the first --ep with this walk variant "
                "(stop_back, slowdown, double_pause, false_alarm, abandon, arrival_delay)")
a = ap.parse_args()

models = None
if a.models:
    from pabe_cbf.g1.train_g1 import load
    models = load(a.models)

if a.variant:   # first episode index of this participant whose random walk variant matches
    a.ep = next(e for e in range(a.ep, a.ep + 500)
                if Scenario(a.family, Participant(a.pid, geom=G1_GEOM), e).walk.variant == a.variant)
sc = Scenario(a.family, Participant(a.pid, geom=G1_GEOM), a.ep)
ep = G1Episode(METHODS[a.method], sc, models, record=True, approach=a.approach)
res, _, _ = ep.run()
print(f"{a.method} / {a.family} ep={a.ep} variant={res['variant']} approach={res['approach']} "
      f"handoff={res['t_handoff']} abort={res['t_abort']}: reason={res['R_reason']} "
      f"peak_F={res['peak_F']:.1f}N violation={res['V_any']}", flush=True)

wh = WalkHandshake(ep.record, sc)
d = mujoco.MjData(wh.mm)


def set_cam(cam):
    cam.lookat[:] = [0.4, 0.0, 0.85]
    cam.distance, cam.azimuth, cam.elevation = 3.6, 90.0, -12.0


if a.video:
    W, H, step = 960, 540, 3
    r = mujoco.Renderer(wh.mm, H, W)
    cam = mujoco.MjvCamera()
    set_cam(cam)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", f"{100 / step / a.speed:.3f}", "-i", "-",
                           "-pix_fmt", "yuv420p", a.video], stdin=subprocess.PIPE)
    for i in range(0, wh.n, step):
        wh.apply(d, i)
        r.update_scene(d, camera=cam)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close()
    ff.wait()
    r.close()
    print("wrote", a.video)
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(wh.mm, d) as v:
        set_cam(v.cam)
        while v.is_running():
            t0 = time.time()
            for i in range(wh.n):
                if not v.is_running():
                    break
                wh.apply(d, i)
                v.sync()
                wait = wh.time(i) / a.speed - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
            time.sleep(1.0)
