# Software y modelos de terceros

Este repositorio se publica bajo licencia MIT (ver `LICENSE`). Usa, sin redistribuirlos, los componentes siguientes:

| Componente | Uso | Licencia | Cómo se obtiene |
| --- | --- | --- | --- |
| [MuJoCo Menagerie: Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) | Modelo del robot (mallas, cinemática, inercias) | BSD-3-Clause, © Unitree Robotics | `scripts/fetch_g1_model.sh` |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Simulador físico | Apache-2.0 | `pip install mujoco` |
| [quadprog](https://github.com/quadprog/quadprog) | Solver del QP | GPL-2.0-or-later | `pip install quadprog` |
| NumPy, SciPy, Matplotlib, Pillow | Cálculo, figuras y videos | BSD / PSF / HPND | `pip` |

Las figuras y videos de `docs/` y `media/` muestran renders del modelo del G1 de MuJoCo Menagerie.

**Nota sobre quadprog (GPL):** el código de este repositorio es MIT e importa quadprog como dependencia instalada por el usuario. Si se redistribuye un binario que incluya quadprog, aplican los términos de la GPL. Para evitarlo, el solver puede sustituirse por uno con licencia permisiva (p. ej. OSQP, Apache-2.0) en `pabe_cbf/core.py:_solve`.
