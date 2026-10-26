# PABe-CBF: extensions, results and limits

Oct 22, 2026 · Eduardo Tijerina

> Exported from the Claude Doc https://claude.ai/artifact/SNwBr7ZW4vA6te5m9HyF7P (Oct 27, 2026, with the Isaac Sim
> results); the hand-off diagram is described in text.

## Summary

On top of the PABe-CBF repository we fixed two arm problems, added the G1 with Revo2 hands (51 DOF), and split the problem into two systems: one for approaching on foot and one for hesitation during the handshake. The results come from MuJoCo, with a port to Isaac Sim checked against it; the full 90-minute pipeline has not been run yet, so the figures come from limited test runs with default thresholds.

What was done, in order:

1. Local reproduction in `/mnt/data/CBF-repro` (Python 3.13, MuJoCo 3.16) and a B1 smoke test.
2. Arm–torso self-collision barrier: the arm passed 5–10 cm through the robot's chest in every family.
3. Palm orientation toward the person's palm (thumb-up handshake).
4. Robot switch `PABE_ROBOT=revo2` using the 51-DOF URDF from the GR00T project.
5. New family `x_walk_hesitation`: both walk toward each other, with six random hesitation variants.
6. Visible hesitation in `iv_hesitation` (the hand pulls back 15 cm instead of 2 cm).
7. A new proxemic feature (the person's own approach speed) and Hall-zone metrics.
8. Two systems with their own beliefs: approach (P0–P3) and handshake (PABe-CBF), joined by a hand-off.

Overall reading: the PABe-style approach policy (P3) is the best of the four (97% hand-offs, 3% false aborts, 83% of departures detected). The arm fixes remove the chest penetration and the misoriented palm without losing handshakes in the stable cases, at the cost of some escape room in lunges and withdrawals.

## Controller fixes

**Arm–torso self-collision.** The QP only had robot–person barriers and joint limits, and the model disables contacts; during the approach the right arm swept through the chest. We added barriers between the arm (distal half of the upper arm, forearm, wrist, palm) and two capsules fitted to the torso mesh, always on and for every filtered method. B0 has no filter and still passes through it.

| Family (B1, 3 participants) | Penetration before | After | Other effect |
| --- | --- | --- | --- |
| iv_hesitation | 10.3 cm | 0 cm | success 0.67 → 1.00 |
| vii_lunge | 6.1 cm | 0 cm | success 0.67 → 0.33 |
| iii_withdraw_contact | 5.6 cm | 0 cm | violations 0.67 → 1.00 |
| other families | 5–7 cm | 0 cm | no noticeable change |

The arm has less room to escape in lunges and withdrawals. Self-collision is left out of the contact-entry check; otherwise B0 never started the handshake.

**Palm orientation.** The nominal controller only aimed the hand's axis; the roll about that axis stayed at its rest value. We added just that roll component so the palm faces the person's palm, and the person's hand is drawn palm to palm.

| Robot | Palm error before | After | Handshakes completed |
| --- | --- | --- | --- |
| G1 (Dex3) | ~45° | 17° | 100% |
| G1 (Revo2) | ~115° | 23° | 100% |

We tried and rejected also making the hands antiparallel (4° error): completed handshakes fell to 46% (G1) and 19% (Revo2), because the meeting point is near the midline and the self-collision barrier blocks the grasp.

## G1 + Revo2 robot (51 DOF)

`PABE_ROBOT=revo2` swaps the MuJoCo Menagerie G1 (Dex3 hands, 43 DOF) for the URDF from the GR00T project (BrainCo Revo2 hands, 51 DOF); without the variable the original is used. The right arm has the same names, kinematics and limits, and the torso is the same mesh, so the barriers apply as they are.

- The URDF is loaded unmodified; its doubled mesh path (`meshes/meshes`) is fixed in memory.
- What the URDF lacks is added: position actuators (kp = 500), the Menagerie G1's armature and friction, floor, sky and light.
- The 10 mimic finger joints become equality constraints; the grasp closes 6 joints.
- Each robot has its own cached compiled model and its own results folder (`results/g1`, `results/g1_revo2`); mixing them is an error.

Verification: with the default robot, the recompiled model is byte-identical and 60 B0/B1 episodes are bit-identical. The Revo2 completes all 10 families with no torso penetration, with figures close to the G1 (i_stable: 100% handshakes, 10.4 N vs 9.9 N).

## New and modified scenarios

**Walking (`x_walk_hesitation`).** Robot and person start about 3.2 m apart and walk toward each other; on arrival, the handshake follows the stable script. In the physics the robot has a fixed base: the person's body is moved in the robot's frame and the legs are animation only. Each episode draws one variant:

| Variant | What the person does | Label | Handshake |
| --- | --- | --- | --- |
| stop_back | stops, half step back (5–12 cm), continues | H | yes |
| slowdown | slows to 25–40% without stopping | H | yes |
| double_pause | two short stops | H on each | yes |
| false_alarm | brief stop, no step back | E | yes (cancelling is a false cancel) |
| abandon | stops, hesitates and backs away 0.6–1.0 m | H then W | not authorized |
| arrival_delay | arrives and waits 0.8–1.6 s before offering a hand | H | yes, late |

**Arm hesitation (`iv_hesitation`, 3D G1 only).** In the original the hesitation was almost invisible. Now, halfway through the reach, the hand pulls back 15 cm toward the body during a pause 1.5 times longer, and in contact the tug grows from 1.2 to 2 cm. That tug can't grow further: above ~2.7 cm the grip spring exceeds the 8 N explicit-pull rule and every hesitation would end in a release. Measured cost: one handshake out of 8 with Revo2 and B1. The 2D planar benchmark keeps the original amplitudes.

## Two systems: approach and handshake

The approach and the hesitation in the handshake are now two systems, each with its own belief, training and benchmark. During the walk the handshake system waits: the arm stays at rest and the safety filter stays on; its HMM starts fresh at the hand-off.

*Diagram (two systems · hand-off and abort):* the approach system (its own HMM on disposition and approach speed, policy P0–P3) commands walk / slow / stop / abort. When the robot is at its station and the torso distance is at most ~1.08 m it hands off to the handshake system (PABe-CBF, HMM on the first three features, reset at hand-off); an abort ends the episode without a handshake.

The new feature y[3] is the speed at which the person approaches, with the robot's own walking removed; it only exists while they are more than ~1 m away. The approach HMM learned 0.46 m/s (engaged), 0.04 (hesitant) and −0.02 (withdrawing).

**Benchmark A, approach system** (120 episodes per policy; hand-off and false aborts over people who do arrive):

| Policy | Hand-off | Time to hand-off (s) | False aborts | Departures detected | Speed while hesitating (m/s) | Waiting (s) |
| --- | --- | --- | --- | --- | --- | --- |
| P3 PABe | 97% | 4.18 | 3% | 83% | 0.45 | 0.11 |
| P2 entropy | 95% | 4.39 | 5% | 67% | 0.38 | 0.13 |
| P1 MAP | 61% | 5.37 | 10% | 75% | 0.24 | 3.60 |
| P0 blind | 100% | 4.16 | 0% | 0% | 0.48 | 0.01 |

**Chained, approach then handshake** (72 episodes each, no violations in any):

| Combination | Hand-off | Handshake completed | False abort | False cancel | Departures detected |
| --- | --- | --- | --- | --- | --- |
| P3 → B1 | 98% | 98% | 2% | 0% | 88% |
| P3 → B9 | 98% | 91% | 2% | 8% | 88% |
| P0 → B1 | 100% | 100% | 0% | 0% | 0% |
| P0 → B9 | 100% | 91% | 0% | 9% | 0% |

## Isaac Sim port

On Isaac Sim 5.1 (PhysX) the same episodes end the same way as in MuJoCo in 20 of 24 cases; the 4 that differ are borderline cases near a release threshold.

The port is a separate project ([PABe-Isaac](https://github.com/EduardoTijerina/PABe-Isaac)). The episode loop talks to a small simulator interface with two backends. The MuJoCo backend reproduces this repository bit for bit (40/40 Revo2 and 80/80 G1 episodes, B0, B1, B2 and B9). The Isaac backend loads the same G1 + Revo2 URDF, converted to USD, with a fixed base and torque control of the arm.

**MuJoCo vs Isaac** (G1 + Revo2, B0 and B1, participants 4000–4002, one episode each; the walking family uses the blind approach P0):

| Family | Episodes | Same outcome and violation flag | What differs |
| --- | --- | --- | --- |
| i_stable | 6 | 6 | — |
| x_walk_hesitation | 6 | 6 | — |
| iii_withdraw_contact | 6 | 5 | B1, one release: peak force 20.1 N in MuJoCo vs 19.6 N in Isaac, on either side of the 20 N limit |
| iv_hesitation (amplified) | 6 | 3 | B1 only: MuJoCo completes 2 and times out 1; Isaac releases 2 on a pull and times out 1 |

Where outcomes agree, peak force differs by 0.5 N (median; max 3.7 N) and episode length by 0.03 s (max 0.53 s); the palm path differs by 0.5 cm (median over episodes). Isaac takes 35.7 s per episode against 1.1 s in MuJoCo.

The likely cause of the borderline cases is joint friction: MuJoCo's G1 has 0.3 N·m of frictionloss per joint and PhysX has no equivalent. Not confirmed yet. Four other PhysX differences were found with probes and corrected: Jacobians at the link's centre of mass, a default mass (+14 kg) on links without inertia, drive gains reloaded from the USD on every reset, and a mass matrix without armature.

**Walking demo** (participant 4006, stop_back, B1 with approach P3): the approach behaves the same, with hand-off at 4.52 s in both and contact at 7.67 s (MuJoCo) vs 7.54 s (Isaac). MuJoCo completes the handshake at 11.2 s; in Isaac the hands stay together without completing and the episode times out at 16.5 s. Not explained yet. The Isaac window replays episodes in real time, with the same kinematic leg animation as the MuJoCo viewer.

## Conclusions

The changes fix real problems in the model and extend the benchmark to approaching on foot; the results are preliminary.

- **Holds:** the paper's core idea carries over to the approach. Deciding on calibrated risk (P3) detects 83% of people who leave with only 3% false aborts, better than MAP (P1) and entropy (P2).
- **Holds:** the self-collision barrier and the palm roll remove the chest penetration and the misoriented palm without losing handshakes in the stable cases, on both robots.
- **Holds with caveats:** the hand-off leaves the handshake system untouched (B9 cancels 8–9% with P0 and with P3 alike); the approach's benefit is almost entirely detecting departures.
- **Does not hold (yet):** the personal-space buffer does not reduce intrusion (~0.1 s in every chained run); it comes from `arrival_delay`, when the person has already arrived and hesitates.
- **Cost:** with the barrier, the arm escapes worse in lunges and withdrawals; the step back in `stop_back` causes most false aborts (P3: 14%); the amplified hesitation costs 1 of 8 handshakes with Revo2.

## Limitations and next steps

The figures in this report are not comparable with the original paper's: they change the controller of every method and the definition of `iv_hesitation`.

- **Pipeline not run:** the handshake threshold validation and the 360 test runs per method (~90 min per robot) are still missing. Everything above uses default thresholds and samples of 3 to 120 episodes.
- **Approach thresholds:** P2 and P3 use default values; they still need to be chosen in validation, as the handshake's are.
- **Kinematic walking:** the robot doesn't balance or walk physically; in `abandon` the person backs away facing the robot because the capsule model can't turn.
- **Hand-off:** it happens when both arrive; if the person hesitates once at their spot, the intrusion is left to the handshake system. Holding the hand-off until the person shows readiness is a small pending change.
- **Revo2 near the midline:** some grasps stop 2 mm short of grasp distance because of the self-collision barrier; moving the meeting point a few centimetres to the robot's right would ease it.
- **Isaac Sim:** runs about 30 times slower than MuJoCo, so the full benchmark there needs Isaac Lab with many robots in parallel (not started); joint friction and the unfinished walking handshake are open.
- **Toward a real G1:** the paper's limits still apply (scripted human, synthetic observations, force as a spring, assumed bounds).
