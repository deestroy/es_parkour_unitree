# Trial log — teacher gait & parkour iterations

Each folder holds the rollout GIFs for one training trial. All results are honest: corridor-enforced
(the course cannot be bypassed), deterministic policy unless noted. GIFs show the event-camera view
picture-in-picture where recorded (standard from trial 06 onward).

## 00-baselines
Random policy and the first from-scratch walkers (hop gait, pre-corridor). Historical reference —
success numbers from this era were inflated by the walk-around-the-obstacles exploit.

## 01-kl-stabilized — `teacher_kl`
**Change:** KL-adaptive learning rate in PPO (cured the reproducible ~1.2M-step policy collapse);
best-checkpoint gate armed early.
**Result @ d0.1:** flat 100% success; hurdle 85%, step 78%, parkour 72% progress. First stable
long run; obstacle skills shallow.

## 02-hard-terrain — `teacher_hard`
**Change:** difficulty 0.4 terrain (28 cm gaps, 17 cm hurdles), calmer 1.5 Hz gait clock, stronger
action smoothing.
**Result:** deterministic policy degenerated (noise-reliant); stochastic flat 100%, step 40%.
Diagnosis: harder terrain suppressed mean-policy refinement.

## 03-anti-drag — `teacher_clear`
**Change:** swing-phase foot-clearance reward (rear feet had been dragging: a 1 mm skim scored as a
swing).
**Result @ d0.4 determ:** flat 100%, step 31%, hurdle 27%; rear-drag 47%→~44% determ (28% stoch).
Deterministic policy restored.

## 04-natural-gait — `teacher_dog3`
**Change:** trunk-height tracking (anti-crumple), front/rear torque balance, posture
regularization, doubled smoothing — from visual review: jittery overworked fronts, underused rears.
**Result @ d0.4 determ:** flat 100%, step 62%, parkour 45%; rear-drag 19–29%; trunk height
0.325 m ± 1.2 cm; front/rear torque 9.6/8.0.

## 05-leap-zone — `teacher_dog4`
**Change:** leap zone near gaps/hurdles (anti-flight penalties + gait clock off), obstacle-height
clearance target, +2 milestone per obstacle, 2.5 s anti-stall termination, snag penalty for
non-foot contacts.
**Result @ d0.4 determ:** step 87% (climbs all six risers), parkour 52%, flat 100%; stalls 0%.
Gap 16% / hurdle 20% — flight legal but not yet discovered.

## 06-leap-discovery (in progress) — Trial A `teacher_dog5` vs Trial B `teacher_leap`
**Trial A (shaping):** takeoff reward (upward velocity paid in the leap zone), all-feet
obstacle-height clearance (the trot clock no longer fights lifting a leg pair), left/right torque
balance (measured rear-left loafing at 0.87× rear-right).
**Trial B (specialist):** same shaping, but gap+hurdle-only world at difficulty 0.5,
transfer-initialized from dog4 so all learning budget goes to the leap skill (the paper lineage's
`a1_leap` approach).

## students
SNN student rollouts (spiking ResNet-18 + GRU + spiking MLP driven by simulated event-camera
input; inset shows the event stream). Distilled via Eq. 8/9 + DAGGER; BPTT sequence-memory
version in progress.
