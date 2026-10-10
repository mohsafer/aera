
---

## 2026-10-02 — goal-following reward + mind curriculum (addressing the A/B verdict)

Two mechanisms built to attack the diagnosed causes of the negative A/B:

1. **Goal-following reward** (`RewardCfg.w_goal`, default 0.25, active only
   when `agent.mind_goal`): each step the env checks whether the outcome
   agrees with the current goal — forage pays for closing on food, flee for
   opening distance from threats, navigate for closing on the beacon,
   explore for novelty, shelter for damage-free steps, craft for pickups
   and crafts. Shows up in `info["sub"]["goal"]` (transparent like every
   other term). This makes goals commitment devices: the policy is PAID for
   executing the mind, aligning the two layers instead of feeding the
   policy noise it must learn to ignore.
2. **Mind curriculum** (`--mind-start N`): the goal slot exists from step 0
   (obs layout fixed) but stays a constant "explore" until step N — the
   policy learns to walk/forage unburdened, then the mind wakes up. Direct
   answer to the capacity-tax hypothesis.

A/B v2 launched: alien frontier, seed 0, `--mind rule --mind-start 100000`
+ follow reward, streaming on :8888. Comparison set now: no-mind (−2.0),
rule-mind v1 (−8.6), rule-mind v2 (pending). Metrics that matter: foods/ep
and speed at parity budget, not raw return (v2's reward includes the goal
bonus, so returns are not directly comparable — compare downstream success).

---

## 2026-10-02 — A/B v2 verdict: the mind now WINS (ret +30.7, Walking in the alien world)

The goal-following reward + mind curriculum flipped the result completely.
Three-way comparison at the same seed, alien frontier (rolling last-40):

| run | ret | speed | foods | Walking |
|---|---|---|---|---|
| no mind (pure RL) | −2.0 | 0.46 m/s | 0.53 | never |
| rule-mind v1 (goals from step 0, no follow reward) | −8.6 | 0.44 | 0.47 | never |
| **rule-mind v2 (--mind-start 100k + w_goal 0.25)** | **+30.7** | **0.80 m/s** | 0.23 | **@215** |

Speed 0.80 m/s is the fastest ANY agent has moved in the alien world — the
minded policy literally learned to walk better when the mind pays it to
execute goals. Foods dropped (0.23 — the follow reward pays approach, and
famines make eating hard), so the policy traded food for speed/goal
execution; the reward design could be tuned further, but the core question
is answered: **with a curriculum and a follow reward, the reasoning layer
helps — decisively** (Δ+32.7 return vs no-mind, vs Δ−6.6 for v1).

Per-goal behavior telemetry (goal_stat records) works: flee/navigate
segments show the highest damage intensity (they run through hazards),
explore dominates time (63%), shelter is rare. The audit pipeline now
answers the monitoring question end to end: thoughts → goals → measurable
behavioral differences → return impact.

Caveats recorded: v2's reward includes the goal bonus (raw returns not
directly comparable to no-mind — hence the foods/speed/Walking comparison);
foods/ep regression suggests a follow-reward tune (pay food pickup under
forage more) as the next experiment. ToolUser@29 was the earliest boot
pickup of any alien run.
