
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
