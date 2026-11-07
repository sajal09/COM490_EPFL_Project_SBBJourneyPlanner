# Route Scoring: Summary

How the planner scores each candidate route. Code: `com490/models/delay_lookup.py` (`path_confidence`, `p_connection_*`) and `com490/backend.py` (`score_routes`, `build_bus_snapshot`, `make_bus_state_fn`).

---

## 1. The big idea
Each route gets one number: **the probability that every connection works and you arrive by the deadline**. Delays can only break a route at two kinds of point:
- **each transfer**: does the incoming vehicle arrive in time for the next one?
- **the final arrival**: does the last vehicle get you there before the deadline?

```
confidence = P(transfer 1) × P(transfer 2) × … × P(arrive by deadline)
```
Multiplying assumes the events are **independent**. That's a simplification: one late train can affect several connections.

---

## 2. Reading the route as events
`path_confidence` walks the route as a list of events `(ts, x, y, z, [walk_time])`:

| Event | Meaning | What happens |
|---|---|---|
| alight | Get off trip `y` at stop `z` | Saved as `last_alight` |
| walk | Walk between stops | Added to `walk_accum_s` |
| board | Get on trip `y` at stop `x` | **A transfer**: its probability is computed |

Boarding at the origin isn't scored, because there's no earlier vehicle to be late.

---

## 3. Scoring one transfer
A transfer works if `actual arrival + walk ≤ actual departure`. That is the same as:
```
arrival delay − departure delay ≤ slack − walk      (slack = scheduled departure − scheduled arrival)
```
Delays come from tables grouped by (day of week, line, stop, hour). When a group has fewer than 50 observations, the lookup falls back to (line, stop, hour), then to (stop, hour).

The method depends on what is known when the query is made.

### Option A: discrete model, exact (static)
- Each delay is a table of probabilities over 71 one-minute buckets, from −10 to +60 minutes.
- `np.convolve(arr_pmf, dep_pmf[::-1])` gives the distribution of (arrival delay − departure delay).
- Summing everything ≤ the slack gives the exact probability, with no randomness.
- The slack is rounded down to whole minutes, which is slightly cautious.

### Option B: Gumbel model (static)
- Departure assumed on time: one closed-form Gumbel CDF.
- Both delays known, with equal β: the difference follows a logistic distribution, also closed form.
- Both delays known, with different β: Monte Carlo.

### Option C: real-time positions
This is used when the incoming bus has already been seen running before "now".

**Ingredients**
- **Snapshot** (`build_bus_snapshot`): replays one historical day, and for each trip keeps the last stop it actually reached before `query_time_s`. `bus_state_fn(trip)` returns `(stop, time seen)`, or None if the trip hasn't started.
- **Segment travel times** (`build_segment_travel_times`): for stop pairs on the same trip, up to 20 stops apart, the time from actual departure at X to actual arrival at Y. It's grouped by (line, from, to, weekday/weekend, hour) and stored as 7 quantiles (5–95%). Samples are drawn by interpolating between those quantiles.

**Calculation**
```python
arr_samples = last_seen_time + travel_time_sample(current_stop → transfer_stop) + walk
dep_samples = connecting bus last_seen_time + its travel_time_sample   # if it's already running
            # else: scheduled time + historical departure delay (no travel data)
            # else: scheduled time (it hasn't started, so assume on time)
p = mean(arr_samples + buffer <= dep_samples)        # 10,000 draws
```
If there's no snapshot entry or no travel-time data for the incoming bus, the code falls back to Option A or B. Monte Carlo is used because both sides are irregular, quantile-based distributions with no closed form. With 10,000 draws the error band is under about ±1 percentage point.

**Example.** Now = 08:17. Your m1 was last seen at X at 08:16:30, and X → B takes about 5 min, so you're at B around 08:21:30. Bus 33, scheduled 08:25, was last seen at Y at 08:19:00, and Y → B takes about 7 min, so it leaves B around 08:26:00. Almost every draw succeeds, so p ≈ 0.99. The static model, assuming bus 33 is on time, would give something like 0.93.

---

## 4. Final arrival
```python
slack_s = deadline − scheduled drop-off − final walk
p_final = arr_dist.cdf(slack_s)     # P(last vehicle's delay ≤ slack), always historical
```

---

## 5. Worked example (static)
Route: A → m1 → B (arr 08:20) → bus 33 (dep 08:25) → D (arr 08:34). Deadline 08:40, Q = 80%.

| Step | Calculation | Example p |
|---|---|---|
| Transfer at B | slack 5 min → P(m1 delay ≤ 5 min) | 0.93 |
| Final arrival | slack 6 min → P(bus 33 delay ≤ 6 min) | 0.95 |
| **Overall** | 0.93 × 0.95 | **0.88 → meets Q** |

(The p values are made-up examples. `breakdown` stores one row per step so the weakest link can be shown.)

---

## 6. `score_routes`: ranking
- Calls `path_confidence` with `buffer_s = 0` (the search already enforced a buffer) and `n_samples = 10_000`.
- Records `overall_p`, `n_transfers`, `meets_Q = p ≥ Q` and the breakdown.
- Sorts **only by departure time** (latest first), with no tie-break on transfers or probability. Routes that fail Q are kept and flagged.

---

## 7. Which method runs in the actual app
`notebook_ui.ipynb` builds the snapshot and passes `bus_state_fn` in, so:
- transfers whose incoming bus is already running use **Option C (Monte Carlo)**;
- other transfers use **Option A or B** (exact or closed form);
- connecting vehicles that haven't started are assumed to leave on time.

So the resume line "scored routes via 10k Monte Carlo runs" is fair for real-time transfers. It's just not the only method used.

---

## 8. Weak points
1. **Independence:** multiplying probabilities ignores links between delays.
2. **No backup plan:** a missed connection counts as failure, even if a later vehicle would still arrive in time.
3. **Hour from the schedule:** a 07:58 arrival uses the 07:00 delay group even if it really arrives after 08:00.
4. **Snapshot cached at a fixed path:** a new day or query time reuses the old snapshot unless the file is deleted.
5. **Dwell time ignored in Option C:** travel time starts at departure but the snapshot gives arrival, so predictions are slightly optimistic.
6. **Bus already past the transfer stop:** no travel-time match, so it falls back to the static model.
7. **Trip IDs must match** between istdaten and the timetable; otherwise Option C never runs.
8. **Final arrival ignores real-time data.**
9. **Quantile sampling** never goes outside the 5–95% range, so extreme delays are cut off.
