# Backward k-best Dijkstra: Notes

Notes on `latest_departure_k_dijkstra_generator` in `com490/routing/router.py`. The running example has destination D and a deadline of 08:40.


## Q1. Walk through the function alongside the code

Here's the function in chunks, with the explanation next to the lines it describes.

---

### 1. Signature and inputs
```python
def latest_departure_k_dijkstra_generator(self, df_transit, df_walk, stop_id_end, deadline_sec,
                                          start_id, k_paths=3, transfer_buffer=TRANSFER_BUFFER_SEC):
```
| Argument | Meaning |
|---|---|
| `df_transit` | Timetable rows loaded for the ~2-hour window: trip, stop, stop order, times |
| `df_walk` | Pairs of stops within walking distance, with walking time in seconds |
| `stop_id_end`, `deadline_sec` | Where you must be, and by when (seconds since midnight) |
| `start_id` | Where you leave from |
| `k_paths` | How many different routes to return |
| `transfer_buffer` | Minimum seconds between getting off one vehicle and the next leg (default 60) |

The function uses `yield`, which makes it a **generator**: it hands back one route at a time instead of building the full list first.

---

### 2. Build the lookup tables
```python
rev_graph  = self._reverse_transit_graph(df_transit)
walk_graph = self._walk_graph(df_walk)
```
- `rev_graph[stop]` lists every vehicle hop that **arrives at** `stop`, as `{trip_id, prev_stop, dep_time, arr_time}`, latest arrival first.
  Example: `rev_graph[D] = [{bus 33, from B, dep 08:25, arr 08:34}]`
- `walk_graph[stop]` lists nearby stops, as `{next_stop, walk_time}`, in both directions.
  Example: `walk_graph[D] = [{C, 480 s}]`

---

### 3. Search state
```python
count_vis = {}                                  # stop -> times explored
pq = [(-deadline_sec, stop_id_end, ())]         # start: destination at the deadline, empty path
paths_yielded = 0
deadline = time.monotonic() + MAX_SEARCH_SEC    # stop after 20 s of real time
```
- **`-deadline_sec`**: `heapq` always pops the smallest value. Storing minus the time means the **latest** time comes out first.
- **`()`**: an empty path. Paths grow as a linked list: `(newest_edge, older_path)`.
- The variable name `deadline` is reused here for the time limit. It's not the arrival deadline; that one is `deadline_sec`.

Queue now: `[D @ 08:40]`

---

### 4. Pop the latest entry
```python
while pq:
    neg_t, stop, path_hist = heapq.heappop(pq)
    t = -neg_t                                   # latest time you can be at `stop`
```
First pop: `stop = D`, `t = 08:40`.

```python
    current_count = count_vis.get(stop, 0)
    if current_count >= k_paths:
        continue                                 # already explored k times, skip
```
Classic Dijkstra explores each stop once. Allowing `k` explorations is what produces `k` different routes.

---

### 5. Reached the origin, so a route is complete
```python
    if stop == start_id:
        latest_mock = {stop_id_end: deadline_sec}
        path_after_mock = {}
        node = path_hist
        while node:                                       # follow the linked list
            u, v, trip_id, dep_time, arr_time, edge_type = node[0]
            node = node[1]                                # move to the older part
            latest_mock[u] = dep_time                     # when you leave stop u
            path_after_mock[u] = (v, trip_id, dep_time, arr_time, edge_type)  # u's next hop
        yield latest_mock, path_after_mock
        paths_yielded += 1
        if paths_yielded >= k_paths:
            break
```
- The search ran backward, so the newest item in `path_hist` is the **first** leg of the real journey. Following the list reads the route in travel order: A → B → D.
- `path_after_mock[A] = (B, m1, 08:15, 08:20, 'transit')` means "from A, take m1 at 08:15 and reach B at 08:20".
- Later, `_reconstruct_raw_backward` follows `path_after` from start to end to produce the list of legs.
- The latest time always pops first, so **the first route yielded has the latest possible departure**.
- After a yield (when fewer than `k` routes are found), execution falls through and keeps expanding from the origin. That's harmless but wasted work.

---

### 6. Mark this stop and find the next leg
```python
    count_vis[stop] = current_count + 1
    last_edge = path_hist[0] if path_hist and len(path_hist) > 0 else None
```
`last_edge` is the leg that comes **right after** this stop in real travel. It's called "last" only because it was the last one added during the backward search.
Its fields are `(from, to, trip_id, dep_time, arr_time, type)`. Index `[2]` is the trip, `[3]` the departure, `[5]` the type.

---

### 7. Move backward along vehicles
```python
    for edge in rev_graph.get(stop, []):          # every hop arriving at `stop`
        if time.monotonic() > deadline:
            break                                  # 20-second safety limit
        if edge['arr_time'] > t:
            continue                               # arrives too late, can't use it
```
Example at D (`t = 08:40`): bus 33 arrives 08:34 ≤ 08:40, so it's allowed.
Waiting at a stop is built into this check: any vehicle arriving at or before `t` works, so no explicit "dwell" edges are needed.

```python
        curr_trip = edge['trip_id']
        if last_edge:
            last_trip = last_edge[2]
            if curr_trip != last_trip:            # changing vehicles = a transfer
```
Staying on the **same trip** skips everything below, since you never left your seat.

```python
                last_parts = last_trip.split('.')
                if len(last_parts) >= 3 and last_parts[2] in curr_trip:
                    continue                       # looks like the same vehicle, skip
```
SBB trip IDs look like `1.TA.91-33-j26-1.5.H`. This takes the third dot-separated part (`91-33-j26-1`) and rejects the hop if the other trip's ID contains it. That blocks a fake "transfer" between duplicate copies of one vehicle. It's a rough text match, so it could reject a real transfer by mistake.

```python
                next_transit_dep = last_edge[3]
                if edge['arr_time'] + transfer_buffer > next_transit_dep:
                    continue                       # connection too tight
```
Example at B, where the next leg is bus 33 leaving at 08:25:
- m1 arrives 08:20 → 08:20 + 60 s = 08:21 ≤ 08:25 → allowed.
- A train arriving 08:24:30 → 08:25:30 > 08:25 → rejected.
A bigger `transfer_buffer` rejects more tight connections, which gives safer routes.

```python
        ps   = edge['prev_stop']
        cand = edge['dep_time']
        new_edge = (ps, stop, curr_trip, cand, edge['arr_time'], 'transit')
        heapq.heappush(pq, (-cand, ps, (new_edge, path_hist)))
```
Step back to the previous stop. The latest you can be there is when this vehicle leaves it.
Example: push `(-08:25, B, (B→D on bus 33, ()))`.
`(new_edge, path_hist)` adds the new leg to the front of the linked list without copying anything.

---

### 8. Move backward on foot
```python
    if last_edge is None or last_edge[5] != 'walk':     # no two walks in a row
        for edge in walk_graph.get(stop, []):
            if time.monotonic() > deadline:
                break
            nbr = edge['next_stop']
            if count_vis.get(nbr, 0) >= k_paths:
                continue                                 # neighbor already used up
            cand = t - edge['walk_time']                 # leave early enough to arrive by t
            new_edge = (nbr, stop, 'walk', cand, t, 'walk')
            heapq.heappush(pq, (-cand, nbr, (new_edge, path_hist)))
```
Example at D (`t = 08:40`), with C 8 minutes away on foot: push `C @ 08:32` with a walk leg C → D.
The "no two walks in a row" rule keeps the search from chaining walks past the walking limit.
The walk is stored with `trip_id = 'walk'`. Moving from a walk to a vehicle therefore counts as a trip change, so the buffer applies to vehicle → walk (rejected if it's too tight), but **not** walk → vehicle.

---

### 9. The full run for the example (k = 3)
| Pop | What it adds to the queue | Queue after |
|---|---|---|
| D @ 08:40 | bus 33 → **B @ 08:25**; walk → **C @ 08:32** | C 08:32, B 08:25 |
| C @ 08:32 | bus 701 → **B @ 08:12** | B 08:25, B 08:12 |
| B @ 08:25 | m1 (08:20 + 60 s ≤ 08:25) → **A @ 08:15**; 701 → **A @ 08:05** | A 08:15, B 08:12, A 08:05 |
| A @ 08:15 | **Route 1:** m1 → B, bus 33 → D | … |
| B @ 08:12 | 701 (same trip, no check) → **A @ 08:05** | A 08:05, A 08:05 |
| A @ 08:05 | **Route 2:** 701 → C, walk → D | … |
| A @ 08:05 | **Route 3:** 701 → B, bus 33 → D, stop (k = 3 reached) | |

`route_dynamic_backward` then reruns this for each buffer and slack value, removes duplicate routes, and the confidence code scores each one.

---
## Q2. What is the point of limiting `count_vis`? What does it prevent?

`count_vis` caps how many times each stop can be explored, at `k_paths`. Without that cap the search would keep exploring every combination of trips. With a cap of 1 it could only ever return one route. Setting it to `k` is the middle ground.

### Why there has to be some limit
In this search a queue entry is a **whole partial route**, not just a stop. Every vehicle arriving at a stop creates a new entry, and each of those entries branches again at the previous stop. Without a cap:

1. **The number of entries explodes at busy stops.** Lausanne station has dozens of trains and buses arriving in a 2-hour window. Each one creates a partial route. Each of those then branches into every vehicle arriving at *its* previous stop, and so on. The number of partial routes grows roughly exponentially with the number of transfers. The queue would fill memory, or the search would just run until the 20-second limit, with most of that time spent on routes nobody would take.
2. **It would explore pointless back-and-forth routes.** Nothing stops a route from passing the same stop twice, for example riding B → C and then walking C → B to catch something earlier. Times always go down in the backward search, so this can't loop forever, but it can bounce around until the time window runs out. Once B's `k` slots are used, these repeat visits are dropped.
3. **It would return many near-copies of one route.** Many entries reach the origin as small variations: the same bus caught one stop earlier, or the same transfer done at a neighboring platform. The cap limits how much any single stop can contribute to that.

### Why the limit isn't 1
With `k = 1` this is classic Dijkstra. Each stop is settled once, at its single latest time. The origin is reached once, so you get **one route**.

Your planner needs alternatives: later but riskier routes, earlier but safer ones. Confidence scoring and the check against Q choose between them. Allowing each stop `k` visits keeps the `k` latest ways through it, so up to `k` different routes can reach the origin. This is the standard way to adapt Dijkstra to find "k shortest paths".

The pops run latest-first, so the `k` entries a stop keeps are its **`k` latest** ones. That fits the goal of leaving as late as possible.

### A case it gets wrong
The textbook guarantee for this trick assumes a stop's best routes don't depend on how you continue from it. That isn't true here. Whether you can use a vehicle arriving at B depends on the **next leg**:
- If you stay on the same trip, no buffer is needed.
- If you change vehicles, you need arrival + 60 s ≤ next departure.
- If the next leg is a walk, you can't walk again.

The counter only counts **stops**, so a valid route can be dropped. With `k = 1`:

1. B is popped first at **08:25**, because the next leg is bus 33 leaving at 08:25 and that's the latest time.
2. The only vehicle into B arrives at **08:24:40**. 08:24:40 + 60 s > 08:25, so the transfer is rejected. This entry leads nowhere.
3. Another queue entry for **B @ 08:22** would have worked: it continues on the same train, so no buffer is needed. But B has already used its one visit, so this entry is skipped.
4. The result is "no route" or a worse route, even though a valid one exists.

With a larger `k` this happens less often, but it can still happen at busy stops where the `k` slots fill up with entries that fail their transfer checks. Note that a stop's counter goes up when it is popped, even if none of its expansions work.

### How it could be fixed:
Count visits per **(stop, how you continue)** instead of per stop. For example, key the counter on `(stop, next trip_id)`, or on `(stop, whether the next leg is a walk)`. A failed transfer then can't use up the slots of a route that stays on its vehicle.

The method used in production journey planners, RAPTOR, avoids this differently. It searches in rounds, one round per number of transfers, so routes with different numbers of transfers never compete for the same slot.

---
## Q3. Why can't a route go B → C → B?

It can happen. Nothing in the code stops one route from visiting the same stop twice. There's no "already on this route" check. What the code does do is make such loops **finite** and **usually crowded out**.

### Why it can happen
Each queue entry carries its own `path_hist`, but the code never looks inside it to ask "is this stop already on this route?" The only per-stop limit is `count_vis`, which is shared by **all** routes.

Example (deadline 08:40 at D):
- The next leg from B is bus 33 at 08:25, so B is in the queue at **08:25**.
- From B, the walk to C (5 min) → push **C @ 08:20**.
- Into C: a bus that left **B at 08:14** and reached C at 08:18 → push **B @ 08:14**.

In real travel order that route is: be at B at 08:14, ride to C, walk back to B, take bus 33 at 08:25. It's a pointless loop, but every individual check passes.

### Why it can't loop forever
Every backward step makes the time strictly earlier:
- Vehicle step: the new time is the vehicle's departure, which is before its arrival.
- Walk step: the new time is `t − walk_time`, and two walks in a row are blocked.

The time can't go below the start of the ~2-hour window, so any loop runs out of time.

### Why it's usually crowded out
A route that loops back to B always reaches B **earlier** than the direct version. In the example, the loop needs you at B at 08:14, while the direct version only needs 08:25. The queue pops the latest time first, so B's direct entries are popped first and use up B's `k` slots. By the time the loop entry (B @ 08:14) comes out, B is usually full and the entry is skipped.

So the cap doesn't forbid loops, it just usually leaves them no slot.

### When one gets through
If B has fewer than `k` entries that work, a loop entry can take a slot and even reach the origin. A loop is **never better** than the direct route: you're at B earlier, so you leave the origin at the same time or earlier. But it can **tie**, if both reach B from the origin on the same vehicle. Then:
- it isn't removed as a duplicate, because its legs are different, and
- nothing breaks ties toward fewer transfers.

So the output can include a route that leaves at the same time as a clean one but has a pointless detour, an extra transfer, and therefore a lower confidence score.

### The fix
Before pushing a step, check whether the previous stop is already in that route's `path_hist`, and skip it if so. Following the linked list costs a few steps per push. A cheaper option is to carry a set of visited stops alongside each entry.

---
## Q3.2. What does "it can tie, if both reach B from the origin on the same vehicle" mean?

"Tie" means **leaving the origin at the same moment**. The planner ranks routes by departure time from the origin (latest first) and has no second rule for breaking ties. When two routes leave at the same time, it treats them as equally good.

### How the loop route can leave at the same time
Both routes start with the **same vehicle** from A to B, then split:

| Time | Direct route | Loop route |
|---|---|---|
| 08:05 | Leave A on the 701 | Leave A on the 701 |
| 08:12 | Arrive B, wait | Arrive B |
| 08:14 | (waiting) | Take a bus B → C |
| 08:18 | (waiting) | Arrive C, walk back to B |
| 08:23 | (waiting) | Arrive B |
| 08:25 | Bus 33 to D | Bus 33 to D |
| 08:34 | Arrive D | Arrive D |

The direct route needs you at B by about 08:24. The loop route needs you at B by about 08:13. The 701 arriving at 08:12 is early enough for **both**. So both leave A at 08:05 and arrive at D at 08:34. Measured by departure time, which is all the planner compares, they're equal.

### When it's not a tie
If a **later** vehicle from A reached B before 08:24, say one leaving A at 08:15, the direct route would use it and leave at 08:15. The loop route can't, because it must be at B by 08:13. The direct route then leaves later and wins outright, and the loop is just a worse option. The tie only happens when the 08:05 vehicle is the last one from A that works for both routes.

### Why the tie matters
- The two routes have different legs, so they aren't removed as duplicates. Both get returned.
- Nothing ranks fewer transfers or less walking higher, even though the project brief asks for that. Their relative order is effectively arbitrary.
- The loop route has two extra connections: 701 → bus at 08:12 → 08:14, and walk → bus 33. Each lowers its confidence. So the user would see a strictly worse route listed next to the good one, with the same departure time.

A tie-break rule (equal departure → fewer transfers, then less walking) would push the loop route below the direct one. A per-route visited-stop check would stop it from being created at all.
