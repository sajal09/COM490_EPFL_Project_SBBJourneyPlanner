# K-Candidate Path Generation

## Debugging Path

To understand how candidate paths are generated, start at the user-facing entry
point and trace the calls backward:

1. `notebook_ui.ipynb`
     - Locate the `find_routes` function.
2. `backend.py`
     - Follow the call to `jp.route_latest`.
3. `journey_planner.py`
     - Follow `JourneyPlanner.route_latest` to the router.
4. `routing/router.py`
     - `route_dynamic_backward` is the primary function to understand.

## K-Path Generation

For every `(buffer_steps, slack_steps)` pair defined in the routing function,
the planner generates a set of K candidate paths.

For a given buffer and slack value, K paths are generated with a time-expanded
Dijkstra algorithm. The key implementation is
`latest_departure_k_dijkstra_generator` in `routing/router.py`.

The implementation does not construct an explicit graph. Instead, it uses:

- A dataframe of transit trips between stops.
- Walking edges available from each stop to other stops.
- Time-expanded nodes represented as `(stop_id, time)` pairs.

Because time is part of each node, a stop can be visited repeatedly at
different times. Therefore, finding the latest departure from the source does
not end the search. The Dijkstra search can continue and discover the source
again with an earlier departure time, allowing it to collect additional
candidate paths.

Sayantan was working on a networkx implementation (graph consisting of ride edges, waiting edges, walk edges, source edges and sink edges) but we did not use it.

## Potential Optimization in latest_departure_k_dijkstra_generator

The following loop can be made faster:

```python
for edge in rev_graph.get(stop, []):
        if time.monotonic() > deadline:
                break
        if edge['arr_time'] > t:
                continue
```

Since `arr_time` is already sorted in ascending order, the eligible edge could
potentially be located with binary search instead of scanning linearly. This
could reduce the lookup from $O(N)$ to $O(\log N)$.

## Role of Buffer and Slack

- **Transfer buffer:** Favors paths with a greater probability of completing
    successfully by allowing additional time for transfers.
- **Slack:** Encourages path diversity by producing alternatives that arrive
    earlier.

# Scoring Paths

You can also backtrack it from `notebook_ui.ipynb` where the final function used is path_confidence in models/delay_lookup.py

It just multiples the probability of fulfilling the connection at each transfer and at the final destination (Independence assumption).

## Calculating the Probability of Fulfilling the Connection

To calculate the probability of fulfilling the connection, we do it smartly:

- **Arrival**: If at a connection, the bus is arriving from an already running trip, then we calculate the arrival time distribution as (real time position at the previous stop of the running trip + segment time distribution of that previous stop to the arrival stop). Else we take the arrival distribution as the discrete/gumbel distribution of past arrival delays at that stop.

- **Departure**: Similarly, if at a connection, the next bus to be taken is already on a running trip, then we calculate the departure time distribution as (real time position at the previous stop of the running trip + segment time distribution of that previous stop to the departure stop). Else we assume zero departure delay since we are planning for the future so no departure delays have happened anywhere.
**Edge Case**: if at a connection, the next bus to be taken is already on a running trip but its real time position is not available, then we calculate the departure time distribution as (scheduled departure time at the current stop + departure delay distribution at the current stop)

**p**: Then to calculate probability of not missing the connection, you can do departure time distribution - arrival time distribution < scheduled gap and calculate it either by CDF or Monte Carlo.