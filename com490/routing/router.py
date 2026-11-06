import heapq
import time
import pandas as pd
import networkx as nx
from db import fetch_async, execute
from db import RIDE_EDGES, DWELL_EDGES, CREATE_RELEVANT_TRIPS ,SOURCE_EDGES, SINK_EDGES, WALK_EDGES, TRANSIT_QUERY, WALK_QUERY

from routing.time_utils import time_to_seconds, day_to_column

WALK_SPEED_M_PER_SEC = 50 / 60
TRANSFER_BUFFER_SEC  = 60   # minimum seconds between alighting one trip and boarding another
MAX_SEARCH_SEC = 20.0
MAX_PQ_SIZE = 20000
class Router:
    
    """
    Two search modes:

      * route_dynamic           — forward earliest-arrival from a given departure time.
      * route_dynamic_backward  — backward latest-departure search anchored at a deadline,
                                   with a slack sweep to surface progressively more
                                   conservative (higher-confidence) candidates.

    Both share the same time-expanded graph built from the user's `stop_times` and
    `stop_to_stop` tables via build_time_expanded_graph.
    """

    def __init__(self, conn, userns: str, sharedns: str, query_window_size: int = 8000):
        self.conn    = conn
        self.userns  = userns
        self.sharedns = sharedns
        self.query_window_size = query_window_size
    

    def route(
        self,
        start_id: int,
        end_id: int,
        departs: str,
        day: str,
        max_walk_m: int = 500,
    ) -> pd.DataFrame:
        depart_sec = time_to_seconds(departs)
        day_col    = day_to_column(day)

        execute(self.conn, CREATE_RELEVANT_TRIPS,
            userns=self.userns,
            query_time=depart_sec,
            day=day,
            window_end=depart_sec + self.query_window_size
        )

        # --- fetch all edge types in parallel ---
        f_ride     = fetch_async(self.conn, RIDE_EDGES,   userns=self.userns, day_col=day_col)
        f_dwell    = fetch_async(self.conn, DWELL_EDGES,  userns=self.userns)
        f_source   = fetch_async(self.conn, SOURCE_EDGES, userns=self.userns, source=start_id, departure_time=depart_sec)
        # f_sink     = fetch_async(self.conn, SINK_EDGES,   userns=self.userns, arrival_time=depart_sec + self.query_window_size)
        f_sink     = fetch_async(self.conn, SINK_EDGES,   userns=self.userns, sink = end_id, arrival_time=depart_sec + self.query_window_size) 
        f_walk     = fetch_async(self.conn, WALK_EDGES,   userns=self.userns, max_walk_m=max_walk_m)

        ride_edges   = f_ride.result()
        dwell_edges  = f_dwell.result()
        source_edges = f_source.result()
        sink_edges   = f_sink.result()
        walk_edges   = f_walk.result()

        print('Finished fetching edges..')

        # --- build graph ---
        G = nx.DiGraph()
        _add_edges(G, ride_edges)
        _add_edges(G, dwell_edges)
        _add_edges(G, source_edges)
        _add_edges(G, sink_edges)
        _add_edges(G, walk_edges)

        # # --- run dijkstra ---
        source_node = (start_id, depart_sec)

        if end_id is not None:
            return self._route_single(G, source_node, end_id, sink_edges)
        else:
            return self._route_all(G, source_node)


    def _route_single(self, G: nx.DiGraph, source_node: tuple, end_id: int, sink_edges: pd.DataFrame) -> pd.DataFrame:
        """Route from source to a single destination stop."""
        # arrival_time = _earliest_arrival(sink_edges, end_id)
        destination_node = (end_id, source_node[1] + self.query_window_size)

        path = nx.dijkstra_path(G, source=source_node, target=destination_node, weight='weight')
        return _path_to_df(path)

    def _route_all(self, G: nx.DiGraph, source_node: tuple) -> dict[str, list[tuple]]:
        """Route from source to all reachable destination stops."""
        lengths, paths = nx.single_source_dijkstra(G, source=source_node, weight='weight')


        best: dict[str, tuple] = {}  # stop_id -> best (stop_id, time) node
        for node in lengths:
            stop_id, arrival_time = node
            if stop_id not in best or arrival_time < best[stop_id][1]:
                best[stop_id] = node

        return {
            stop_id: _path_to_edges(G, paths[node])
            for stop_id, node in best.items()
        }



    # public: forward search


    def route_dynamic(self, start_id, departs, day, end_id=None, max_walk_m=100, top_k = 5,
                       id_to_name=None, transfer_buffer_sec=TRANSFER_BUFFER_SEC):
        """
        Forward earliest-arrival route from `start_id` at time `departs`.

        Returns:
            dist       : stop_id -> earliest arrival (seconds since midnight)
            prev       : raw back-pointer dict
            route      : event tuples for the target stop (or dict-of-routes if end_id is None)
            results_df : per-stop summary DataFrame sorted by arrival time
        """
        query_time_sec = time_to_seconds(departs)
        day_col        = day_to_column(day)

        df_transit, df_walk = self.build_time_expanded_graph(query_time_sec, day_col,
                                                               max_walk_m=max_walk_m)
        path_generator = self.earliest_arrival_k_dijkstra_generator(df_transit, df_walk, start_id,
                                                     query_time_sec, end_id, k_paths = top_k, transfer_buffer = transfer_buffer_sec)
        routes = []
        first_dist, first_prev = None, None
        for dist, prev in path_generator:
            if first_dist is None:
                first_dist, first_prev = dist, prev
            route = self.reconstruct_route(prev, dist, start_id, end_id)
            routes.append(route)

        results = {name: t for s, t in first_dist.items()
                   if (name := id_to_name.get(s)) is not None}
        results_df = pd.DataFrame(list(results.items()),
                                   columns=['stop_name', 'earliest_arrival_sec'])
        results_df['earliest_arrival_hhmm'] = results_df['earliest_arrival_sec'].apply(
            lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}")
        results_df = results_df.sort_values('earliest_arrival_sec').reset_index(drop=True)
        results_df['travel_time_sec']  = results_df['earliest_arrival_sec'] - query_time_sec
        results_df['travel_time_hhmm'] = results_df['travel_time_sec'].apply(
            lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}")
        return first_dist, first_prev, routes, results_df




    # public: backward search with slack sweep 

    def route_dynamic_backward(self, start_id, end_id, deadline, day,
                                max_walk_m=500, slack_steps=(0, 180, 360, 540, 720),
                                window_sec=7400, top_k =5,
                                buffer_steps=(TRANSFER_BUFFER_SEC,)):
        """
        Backward latest-departure search anchored at `deadline`.

        Builds the time-expanded graph ONCE for the window [deadline - window_sec, deadline],
        then runs one latest-departure Dijkstra per (buffer, slack) pair.

        A slack value `s` (seconds) shifts the effective deadline to `deadline - s`,
        forcing the search onto paths that arrive earlier. A buffer value `b` (seconds)
        is the minimum gap required between alighting one trip and boarding another;
        larger `b` rejects tight transfers and yields more conservative itineraries.
        Both knobs push toward higher confidence at the cost of an earlier departure
        or fewer candidate transfers.

        Returns:
            list of event-tuple paths, deduplicated by signature. Outer order is
            `buffer_steps`, inner order is `slack_steps` — so the first path is the
            latest-departing candidate at the smallest buffer.
        """
        deadline_sec = time_to_seconds(deadline)
        day_col      = day_to_column(day)
        window_start = deadline_sec - window_sec

        df_transit, df_walk = self.build_time_expanded_graph(
            query_time_sec = window_start,
            day_col        = day_col,
            window_sec     = window_sec,
            max_walk_m     = max_walk_m,
        )

        paths    = []
        seen_sig = set()
        for buffer_sec in buffer_steps:
            for slack in slack_steps:
                effective = deadline_sec - slack
                if effective < window_start:
                    continue   # slack pushes effective deadline outside the graph window
                
                path_generator = self.latest_departure_k_dijkstra_generator(
                    df_transit, df_walk, end_id, effective, start_id=start_id,
                    k_paths=top_k, transfer_buffer=buffer_sec
                )
                for latest, path_after in path_generator:
                    if start_id not in latest:
                        continue   # unreachable at this slack level
                    raw = _reconstruct_raw_backward(path_after, start_id, end_id)
                    if not raw:
                        continue
                    if start_id not in latest:
                        continue   # unreachable at this (slack, buffer) level

                    raw = _reconstruct_raw_backward(path_after, start_id, end_id)
                    if not raw:
                        continue

                    sig = _raw_sig(raw)
                    if sig in seen_sig:
                        continue
                    seen_sig.add(sig)
                    paths.append(_raw_to_events(raw))

        return paths



    # public: graph build

    def build_time_expanded_graph(self, query_time_sec, day_col,
                                    window_sec=7400, max_walk_m=100):
        """
        Pulls transit + walk edges from Trino in parallel.

        For FORWARD search, pass `query_time_sec = departure_time` — trips with any
        departure in [query_time_sec, query_time_sec + window_sec] are loaded.

        For BACKWARD search, pass `query_time_sec = deadline - window_sec` so the
        same filter loads trips whose departures fall in the window leading up to
        the deadline (assumes journeys ≤ window_sec long).
        """
        f_transit = fetch_async(self.conn, TRANSIT_QUERY,
                                userns     = self.userns,
                                query_time = query_time_sec,
                                day        = day_col,
                                window_end = query_time_sec + window_sec)
        f_walk    = fetch_async(self.conn, WALK_QUERY,
                                userns     = self.userns,
                                max_walk_m = max_walk_m)
        df_transit = f_transit.result()
        df_walk    = f_walk.result()
        df_walk['walk_time_sec'] = (df_walk['distance'] / WALK_SPEED_M_PER_SEC).round().astype(int)
        return df_transit, df_walk



    # public: forward Dijkstra

    def earliest_arrival_dijkstra(self, df_transit, df_walk, stop_id_start, query_time,
                                   target_stop_id=None,
                                   transfer_buffer_sec=TRANSFER_BUFFER_SEC):
        """
        Forward Dijkstra over the time-expanded graph — minimise arrival time.

        Returns:
            dist : stop_id -> earliest arrival (seconds)
            prev : stop_id -> (prev_stop, trip_id_or_'walk', dep_time, edge_type)
        """
        graph      = self._forward_transit_graph(df_transit)
        walk_graph = self._walk_graph(df_walk)

        best = {stop_id_start: query_time}
        dist = {}
        prev = {}
        pq   = [(query_time, stop_id_start)]

        while pq:
            t, stop = heapq.heappop(pq)
            if stop in dist:
                continue
            dist[stop] = t

            if target_stop_id and stop == target_stop_id:
                break

            for edge in graph.get(stop, []): ## transit edges
                if edge['dep_time'] < t:
                    continue
                # Buffer only on transit-to-transit transfers across different trips.
                if (stop in prev
                        # and prev[stop][3] == 'transit' ### remove thi so walk doesnt slip by 
                        and edge['dep_time'] < t + transfer_buffer_sec
                        and edge['trip_id'] != prev[stop][1]):
                    continue

                nxt = edge['next_stop']
                arr = edge['arr_time']
                if arr < best.get(nxt, float('inf')):
                    best[nxt] = arr
                    prev[nxt] = (stop, edge['trip_id'], edge['dep_time'], 'transit')
                    heapq.heappush(pq, (arr, nxt))

            for edge in walk_graph.get(stop, []):              ## walking edges
                nxt = edge['next_stop']
                arr = t + edge['walk_time']
                if arr < best.get(nxt, float('inf')):
                    best[nxt] = arr
                    prev[nxt] = (stop, 'walk', t, 'walk')
                    heapq.heappush(pq, (arr, nxt))

        return dist, prev




    # public: backward Dijkstra 

    def latest_departure_dijkstra(self, df_transit, df_walk, stop_id_end, deadline_sec,
                                   start_id=None,
                                   transfer_buffer_sec=TRANSFER_BUFFER_SEC):
        """
        Backward Dijkstra — maximise the latest time at every stop such that
        we can still reach `stop_id_end` by `deadline_sec`.

        Every edge in this search decreases the value (transit dep_time < arr_time;
        walking subtracts walk_time), so an early break when `start_id` is settled
        is correct: no later relaxation can produce a larger value at start_id.

        Returns:
            latest     : stop_id -> latest time we can be at this stop (settled only)
            path_after : stop_id -> (next_stop, trip_or_walk, dep_time, arr_time, edge_type)
                         the FIRST forward edge out of this stop on the optimal path.
        """
        rev_graph  = self._reverse_transit_graph(df_transit)
        walk_graph = self._walk_graph(df_walk)

        latest     = {stop_id_end: deadline_sec}
        settled    = {}
        path_after = {}
        pq         = [(-deadline_sec, stop_id_end)]   # max-heap via negation

        while pq:
            neg_t, stop = heapq.heappop(pq)
            t = -neg_t
            if stop in settled:
                continue
            settled[stop] = t

            if start_id is not None and stop == start_id:
                break

            # `nxt_edge` is the forward-next move from `stop` on the optimal path.
            # The transfer buffer kicks in when we'd ALIGHT (incoming transit) and
            # then BOARD a different transit trip at `stop`.
            nxt_edge = path_after.get(stop)

            for edge in rev_graph.get(stop, []):               ## incoming transit edges
                if edge['arr_time'] > t:
                    continue
                if (nxt_edge is not None
                        # and nxt_edge[4] == 'transit' #################### removed --> problematic 
                        and edge['trip_id'] != nxt_edge[1]
                        and edge['arr_time'] + transfer_buffer_sec > t):
                    # print(f"BUFFER REJECT  stop={stop}  arr={edge['arr_time']}  t={t}  buf={transfer_buffer_sec}  nxt_type={nxt_edge[4]}  nxt[1]={nxt_edge[1]}")
                    continue

                ps   = edge['prev_stop']
                cand = edge['dep_time']
                if cand > latest.get(ps, float('-inf')):
                    latest[ps]     = cand
                    path_after[ps] = (stop, edge['trip_id'], edge['dep_time'],
                                       edge['arr_time'], 'transit')
                    heapq.heappush(pq, (-cand, ps))

            for edge in walk_graph.get(stop, []):              ## walking edges, bidirectional
                nbr  = edge['next_stop']
                cand = t - edge['walk_time']
                if cand > latest.get(nbr, float('-inf')):
                    latest[nbr]     = cand
                    # For a walking hop we leave `nbr` at `cand` and arrive at `stop` at `t`.
                    path_after[nbr] = (stop, 'walk', cand, t, 'walk')
                    heapq.heappush(pq, (-cand, nbr))

        return latest, path_after


    def earliest_arrival_k_dijkstra_generator(self, df_transit, df_walk, stop_id_start, query_time,
                                            target_stop_id, k_paths=3, transfer_buffer=TRANSFER_BUFFER_SEC):
        """
        Forward K-Dijkstra generator over the time-expanded graph — minimises arrival time.
        Yields up to `k_paths` distinct routing solutions from `stop_id_start` to `target_stop_id`.

        Yields:
            dist : stop_id -> earliest arrival (seconds)
            prev : stop_id -> (prev_stop, trip_id_or_'walk', dep_time, edge_type)
        """
        graph      = self._forward_transit_graph(df_transit)
        walk_graph = self._walk_graph(df_walk)
        count_vis = {}
        best_arr  = {}
        pq = [(query_time, stop_id_start, ())]
        paths_yielded = 0
        deadline = time.monotonic() + MAX_SEARCH_SEC
        while pq:
    
            t, stop, path_hist = heapq.heappop(pq)
            current_count = count_vis.get(stop, 0)
            if current_count >= k_paths:
                continue
            if stop == target_stop_id:
                dist_mock = {stop_id_start: query_time}
                prev_mock = {}
                node = path_hist
                while node:
                    u, v, trip_id, dep_time, arr_time, edge_type = node[0]
                    node = node[1]
                    dist_mock[v] = arr_time
                    prev_mock[v] = (u, trip_id, dep_time, edge_type)
                yield dist_mock, prev_mock
                paths_yielded += 1
                if paths_yielded >= k_paths:
                    break
            count_vis[stop] = current_count + 1
            last_edge = path_hist[0] if path_hist and len(path_hist) > 0 else None
            ## --- 1. TRANSIT EDGES ---
            for edge in graph.get(stop, []):
                if time.monotonic() > deadline:
                    break
                if edge['dep_time'] < t:
                    continue
                curr_trip = edge['trip_id']
                
                if last_edge:
                    last_trip = last_edge[2]
                    
                    if curr_trip != last_trip:
                        last_parts = last_trip.split('.')
                        if len(last_parts) >= 3 and last_parts[2] in curr_trip:
                            continue
                        if edge['dep_time'] < t + transfer_buffer:
                            continue
                nxt = edge['next_stop']
                arr = edge['arr_time']
                new_edge = (stop, nxt, curr_trip, edge['dep_time'], arr, 'transit')
                heapq.heappush(pq, (arr, nxt, (new_edge, path_hist)))
            ## --- 2. WALKING EDGES ---
            if last_edge is None or last_edge[5] != 'walk':
                for edge in walk_graph.get(stop, []):
                    if time.monotonic() > deadline:
                        break
                    nxt = edge['next_stop']
                    if count_vis.get(nxt, 0) >= k_paths:
                        continue
                    arr = t + edge['walk_time']
                    vis_key = (nxt, count_vis.get(nxt, 0))
                    if best_arr.get(vis_key, float('inf')) <= arr:
                        continue
                    best_arr[vis_key] = arr
                    new_edge = (stop, nxt, 'walk', t, arr, 'walk')
                    heapq.heappush(pq, (arr, nxt, (new_edge, path_hist)))

    def latest_departure_k_dijkstra_generator(self, df_transit, df_walk, stop_id_end, deadline_sec,
                                           start_id, k_paths=3, transfer_buffer=TRANSFER_BUFFER_SEC):
        """
        Backward K-Dijkstra generator over the time-expanded graph — maximises departure time.
        Yields up to `k_paths` distinct routing solutions from `start_id` to `stop_id_end`.

        Yields:
            latest     : stop_id -> latest departure/arrival time (seconds)
            path_after : stop_id -> (next_stop, trip_id, dep_time, edge_type)
        """
        rev_graph  = self._reverse_transit_graph(df_transit)
        walk_graph = self._walk_graph(df_walk)
        count_vis = {}
        # PQ stores: (-departure_time, current_stop, path_history)
        pq = [(-deadline_sec, stop_id_end, ())]
        paths_yielded = 0
        deadline = time.monotonic() + MAX_SEARCH_SEC
        
        while pq:
            neg_t, stop, path_hist = heapq.heappop(pq)
            t = -neg_t
            
            current_count = count_vis.get(stop, 0)
            if current_count >= k_paths:
                continue
                
            if stop == start_id:
                latest_mock = {stop_id_end: deadline_sec}
                path_after_mock = {}
                node = path_hist
                while node:
                    u, v, trip_id, dep_time, arr_time, edge_type = node[0]
                    node = node[1]
                    latest_mock[u] = dep_time
                    path_after_mock[u] = (v, trip_id, dep_time, arr_time, edge_type)
                yield latest_mock, path_after_mock
                paths_yielded += 1
                if paths_yielded >= k_paths:
                    break
                    
            count_vis[stop] = current_count + 1
            last_edge = path_hist[0] if path_hist and len(path_hist) > 0 else None
            
            ## --- 1. TRANSIT EDGES (Moving Backward) ---
            for edge in rev_graph.get(stop, []):
                if time.monotonic() > deadline:
                    break
                if edge['arr_time'] > t:
                    continue
                curr_trip = edge['trip_id']
                
                if last_edge:
                    last_trip = last_edge[2]
                    
                    if curr_trip != last_trip:
                        last_parts = last_trip.split('.')
                        if len(last_parts) >= 3 and last_parts[2] in curr_trip:
                            continue
                        # Transfer buffer check: must arrive before the next leg departs
                        next_transit_dep = last_edge[3]
                        if edge['arr_time'] + transfer_buffer > next_transit_dep:
                            continue
                            
                ps   = edge['prev_stop']
                cand = edge['dep_time']
                new_edge = (ps, stop, curr_trip, cand, edge['arr_time'], 'transit')
                heapq.heappush(pq, (-cand, ps, (new_edge, path_hist)))
                
            ## --- 2. WALKING EDGES (Moving Backward) ---
            if last_edge is None or last_edge[5] != 'walk':
                for edge in walk_graph.get(stop, []):
                    if time.monotonic() > deadline:
                        break
                    nbr = edge['next_stop']
                    if count_vis.get(nbr, 0) >= k_paths:
                        continue
                    cand = t - edge['walk_time']
                    new_edge = (nbr, stop, 'walk', cand, t, 'walk')
                    heapq.heappush(pq, (-cand, nbr, (new_edge, path_hist)))
        
    # ── public: forward reconstruction (kept for route_dynamic) ─────────────

    def reconstruct_route(self, prev, dist, start_id, target_stop_id=None):
        """
        Reconstructs the event list for a forward search.

        Event format:
            (ts, stop,      trip_id, None)   board trip_id at stop at ts
            (ts, None,      trip_id, stop)   alight trip_id at stop at ts
            (ts, prev_stop, None,    stop)   reach stop by walking from prev_stop at ts
        """
        if target_stop_id is None:
            routes = {}
            for stop in dist:
                if stop == start_id:
                    continue
                routes[int(stop)] = _reconstruct_single(prev, dist, start_id, stop)
            return routes
        return _reconstruct_single(prev, dist, start_id, target_stop_id)



    
    # internal: graph builders

    @staticmethod
    def _forward_transit_graph(df_transit):
        """stop_id -> list of outgoing transit edges, sorted by dep_time ascending."""
        df_sorted = df_transit.sort_values(['trip_id', 'stop_sequence']).copy()
        df_sorted['next_stop_id'] = df_sorted.groupby('trip_id')['stop_id'].shift(-1)
        df_sorted['next_arrival'] = df_sorted.groupby('trip_id')['arrival_time'].shift(-1)
        df_edges = df_sorted.dropna(subset=['next_stop_id'])

        graph = {}
        for _, row in df_edges.iterrows():
            graph.setdefault(row['stop_id'], []).append({
                'trip_id':   row['trip_id'],
                'next_stop': row['next_stop_id'],
                'dep_time':  row['departure_time'],
                'arr_time':  row['next_arrival'],
            })
        for stop in graph:
            graph[stop].sort(key=lambda e: e['dep_time'])
        return graph




    @staticmethod
    def _reverse_transit_graph(df_transit):
        """next_stop_id -> list of INCOMING transit edges, sorted by arr_time descending."""
        df_sorted = df_transit.sort_values(['trip_id', 'stop_sequence']).copy()
        df_sorted['next_stop_id'] = df_sorted.groupby('trip_id')['stop_id'].shift(-1)
        df_sorted['next_arrival'] = df_sorted.groupby('trip_id')['arrival_time'].shift(-1)
        df_edges = df_sorted.dropna(subset=['next_stop_id'])

        rev = {}
        for _, row in df_edges.iterrows():
            rev.setdefault(row['next_stop_id'], []).append({
                'trip_id':   row['trip_id'],
                'prev_stop': row['stop_id'],
                'dep_time':  row['departure_time'],
                'arr_time':  row['next_arrival'],
            })
        for stop in rev:
            rev[stop].sort(key=lambda e: e['arr_time'], reverse=True)   # latest-first
        return rev



    @staticmethod
    def _walk_graph(df_walk):
        """stop_id -> list of bidirectional walking edges."""
        walk = {}
        for _, row in df_walk.iterrows():
            a, b, wt = row['a_stop_id'], row['b_stop_id'], row['walk_time_sec']
            walk.setdefault(a, []).append({'next_stop': b, 'walk_time': wt})
            walk.setdefault(b, []).append({'next_stop': a, 'walk_time': wt})
        return walk


def _reconstruct_raw(prev, dist, start_id, target_id):
    """
    Walk the forward back-pointer chain target → start, then reverse.

    Raw per-hop format: (dep_time, from_stop, trip_or_walk, edge_type, arr_time, to_stop)
    """
    if target_id not in prev:
        return []
    raw = []
    cur = target_id
    while cur != start_id:
        prev_stop, trip_or_walk, dep_time, edge_type = prev[cur]
        arr_time = dist[cur]
        raw.append((int(dep_time), int(prev_stop), trip_or_walk, edge_type,
                    int(arr_time), int(cur)))
        cur = prev_stop
    raw.reverse()

    return raw



def _reconstruct_raw_backward(path_after, start_id, end_id):
    """
    Walk the backward forward-pointer chain start → end. No reversal needed —
    path_after is already forward-oriented.
    """
    if start_id not in path_after:
        return []
    raw = []
    cur = start_id
    while cur != end_id:
        next_stop, trip_or_walk, dep_time, arr_time, edge_type = path_after[cur]
        raw.append((int(dep_time), int(cur), trip_or_walk, edge_type,
                    int(arr_time), int(next_stop)))
        cur = next_stop
    return raw


def _raw_to_events(raw):
    """
    Compress a raw edge list into the event-tuple format the downstream
    pipeline (path_confidence etc.) consumes:

        (dep_ts, board_stop, trip_id, None)       board
        (arr_ts, None,       trip_id, alight_stop) alight
        (arr_ts, from_stop,  None,    to_stop)     walk

    Consecutive hops on the same trip_id are collapsed into a single
    BOARD/ALIGHT pair.
    """
    events = []
    i = 0
    while i < len(raw):
        dep_time, from_stop, trip_or_walk, edge_type, arr_time, to_stop = raw[i]
        if edge_type == 'transit':
            j = i
            while (j + 1 < len(raw)
                    and raw[j+1][2] == trip_or_walk
                    and raw[j+1][3] == 'transit'):
                j += 1
            events.append((dep_time,    from_stop, trip_or_walk, None))
            events.append((raw[j][4],   None,      trip_or_walk, raw[j][5]))
            i = j + 1
        #else:
        #    events.append((arr_time, from_stop, None, to_stop)) ############### debugging here
        #    i += 1
        else:
            walk_time = arr_time - dep_time
            events.append((arr_time, from_stop, None, to_stop, walk_time))
            i += 1
    return events


def _raw_sig(raw):
    """Hashable dedup key for a raw path: (from_stop, trip_or_walk, dep_time) per hop."""
    return tuple((e[1], e[2], e[0]) for e in raw)


def _reconstruct_single(prev, dist, start_id, target_id):
    if target_id not in prev:
        print("The targeted node is unreachable from this starting node.")
        return []
    return _raw_to_events(_reconstruct_raw(prev, dist, start_id, target_id))


def _path_to_df(path: list) -> pd.DataFrame:
    """Convert a dijkstra path (list of (stop, time) nodes) to a DataFrame."""
    return pd.DataFrame(path, columns=['stop_id', 'time_sec'])

def _add_edges(G: nx.DiGraph, edges_df: pd.DataFrame) -> None:
    ebunch = zip(
        zip(edges_df['from_stop'], edges_df['from_time']),
        zip(edges_df['to_stop'],   edges_df['to_time']),
        edges_df[['weight', 'edge_type', 'trip_id']].to_dict('records')
    )
    G.add_edges_from(ebunch)

def _path_to_edges( G: nx.DiGraph, path: list[tuple]) -> list[tuple]:
    result = []
    for u, v in zip(path, path[1:]):
        attr = G[u][v]
        if attr['edge_type'] == 'ride':
            result.append((
                u[1],
                u[0],
                attr.get('trip_id'),
                None,
            ))
            result.append((
                v[1],
                None,
                attr.get('trip_id'),
                v[0],
            ))
        else:
            result.append((
                v[1],
                u[0],
                None,
                v[0],
            ))
    return result