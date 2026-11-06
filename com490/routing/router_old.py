import heapq
import pandas as pd

from db import fetch_async, execute
from db import RIDE_EDGES, DWELL_EDGES, CREATE_RELEVANT_TRIPS ,SOURCE_EDGES, SINK_EDGES, WALK_EDGES, TRANSIT_QUERY, WALK_QUERY
from routing.time_utils import time_to_seconds, seconds_to_time, day_to_column

WALK_SPEED_M_PER_SEC = 50 / 60


# +
#### here the prefered method is using route_dynamic, which uses djikstras algorithm on a time expanded graph

class Router: 
    def __init__(self, conn, userns: str, sharedns: str, query_window_size: int = 16000):
        self.conn    = conn
        self.userns  = userns
        self.sharedns = sharedns
        self.query_window_size = query_window_size


    def route_dynamic(self, start_id, departs, day, end_id=None, max_walk_m=100, id_to_name=None): ## prefered method
        """
            Loads data, runs Dijkstra, reconstructs routes, and returns:
                dist      : stop_id -> earliest arrival (seconds)
                prev      : raw prev-pointer dict
                route     : list of tuples (single target) or dict of lists (all stops)
                results_df: summary DataFrame sorted by earliest arrival
        """
        query_time_sec = time_to_seconds(departs)
        day_col    = day_to_column(day)

        df_transit, df_walk = self.build_time_expanded_graph( query_time_sec, day_col, max_walk_m=max_walk_m)
        dist, prev = self.earliest_arrival_dijkstra(df_transit, df_walk, start_id, query_time_sec, end_id)

        route = self.reconstruct_route(prev, dist, start_id, end_id)

        results = {name: t for s, t in dist.items() if (name := id_to_name.get(s)) is not None}
        results_df = pd.DataFrame(list(results.items()), columns=['stop_name', 'earliest_arrival_sec'])
        results_df['earliest_arrival_hhmm'] = results_df['earliest_arrival_sec'].apply(lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}")
        results_df = results_df.sort_values('earliest_arrival_sec').reset_index(drop=True)
        results_df['travel_time_sec'] = results_df['earliest_arrival_sec'] - query_time_sec
        results_df['travel_time_hhmm'] = results_df['travel_time_sec'].apply(lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}")

        return dist, prev, route, results_df

    def route_dynamic_k(self, start_id, departs, day, end_id, max_walk_m=100, k=10):
        ## CHANGED: Replaced Yen's algorithm with a faster trip-blocking heuristic.
        ## Yen's was too costly (O(k × path_length) Dijkstra runs, each rebuilding the graph).
        ##
        ## New approach — BFS over trip IDs:
        ##   1. Find the best path, collect its trip IDs.
        ##   2. Block each trip ID one at a time, rerun Dijkstra, collect any new distinct path.
        ##   3. Queue each new path found and repeat from it.
        ## Each trip ID is blocked at most once across the whole search.
        ## Total runs ≈ unique trips encountered — typically 10–20 for k=10.

        query_time_sec = time_to_seconds(departs)
        day_col = day_to_column(day)

        df_transit, df_walk = self.build_time_expanded_graph(query_time_sec, day_col, max_walk_m=max_walk_m)

        ## Find the unrestricted best path first
        dist, prev = self.earliest_arrival_dijkstra(df_transit, df_walk, start_id, query_time_sec, end_id)
        first_raw = _reconstruct_raw(prev, dist, start_id, end_id)
        if not first_raw:
            return []

        candidates = [_raw_to_events(first_raw)]
        seen_sigs  = {_raw_sig(first_raw)}
        queue      = [first_raw]   ## raw paths whose trips still need to be explored
        explored   = set()         ## trip IDs already blocked in a standalone run

        while queue and len(candidates) < k:
            path_raw = queue.pop(0)

            ## Unique trips in this path not yet tried as a blocking candidate
            path_trips = list(dict.fromkeys(
                e[2] for e in path_raw if e[3] == 'transit' and e[2] not in explored
            ))

            for trip_id in path_trips:
                if len(candidates) >= k:
                    break

                explored.add(trip_id)

                ## Remove this trip entirely from the graph and rerun from start
                df_filtered = df_transit[~df_transit['trip_id'].isin({trip_id})]
                dist, prev  = self.earliest_arrival_dijkstra(df_filtered, df_walk, start_id, query_time_sec, end_id)
                raw         = _reconstruct_raw(prev, dist, start_id, end_id)
                if not raw:
                    continue

                sig = _raw_sig(raw)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    candidates.append(_raw_to_events(raw))
                    queue.append(raw)  ## explore alternatives from this path too

        return candidates
    
    def build_time_expanded_graph(self, query_time_sec, day_col, window_sec=7400, max_walk_m=100):
    
        """
        Returns:
            df_transit : DataFrame of transit stop_times for relevant trips
            df_walk    : DataFrame of walkable stop pairs with walk_time_sec column
        """

        f_transit     = fetch_async(self.conn, TRANSIT_QUERY,   
                                    userns=self.userns, 
                                    query_time=query_time_sec, 
                                    day=day_col, 
                                    window_end=query_time_sec + window_sec
                        )
        f_walk        = fetch_async(self.conn, WALK_QUERY,      userns=self.userns, max_walk_m=max_walk_m)
        df_transit    = f_transit.result()
        df_walk       = f_walk.result()

        df_walk['walk_time_sec'] = (df_walk['distance'] / WALK_SPEED_M_PER_SEC).round().astype(int)

        return df_transit, df_walk

    def earliest_arrival_dijkstra(self, df_transit, df_walk, stop_id_start, query_time,
                                   target_stop_id=None):
        """
        Returns:
            dist : dict  stop_id -> earliest arrival time (seconds)
            prev : dict  stop_id -> (prev_stop, trip_id_or_'walk', dep_time, edge_type)
        """
        TRANSFER_BUFFER_SEC = 60  # minimum seconds between alighting one trip and boarding another

        ### Transit graph (unidirectional)
        graph = {}
        df_sorted = df_transit.sort_values(['trip_id', 'stop_sequence']) # MM: allready sorted
        df_sorted = df_sorted.copy()
        df_sorted['next_stop_id']  = df_sorted.groupby('trip_id')['stop_id'].shift(-1)
        df_sorted['next_arrival']  = df_sorted.groupby('trip_id')['arrival_time'].shift(-1)
        df_edges = df_sorted.dropna(subset=['next_stop_id'])

        for _, row in df_edges.iterrows():
            graph.setdefault(row['stop_id'], []).append({
                'trip_id':   row['trip_id'],
                'next_stop': row['next_stop_id'],
                'dep_time':  row['departure_time'],
                'arr_time':  row['next_arrival'],
            })

        for stop in graph:
            graph[stop].sort(key=lambda e: e['dep_time']) # sort by time for faster results

        ### Walking graph (bidirectional)
        walk_graph = {}
        for _, row in df_walk.iterrows():
            a, b, wt = row['a_stop_id'], row['b_stop_id'], row['walk_time_sec']
            walk_graph.setdefault(a, []).append({'next_stop': b, 'walk_time': wt})
            walk_graph.setdefault(b, []).append({'next_stop': a, 'walk_time': wt})

        ### Dijkstra
        best = {stop_id_start: query_time}   # best known arrival per stop -> stops putting nodes in queue if no need
        dist = {} # processed stops # MM: here this is more a times_at_nodes than a dist.
        prev = {} # for path reconstruction
        pq   = [(query_time, stop_id_start)]

        while pq:
            t, stop = heapq.heappop(pq) # MM: select the smallest one in pq

            if stop in dist:
                continue # already settled with a better time

            dist[stop] = t

            if target_stop_id and stop == target_stop_id: # Reached single given target
                break

            for edge in graph.get(stop, []): ## PT edges
                if edge['dep_time'] < t:
                    continue

                # enforce buffer only on transit-to-transit transfers; walk arrivals board immediately
                if (stop in prev
                        and prev[stop][3] == 'transit'
                        and edge['dep_time'] < t + TRANSFER_BUFFER_SEC
                        and edge['trip_id'] != prev[stop][1]):
                    continue

                nxt = edge['next_stop']
                arr = edge['arr_time']
                if arr < best.get(nxt, float('inf')):
                    best[nxt] = arr
                    prev[nxt] = (stop, edge['trip_id'], edge['dep_time'], 'transit') # formatted as asked
                    heapq.heappush(pq, (arr, nxt))

            for edge in walk_graph.get(stop, []): ## Walking edges
                nxt = edge['next_stop']
                arr = t + edge['walk_time']

                if arr < best.get(nxt, float('inf')):
                    best[nxt] = arr
                    prev[nxt] = (stop, 'walk', t, 'walk') # formated as asked
                    heapq.heappush(pq, (arr, nxt))

        return dist, prev

    
    def reconstruct_route(self, prev, dist, start_id, target_stop_id=None): 
        
        """
        Reconstructs the chronologically ordered list of route tuples as specified:

            (ts, stop, trip_id, None)   board trip_id at stop at time ts
            (ts, None, trip_id,stop) arrive at stop via trip_id at time ts
            (ts, stop_prev, None, stop)   reach stop by walking from stop_prev at ts
        """
        
        if target_stop_id is None:
            
            # All-stops mode: reconstruct a route per reachable stop
            routes = {}
            for stop in dist:
                if stop == start_id:
                    continue
                routes[int(stop)] = _reconstruct_single(prev, dist, start_id, stop)
            return routes
        else:
            return _reconstruct_single(prev, dist, start_id, target_stop_id)

# -

def _reconstruct_raw(prev, dist, start_id, target_id):
    # Returns the raw per-hop edge list used for dedup and path building in route_dynamic_k.
    # Format per entry: (dep_time, from_stop, trip_or_walk, edge_type, arr_time, to_stop)
    if target_id not in prev:
        return []
    raw = []
    cur = target_id
    while cur != start_id:
        prev_stop, trip_or_walk, dep_time, edge_type = prev[cur]
        arr_time = dist[cur]
        raw.append((int(dep_time), int(prev_stop), trip_or_walk, edge_type, int(arr_time), int(cur)))
        cur = prev_stop
    raw.reverse()
    return raw


def _raw_to_events(raw):
    # Compresses a raw edge list into the event-tuple format the pipeline consumes.
    events = []
    i = 0
    while i < len(raw):
        dep_time, from_stop, trip_or_walk, edge_type, arr_time, to_stop = raw[i]
        if edge_type == 'transit':
            j = i
            while j + 1 < len(raw) and raw[j+1][2] == trip_or_walk and raw[j+1][3] == 'transit':
                j += 1
            events.append((dep_time, from_stop, trip_or_walk, None))
            events.append((raw[j][4], None, trip_or_walk, raw[j][5]))
            i = j + 1
        else:
            events.append((arr_time, from_stop, None, to_stop))
            i += 1
    return events


def _raw_sig(raw):
    # Hashable dedup key: (from_stop, trip_or_walk, dep_time) per hop.
    return tuple((e[1], e[2], e[0]) for e in raw)


def _reconstruct_single(prev, dist, start_id, target_id):
    if target_id not in prev:
        print("The targeted node is unreachable from this starting node.")
        return []
    return _raw_to_events(_reconstruct_raw(prev, dist, start_id, target_id))


def _path_to_df(path: list) -> pd.DataFrame:
    """Convert a dijkstra path (list of (stop, time) nodes) to a DataFrame."""
    return pd.DataFrame(path, columns=['stop_id', 'time_sec'])










