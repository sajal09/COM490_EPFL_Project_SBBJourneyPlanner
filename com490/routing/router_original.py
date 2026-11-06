import heapq
import networkx as nx
import pandas as pd
from typing import Optional

from db import fetch_async, execute
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

    def route_dynamic_k(self, start_id, departs, day, end_id, max_walk_m=100, k=5):
        query_time_sec = time_to_seconds(departs)
        day_col = day_to_column(day)
    
        df_transit, df_walk = self.build_time_expanded_graph(query_time_sec, day_col, max_walk_m=max_walk_m)
    
        candidates = []
        seen_sigs = set()
        blocked_trips = set()
    
        for _ in range(k + 5):
            if len(candidates) >= k:
                break
    
            df_filtered = df_transit[~df_transit['trip_id'].isin(blocked_trips)]
            dist, prev = self.earliest_arrival_dijkstra(df_filtered, df_walk, start_id, query_time_sec, end_id)
            route = self.reconstruct_route(prev, dist, start_id, end_id)
    
            if not route:
                break
    
            sig = tuple(e[2] for e in route if e[2] is not None)
            if not sig or sig in seen_sigs:
                break
    
            seen_sigs.add(sig)
            candidates.append(route)
    
            first_trip = next((e[2] for e in route if e[2] is not None), None)
            if first_trip is None:
                break
            blocked_trips.add(first_trip)
    
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

    def earliest_arrival_dijkstra(self, df_transit, df_walk, stop_id_start, query_time, target_stop_id=None):
        """
        Returns:
            dist : dict  stop_id -> earliest arrival time (seconds)
            prev : dict  stop_id -> (prev_stop, trip_id_or_'walk', dep_time, edge_type)
        """

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

                # MM: penalize transfer
                if stop in prev and edge['dep_time']==t and edge['trip_id'] != prev[stop][1]:
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

def _add_edges(G: nx.DiGraph, edges_df: pd.DataFrame) -> None:
    ebunch = zip(
        zip(edges_df['from_stop'], edges_df['from_time']),
        zip(edges_df['to_stop'],   edges_df['to_time']),
        edges_df[['weight', 'edge_type', 'trip_id']].to_dict('records')
    )
    G.add_edges_from(ebunch)


def _path_to_df(path: list) -> pd.DataFrame:
    """Convert a dijkstra path (list of (stop, time) nodes) to a DataFrame."""
    return pd.DataFrame(path, columns=['stop_id', 'time_sec'])










