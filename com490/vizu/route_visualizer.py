# I get p, br from path_confidence


# title: from A to B, arrival before T

#  interactive map 
#  (take from assignment 1)
# tick boxes to see different paths (each path -> different color)
# each path is define by: departure time, arrival time, confidence, number of transaction, walking distance/time
# - - - - : for walking
#  _______ : for buses
# o : between different trip
# edge's hoverbox: 
# - trip id (select only bus lane name)
# - from last transaction node id to next transaction node id
# - trip length (time)
#  node's hoverbox: arrival from trip id at xx:xx, departure to trip id at xx:xx, with confidence xx%

# overview of each trip 
# list of paths available
# select one to display 

# # IMPORTS

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# # TIME HELPERS

def format_time(seconds):
    if seconds is None or pd.isna(seconds):
        return "N/A"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_time(timestr):
    h, m = map(int, timestr.split(":"))
    return h * 3600 + m * 60


# # TRANSPORT MODE

def infer_transport_mode(line_name: str) -> str:
    """
    Infer the transport mode label from a SBB/TL line name.

    Rules (order matters — first match wins):
      • Starts with 'IC', 'IR', 'RE', 'R', 'S' followed by digits → Train
      • Starts with 'M' followed by digit (M1, M2) → Metro
      • Starts with 'NL', 'N' followed by digit → Night Bus
      • Pure digits or short alphanumeric (1–3 chars) → Bus
      • 'T' + digit (T1, T78 …) → Tram
      • Everything else → Bus (safe default)
    """
    if not line_name or line_name in ('Walking', 'walk', ''):
        return 'Walk'

    import re
    s = str(line_name).strip()

    if re.match(r'^(IC|IR|RE|TGV|EC|RB|RJX?)', s, re.IGNORECASE):
        return 'Train'
    if re.match(r'^R\d', s, re.IGNORECASE):
        return 'Train'
    if re.match(r'^S\d', s, re.IGNORECASE):
        return 'Train'   # S-Bahn
    if re.match(r'^M\d', s, re.IGNORECASE):
        return 'Metro'
    if re.match(r'^T\d', s, re.IGNORECASE):
        return 'Tram'
    if re.match(r'^(NL|N)\d', s, re.IGNORECASE):
        return 'Night Bus'
    # Numeric or short alphanumeric → Bus
    if re.match(r'^\d+[A-Za-z]?$', s):
        return 'Bus'
    return 'Bus'


MODE_ICON = {
    'Train':     '🚆',
    'Metro':     '🚇',
    'Tram':      '🚊',
    'Bus':       '🚌',
    'Night Bus': '🌙',
    'Walk':      '🚶',
}


def line_label(line_name: str) -> str:
    """Return 'Bus 10', 'Metro M2', 'Train IC5', etc."""
    mode = infer_transport_mode(line_name)
    icon = MODE_ICON.get(mode, '')
    return f"{icon} {mode} {line_name}".strip()


# # ROUTE SUMMARY

def build_summary(candidates,
                  path_confidence_fn,
                  stop_delays,
                  travel_times,
                  trip_to_line,
                  day_to_dow,
                  day,
                  query_time_s,
                  deadline_s,
                  bus_state_fn,
                  q_threshold=0.80,
                  rng=None):

    rng = rng or np.random.default_rng(42)

    rows = []

    for rid, (depart_str, evs) in enumerate(candidates):

        p, br = path_confidence_fn(
            events=evs,
            stop_delays=stop_delays,
            travel_times=travel_times,
            trip_to_line=trip_to_line,
            dow=day_to_dow(day),
            query_time_s=query_time_s,
            deadline_s=deadline_s,
            buffer_s=0,
            bus_state_fn=bus_state_fn,
            n_samples=10000,
            rng=rng,
        )

        arr_ts = next(
            (b['sched_arr_ts'] for b in br if b['type'] == 'arrival'),
            evs[-1][0] if evs else None
        )

        walk_s = sum(
            b.get('walk_s', 0)
            for b in br
            if b['type'] == 'transfer'
        )

        rows.append({
            'route_id': rid,
            'depart_str': depart_str,
            'arr_ts': arr_ts,
            'n_transfers': sum(1 for b in br if b['type'] == 'transfer'),
            'overall_p': p,
            'walk_s': walk_s,
            'travel_s': arr_ts - parse_time(depart_str),
            'meets_Q': p >= q_threshold,
            'events': evs,
            'breakdown': br
        })

    return pd.DataFrame(rows)


# # EDGE BUILDING

def build_edges_df(route_events):

    edges = []
    i = 0

    while i < len(route_events) - 1:
        curr = route_events[i]
        time_c, from_c, trip_c, to_c = curr[:4]

        # --- RIDE EDGE ---
        if trip_c is not None and from_c is not None and to_c is None:
            nxt = route_events[i + 1]
            time_n, from_n, trip_n, to_n = nxt[:4]

            if trip_n == trip_c and from_n is None and to_n is not None:
                edges.append({
                    'from_stop': int(from_c),
                    'to_stop': int(to_n),
                    'trip_id': trip_c,
                    'departure_time': float(time_c),
                    'arrival_time': float(time_n),
                    'type': 'ride'
                })
                i += 2
                continue

        # --- WALK EDGE ---
        if trip_c is None and from_c is not None and to_c is not None:
            # Look ahead for the next event's timestamp to get walk arrival time
            nxt = route_events[i + 1] if i + 1 < len(route_events) else None
            walk_arr = float(route_events[i + 1][0]) if nxt is not None else float(time_c)
            walk_duration = curr[4] # MM check 
            
            edges.append({
                'from_stop': int(from_c),
                'to_stop': int(to_c),
                'trip_id': 'walk',
                'departure_time': float(time_c),
                'arrival_time': walk_arr,
                'walk_s': walk_duration,
                'type': 'walk'
            })

        i += 1

    return pd.DataFrame(edges)


def enrich_edges(edges_df, stops, trip_to_line):

    if edges_df.empty:
        return edges_df

    # Coordinates
    edges_df = edges_df.merge(
        stops[['stop_id', 'stop_lat', 'stop_lon']],
        left_on='from_stop',
        right_on='stop_id'
    ).rename(columns={
        'stop_lat': 'lat_from',
        'stop_lon': 'lon_from'
    }).drop(columns='stop_id')

    edges_df = edges_df.merge(
        stops[['stop_id', 'stop_lat', 'stop_lon']],
        left_on='to_stop',
        right_on='stop_id'
    ).rename(columns={
        'stop_lat': 'lat_to',
        'stop_lon': 'lon_to'
    }).drop(columns='stop_id')

    # Names
    edges_df = edges_df.merge(
        stops[['stop_id', 'stop_name']],
        left_on='from_stop',
        right_on='stop_id'
    ).rename(columns={'stop_name': 'from_name'}).drop(columns='stop_id')

    edges_df = edges_df.merge(
        stops[['stop_id', 'stop_name']],
        left_on='to_stop',
        right_on='stop_id'
    ).rename(columns={'stop_name': 'to_name'}).drop(columns='stop_id')

    # Bus line names + transport mode
    edges_df['line_name'] = edges_df['trip_id'].map(trip_to_line)
    edges_df.loc[edges_df['type'] == 'walk', 'line_name'] = 'Walking'
    edges_df['mode'] = edges_df['line_name'].apply(infer_transport_mode)
    edges_df['line_label'] = edges_df['line_name'].apply(line_label)

    return edges_df


# # ROUTE DETAILS DISPLAY

def display_details_selected_route(
    summary,
    selected_routes,
    stops,
    trip_to_line
):

    if len(selected_routes) == 0:
        display(HTML("<b>No route selected</b>"))
        return

    selected = summary[
        summary.route_id == selected_routes[0]
    ].iloc[0]

    edges_df = build_edges_df(selected['events'])
    edges_df = enrich_edges(edges_df, stops, trip_to_line)

    transfers = [
        b for b in selected['breakdown']
        if b['type'] == 'transfer'
    ]

    ride_edges = edges_df[
        edges_df['type'] == 'ride'
    ]

    if ride_edges.empty:
        display(HTML("<b>No transit rides found</b>"))
        return

    first_ride = ride_edges.iloc[0]
    last_ride = ride_edges.iloc[-1]

    html = ""

    # =========================================================
    # Header
    # =========================================================

    html += f"""
    <div style="
        border:1px solid #ddd;
        border-radius:10px;
        padding:20px;
        margin-top:15px;
        background:#fafafa;
        font-family:Arial;
    ">

    <h2 style="margin-top:0;">
        Journey details
    </h2>

    <div style="
        display:flex;
        gap:30px;
        margin-bottom:20px;
        flex-wrap:wrap;
    ">

        <div>
            <b>Departure</b><br>
            {first_ride['from_name']}<br>
            {format_time(first_ride['departure_time'])}
        </div>

        <div>
            <b>Arrival</b><br>
            {last_ride['to_name']}<br>
            {format_time(last_ride['arrival_time'])}
        </div>

        <div>
            <b>Travel time</b><br>
            {format_time(selected.travel_s)}
        </div>

        <div>
            <b>Walking</b><br>
            {format_time(selected.walk_s)}
        </div>

        <div>
            <b>Transfers</b><br>
            {selected.n_transfers}
        </div>

        <div>
            <b>Confidence</b><br>
            {selected.overall_p:.1%}
        </div>

    </div>
    """

    # =========================================================
    # Journey steps
    # =========================================================

    transfer_idx = 0

    for _, edge in edges_df.iterrows():

        # -----------------------------------------------------
        # Ride segment
        # -----------------------------------------------------

        if edge['type'] == 'ride':

            duration = (
                edge['arrival_time']
                - edge['departure_time']
            )

            html += f"""
            <div style="
                border-left:6px solid #4CAF50;
                background:white;
                padding:15px;
                margin-bottom:10px;
                border-radius:8px;
                box-shadow:0 1px 3px rgba(0,0,0,0.1);
            ">

                <div style="
                    font-size:18px;
                    font-weight:bold;
                    margin-bottom:10px;
                    color:#2c3e50;
                ">
                    {edge['line_label']}
                </div>

                <table style="
                    width:100%;
                    border-collapse:collapse;
                ">
                    <tr>
                        <td><b>Board</b></td>
                        <td>{edge['from_name']}</td>
                        <td>{format_time(edge['departure_time'])}</td>
                    </tr>

                    <tr>
                        <td><b>Alight</b></td>
                        <td>{edge['to_name']}</td>
                        <td>{format_time(edge['arrival_time'])}</td>
                    </tr>

                    <tr>
                        <td><b>Duration</b></td>
                        <td colspan="2">
                            {format_time(duration)}
                        </td>
                    </tr>
                </table>
            </div>
            """

        # -----------------------------------------------------
        # Walking segment
        # -----------------------------------------------------

        elif edge['type'] == 'walk':

            walk_dur = edge['walk_s'] 

            html += f"""
            <div style="
                border-left:6px solid gray;
                background:#f2f2f2;
                padding:12px;
                margin-bottom:10px;
                border-radius:8px;
            ">
                🚶 Walk from
                <b>{edge['from_name']}</b>
                to
                <b>{edge['to_name']}</b>
                &nbsp;·&nbsp; {format_time(walk_dur)}
            </div>
            """

        # -----------------------------------------------------
        # Transfer block
        # -----------------------------------------------------

        if transfer_idx < len(transfers):

            transfer = transfers[transfer_idx]

            if (
                edge['type'] == 'ride'
                and edge['to_stop']
                == transfer['alight_stop']
            ):

                wait_time = (
                    transfer['board_ts']
                    - transfer['alight_ts']
                )

                stop_info = stops[
                    stops.stop_id
                    == transfer['alight_stop']
                ]

                stop_name = (
                    stop_info.iloc[0]['stop_name']
                    if not stop_info.empty
                    else "Unknown stop"
                )

                html += f"""
                <div style="
                    margin-left:25px;
                    margin-bottom:15px;
                    padding:12px;
                    border-left:4px dashed #999;
                    background:#fff8e1;
                    border-radius:6px;
                ">

                    <div style="
                        font-weight:bold;
                        margin-bottom:8px;
                        color:#e67e22;
                    ">
                        Transfer
                    </div>

                    <div>
                        <b>Stop:</b> {stop_name}
                    </div>

                    <div>
                        <b>Incoming:</b>
                        {line_label(transfer['incoming_line'])}
                        ({format_time(transfer['alight_ts'])})
                    </div>

                    <div>
                        <b>Outgoing:</b>
                        {line_label(transfer['outgoing_line'])}
                        ({format_time(transfer['board_ts'])})
                    </div>

                    <div>
                        <b>Waiting time:</b>
                        {format_time(wait_time)}
                    </div>

                    <div>
                        <b>Walking time:</b>
                        {format_time(transfer.get('walk_s', 0))}
                    </div>

                    <div>
                        <b>Confidence:</b>
                        {transfer['p']:.1%}
                    </div>

                </div>
                """

                transfer_idx += 1

    html += "</div>"

    display(HTML(html))


# # MAP DISPLAY

def display_candidate_routes(stops, summary, selected_routes, trip_to_line):

    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for k, route_id in enumerate(selected_routes):

        row = summary[summary.route_id == route_id].iloc[0]
        edges_df = build_edges_df(row['events'])
        edges_df = enrich_edges(edges_df, stops, trip_to_line)

        if edges_df.empty:
            continue

        color = colors[k % len(colors)]

        # Edges
        for _, edge in edges_df.iterrows():

            is_walk = edge['type'] == 'walk'
            edge_color = 'gray' if is_walk else color
            walk_dur = edge['walk_s']

            if is_walk:
                hover = (
                    f"<b>🚶 Walking</b><br>"
                    f"From: {edge['from_name']}<br>"
                    f"To: {edge['to_name']}<br>"
                    f"Walking time: {format_time(walk_dur)}"
                )
            else:
                trip_dur = edge['arrival_time'] - edge['departure_time']
                hover = (
                    f"<b>{edge['line_label']}</b><br>"
                    f"From: {edge['from_name']}<br>"
                    f"To: {edge['to_name']}<br>"
                    f"Trip time: {format_time(trip_dur)}"
                )

            edge_label = edge['line_label'] if not is_walk else '🚶'

            fig.add_trace(go.Scattermapbox(
                lat=[edge['lat_from'], edge['lat_to']],
                lon=[edge['lon_from'], edge['lon_to']],
                mode='lines',
                line=dict(
                    width=4 if not is_walk else 2,
                    color=edge_color
                ),
                hoverinfo='skip',
                showlegend=False
            ))

            # midpoint hover — marker colour matches edge colour
            fig.add_trace(go.Scattermapbox(
                lat=[(edge['lat_from'] + edge['lat_to']) / 2],
                lon=[(edge['lon_from'] + edge['lon_to']) / 2],
                mode='markers+text',
                marker=dict(size=1, color=edge_color, opacity=0.85),
                text=edge_label,
                textposition="top center",
                hoverinfo='text',
                hovertext=hover,
                showlegend=False
            ))

            if is_walk:
                fig.add_trace(go.Scattermapbox(
                    lat=[edge['lat_to']],
                    lon=[edge['lon_to']],
                    mode='markers',
                    marker=dict(size=8, color='gray'),
                    hoverinfo='text',
                    hovertext=f"<b>Walk arrival</b><br>{edge['to_name']}<br>{format_time(edge['arrival_time'])}",
                    showlegend=False
                ))

        # Transfer nodes
        transfers = [
            b for b in row['breakdown']
            if b['type'] == 'transfer'
        ]
        
        node_lats = []
        node_lons = []
        node_texts = []
        node_labels = []
        
        for b in transfers:

            incoming_label = line_label(b['incoming_line'])
            outgoing_label = line_label(b['outgoing_line'])

            walk_s = b.get('walk_s', 0)

            if walk_s == 0:
            
                stop_info = stops[stops.stop_id == b['alight_stop']]
            
                if stop_info.empty:
                    continue
            
                stop_row = stop_info.iloc[0]
            
                lat = stop_row['stop_lat']
                lon = stop_row['stop_lon']
                stop_name = stop_row['stop_name']
            
                node_lats.append(lat)
                node_lons.append(lon)
            
                hover_text = (
                    f"<b>Transfer</b><br>"
                    f"Stop: {stop_name}<br>"
                    f"Arrival from {incoming_label} "
                    f"at {format_time(b['alight_ts'])}<br>"
                    f"Departure to {outgoing_label} "
                    f"at {format_time(b['board_ts'])}<br>"
                    f"Wait: {format_time(b['board_ts'] - b['alight_ts'])}<br>"
                    f"Confidence: {b['p']:.1%}"
                )
                
                node_lats.append(lat)
                node_lons.append(lon)
                node_texts.append(hover_text)
                node_labels.append(f"{stop_name}")

            else: # walking transfer

                # first node
                out_info = stops[stops.stop_id == b['alight_stop']]
        
                if not out_info.empty:
        
                    out_row = out_info.iloc[0]
        
                    hover_text = (
                        f"<b>Transfer</b><br>"
                        f"Stop: {out_row['stop_name']}<br>"
                        f"Arrival from {incoming_label} "
                        f"at {format_time(b['alight_ts'])}<br>"
                        f"Walk begins<br>"
                        f"Walk: {format_time(walk_s)}<br>"
                        f"Confidence: {b['p']:.1%}"
                    )
        
                    node_lats.append(out_row['stop_lat'])
                    node_lons.append(out_row['stop_lon'])
                    node_texts.append(hover_text)
                    node_labels.append(out_row['stop_name'])
        
                # second node
                in_info = stops[stops.stop_id == b['board_stop']]
        
                if not in_info.empty:
        
                    in_row = in_info.iloc[0]
        
                    hover_text = (
                        f"<b>Transfer</b><br>"
                        f"Stop: {in_row['stop_name']}<br>"
                        f"Departure to {outgoing_label} "
                        f"at {format_time(b['board_ts'])}<br>"
                        f"Walk: {format_time(walk_s)}<br>"
                        f"Confidence: {b['p']:.1%}"
                    )
        
                    node_lats.append(in_row['stop_lat'])
                    node_lons.append(in_row['stop_lon'])
                    node_texts.append(hover_text)
                    node_labels.append(in_row['stop_name'])
        
        if node_lats:
            fig.add_trace(go.Scattermapbox(
                lat=node_lats,
                lon=node_lons,
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=color
                ),
                text=node_labels,
                textposition="top center",
                hoverinfo='text',
                hovertext=node_texts,
                showlegend=False
            ))
            
        # Start and arrival nodes
        
        # first/last RIDE edge only
        first_edge = edges_df.iloc[0]
        last_edge = edges_df.iloc[-1]
        
        ride_edges = edges_df[edges_df['type'] == 'ride']
        
        first_ride = ride_edges.iloc[0]
        last_ride = ride_edges.iloc[-1]
        
        start_lat = first_edge['lat_from']
        start_lon = first_edge['lon_from']
        
        arrival_lat = last_ride['lat_to']
        arrival_lon = last_ride['lon_to']
        
        start_name = first_edge['from_name']
        arrival_name = last_ride['to_name']
        
        if first_edge['type'] == 'walk':
            outgoing_label = "🚶 Walking"
            dep_time = first_edge['departure_time']
        else:
            outgoing_label = first_ride['line_label']
            dep_time = first_ride['departure_time']
        incoming_label = last_ride['line_label']

        start_hover = (
            f"<b>Start</b><br>"
            f"Stop: {start_name}<br>"
            f"Departure on {outgoing_label}<br>"
            f"at {format_time(dep_time)}"
        )

        arrival_hover = (
            f"<b>Arrival</b><br>"
            f"Stop: {arrival_name}<br>"
            f"Arrival from {incoming_label}<br>"
            f"at {format_time(last_ride['arrival_time'])}"
        )
        
        fig.add_trace(go.Scattermapbox(
            lat=[start_lat],
            lon=[start_lon],
            mode='markers+text',
            marker=dict(
                size=10,
                color=color
            ),
            text=[
                f"{start_name}"
            ],
            textposition="top center",
            hoverinfo='text',
            hovertext=[start_hover],
            showlegend=False
        ))
        
        fig.add_trace(go.Scattermapbox(
            lat=[arrival_lat],
            lon=[arrival_lon],
            mode='markers+text',
            marker=dict(
                size=10,
                color=color
            ),
            text=[
                f"{arrival_name}"
            ],
            textposition="top center",
            hoverinfo='text',
            hovertext=[arrival_hover],
            showlegend=False
        ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(
                lat=stops['stop_lat'].mean(),
                lon=stops['stop_lon'].mean()
            ),
            zoom=11.5
        ),
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0}
    )

    fig.show(config={"scrollZoom": True})


# # MAIN UI

def launch_route_ui(stops, summary, trip_to_line,
                    origin_name: str = "",
                    destination_name: str = "",
                    deadline: str = ""):

    output = widgets.Output()

    # ── Title ────────────────────────────────────────────────────────────
    if origin_name and destination_name and deadline:
        title_html = (
            f"<h2>From <b>{origin_name}</b> to <b>{destination_name}</b>"
            f" — arrival before <b>{deadline}</b></h2>"
        )
    elif origin_name and destination_name:
        title_html = (
            f"<h2>From <b>{origin_name}</b> to <b>{destination_name}</b></h2>"
        )
    else:
        title_html = "<h2>Journey Planner</h2>"

    route_selector = widgets.SelectMultiple(
        options=[
            (
                f"Dep {r.depart_str} | "
                f"Arr {format_time(r.arr_ts)[:-3]} | "
                f"P={r.overall_p:.1%} | "
                f"transfers={r.n_transfers} | "
                f"walking={format_time(r.walk_s)}",
                r.route_id
            )
            for _, r in summary.iterrows()
        ],
        value=[summary.iloc[0].route_id],
        description='Routes:',
        layout=widgets.Layout(width='600px', height='200px')
    )

    def refresh(change=None):
        with output:
            clear_output()

            display_candidate_routes(
                stops=stops,
                summary=summary,
                selected_routes=route_selector.value,
                trip_to_line=trip_to_line
            )

            display_details_selected_route(
                summary,
                route_selector.value,
                stops,
                trip_to_line
            )

    route_selector.observe(refresh, names='value')

    ui = widgets.VBox([
        widgets.HTML(title_html),
        route_selector,
        output
    ])

    display(ui)
    refresh()


