from pathlib import Path

SQL_DIR = Path(__file__).parent.parent / "sql"

# prepare
CREATE_STOPS            = SQL_DIR / "prepare/create_stops.sql"
CREATE_STOP_TIMES       = SQL_DIR / "prepare/create_stop_times.sql"
CREATE_STOP_TO_STOP     = SQL_DIR / "prepare/create_stop_to_stop.sql"
CREATE_WALK_PATHS       = SQL_DIR / "prepare/create_walk_paths.sql"

# routing
CREATE_RELEVANT_TRIPS   = SQL_DIR / "routing/create_relevant_trips.sql"
WALK_EDGES              = SQL_DIR / "routing/walk_edges.sql"
RIDE_EDGES              = SQL_DIR / "routing/ride_edges.sql"
DWELL_EDGES             = SQL_DIR / "routing/dwell_edges.sql"
SINK_EDGES              = SQL_DIR / "routing/sink_edges.sql"
SOURCE_EDGES            = SQL_DIR / "routing/source_edges.sql"
TRANSIT_QUERY           = SQL_DIR / "routing/transit_query.sql"
WALK_QUERY              = SQL_DIR / "routing/walk_query.sql"
