"""
backend.py — COM-490 Final Project: Robust Journey Planner
==========================================================
All pipeline logic encapsulated in one importable module.
The notebook UI imports from here; this file should never
need to be edited during a normal demo session.
"""

from __future__ import annotations

import os
import sys
import pwd
import json
import time
import subprocess
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Environment & paths
# ---------------------------------------------------------------------------

def load_environment() -> dict:
    """
    Validate required environment variables and return a configuration dict.

    Returns
    -------
    dict with keys:
        username, hadoopFS, USER_NS, FINAL_DIR,
        ISTDATEN_PARQUET, and all P_* parquet paths.

    Raises
    ------
    AssertionError  if any required env var is missing.
    """
    assert os.getenv("EPFL_COM490_TOKEN"), "Missing EPFL_COM490_TOKEN env var"
    assert os.getenv("TRINO_URL"),         "Missing TRINO_URL env var"
    assert os.getenv("HADOOP_FS"),         "Missing HADOOP_FS env var"

    username  = pwd.getpwuid(os.getuid()).pw_name
    hadoopFS  = os.environ["HADOOP_FS"]
    USER_NS   = f"iceberg.{username}_iceberg"
    GROUP_NAME = "D2"
    FINAL_DIR = f"{hadoopFS}/user/{username}/final"

    paths = dict(
        username         = username,
        hadoopFS         = hadoopFS,
        USER_NS          = USER_NS,
        GROUP_NAME       = GROUP_NAME,
        FINAL_DIR        = FINAL_DIR,
        #ISTDATEN_PARQUET = f"{hadoopFS}/user/{username}/assignment2/lausanne_clean.parquet",
        ISTDATEN_PARQUET      = f"{FINAL_DIR}/istdaten_clean.parquet",
        # Discrete-PMF stop-delay tables — 3-tier fallback
        # primary (line, bpuic, dow, hour) → no_dow (line, bpuic, hour) → no_line (bpuic, hour)
        P_STOP_DELAYS         = f"{FINAL_DIR}/stop_delays.parquet",
        P_STOP_DELAYS_NODOW   = f"{FINAL_DIR}/stop_delays_no_dow.parquet",
        P_STOP_DELAYS_NOLINE  = f"{FINAL_DIR}/stop_delays_no_line.parquet",
        # Gumbel-parameterised twin (same fallback hierarchy; different schema:
        # arr_mu/arr_beta/arr_n + dep_mu/dep_beta/dep_n instead of bucket PMF cols).
        P_STOP_DELAYS_G       = f"{FINAL_DIR}/stop_delays_gumbel.parquet",
        P_STOP_DELAYS_G_NODOW = f"{FINAL_DIR}/stop_delays_gumbel_no_dow.parquet",
        P_STOP_DELAYS_G_NOLINE= f"{FINAL_DIR}/stop_delays_gumbel_no_line.parquet",
        P_SEGMENT_TT          = f"{FINAL_DIR}/segment_tt.parquet",
        P_SEGMENT_TT_NOLINE   = f"{FINAL_DIR}/segment_tt_no_line.parquet",
        P_BUS_SNAPSHOT        = f"{FINAL_DIR}/bus_state_snapshot.parquet",
        P_TRIP_TO_LINE        = f"{FINAL_DIR}/trip_to_line.parquet",
    )
    os.makedirs(f"/tmp/{pwd.getpwuid(os.getuid()).pw_name}/", exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# 2. Spark session
# ---------------------------------------------------------------------------

def create_spark_session(username: str, hadoopFS: str):
    """
    Create and return a configured SparkSession for the EPFL cluster.

    Parameters
    ----------
    username : str
    hadoopFS : str  root HDFS path, e.g. hdfs://...

    Returns
    -------
    pyspark.sql.SparkSession
    """
    from pyspark.sql import SparkSession
    from random import randrange

    spark = (
        SparkSession.builder
        .appName(f"{username}-final")
        .config("spark.ui.port", randrange(4050, 4450, 5))
        .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path))
        .config(
            "spark.jars",
            f"{hadoopFS}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar,"
            f"{hadoopFS}/data/com-490/jars/sedona-spark-shaded-3.5_2.13-1.7.1.jar,"
            f"{hadoopFS}/data/com-490/jars/geotools-wrapper-1.7.1-28.5.jar",
        )
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg.type", "hadoop")
        .config("spark.sql.catalog.iceberg.warehouse", f"{hadoopFS}/data/com-490/silver/")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.iceberg.spark.SparkSessionCatalog")
        .config("spark.sql.catalog.spark_catalog.type", "hadoop")
        .config("spark.sql.catalog.spark_catalog.warehouse",
                f"{hadoopFS}/user/{username}/final/warehouse")
        .config("spark.sql.warehouse.dir",
                f"{hadoopFS}/user/{username}/final/spark/warehouse")
        .config("spark.executor.memory",    "6g")
        .config("spark.executor.cores",     "4")
        .config("spark.executor.instances", "4")
        .master("yarn")
        .getOrCreate()
    )
    return spark


# ---------------------------------------------------------------------------
# 3. HDFS utilities
# ---------------------------------------------------------------------------

def hdfs_exists(spark, path: str) -> bool:
    """Return True if *path* exists on HDFS."""
    sc  = spark.sparkContext
    URI  = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    fs   = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(
        URI(path.split("/user/")[0]),
        sc._jsc.hadoopConfiguration(),
    )
    return fs.exists(Path(path))


def hdfs_get(remote: str, local: str) -> None:
    """Copy a remote HDFS file to a local path (no-op if local already exists)."""
    if not os.path.exists(local):
        subprocess.check_call(["hdfs", "dfs", "-get", remote, local])


# ---------------------------------------------------------------------------
# 4. Journey Planner initialisation
# ---------------------------------------------------------------------------

def init_journey_planner(
    USER_NS: str,
    REGION_UUIDS: list[str],
    force_prepare: bool = False,
):
    """
    Open a Trino connection and optionally materialise the per-user views.

    Parameters
    ----------
    USER_NS        : iceberg namespace, e.g. 'iceberg.jdoe'
    REGION_UUIDS   : list of region UUID strings for jp.prepare()
    force_prepare  : re-run prepare() even if views already exist

    Returns
    -------
    JourneyPlanner instance (with name_to_id / id_to_name populated)
    """
    from journey_planner import JourneyPlanner

    jp = JourneyPlanner(schema=USER_NS)
    print("Trino connection live.")
    print(f"Shared namespace : {jp.sharedns}")
    print(f"User namespace   : {jp.userns}")

    if force_prepare or REGION_UUIDS:
        jp.prepare(regions=REGION_UUIDS)
        print(f"\nStops in {USER_NS}.stops: {len(jp.name_to_id)} distinct names.")
    else:
        _load_stop_names(jp)

    return jp


def _load_stop_names(jp) -> None:
    """Populate jp.name_to_id / jp.id_to_name from the Trino stops table."""
    if jp.name_to_id is None:
        stops_df      = pd.read_sql(
            f"SELECT stop_id, stop_name FROM {jp.userns}.stops", jp.conn
        )
        jp.name_to_id = dict(zip(stops_df["stop_name"], stops_df["stop_id"]))
        jp.id_to_name = {v: k for k, v in jp.name_to_id.items()}
        print(f"Loaded {len(jp.name_to_id)} stop names from Trino.")


def search_stops(jp, substring: str) -> list[tuple[str, str]]:
    """
    Return a list of (stop_name, stop_id) whose name contains *substring* (case-insensitive).
    """
    sub = substring.upper()
    return sorted(
        (name, sid)
        for name, sid in jp.name_to_id.items()
        if sub in name.upper()
    )


# ---------------------------------------------------------------------------
# 5. Bild the cleaned istdaten table(Spark — run once)
# ---------------------------------------------------------------------------

def build_istdaten_clean(
    spark,
    jp,
    cfg: dict,
    holdout_days: int = 30,
    min_year:     int = 2025,
) -> None:
    """
    Build the cleaned istdaten parquet from iceberg.sbb.istdaten.

    - Region scope : bpuic of stops in jp.userns.stops (any region given to jp).
    - Time scope   : from `min_year-01-01` up to (max_available_day - holdout_days),
                     so the last `holdout_days` days of AVAILABLE data are held
                     out — guarantees non-empty evaluation window even when the
                     dataset doesn't run up to today.
    - Quality      : unplanned == False, arr_actual not null,
                     arr_status REAL/GESCHAETZT.

    Output: cfg["ISTDATEN_PARQUET"]
    """
    import pyspark.sql.functions as F
    from datetime import timedelta

    if hdfs_exists(spark, cfg["ISTDATEN_PARQUET"]):
        print("istdaten clean already exists — skipping.")
        return

    # ── Region scope ──────────────────────────────────────────────────────
    stops_df = pd.read_sql(
        f"SELECT DISTINCT stop_id FROM {jp.userns}.stops", jp.conn
    )
    region_bpuics = [int(x) for x in stops_df["stop_id"].tolist()]
    print(f"Region scope: {len(region_bpuics):,} distinct bpuic from {jp.userns}.stops")

    # ── Find the latest available operating_day in istdaten ───────────────
    max_day_row = (
        spark.table("iceberg.sbb.istdaten")
        .agg(F.max("operating_day").alias("d"))
        .first()
    )
    max_day = max_day_row["d"]                    # datetime.date
    cutoff  = max_day - timedelta(days=holdout_days)
    print(f"Available istdaten range up to : {max_day}")
    print(f"Training cutoff (max − {holdout_days}d) : {cutoff}")
    print(f"Holdout window for evaluation  : ({cutoff}, {max_day}]")

    (
        spark.table("iceberg.sbb.istdaten")
        .filter(F.year("operating_day") >= min_year)
        .filter(F.col("operating_day") < F.lit(cutoff.isoformat()))
        .filter(F.col("unplanned") == False)
        .filter(F.col("arr_actual").isNotNull())
        .filter(F.col("arr_status").isNotNull())
        .filter(F.upper(F.trim(F.col("arr_status"))).isin(["REAL", "GESCHAETZT"]))
        .filter(F.col("bpuic").isin(region_bpuics))
        .write.parquet(cfg["ISTDATEN_PARQUET"], mode="overwrite")
    )
    print(f"Wrote {cfg['ISTDATEN_PARQUET']}")


# ---------------------------------------------------------------------------
# 6. Delay-distribution tables (Spark — run once)
# ---------------------------------------------------------------------------

def build_delay_tables(spark, cfg: dict,
                       stop_delay_distribution: str = "discrete") -> None:
    """
    Build stop_delays and segment_tt parquets if they do not yet exist.
    Slow step (~minutes); safe to call repeatedly.

    Parameters
    ----------
    spark : SparkSession
    cfg   : dict returned by load_environment()
    stop_delay_distribution : 'gumbel' or 'discrete'
    The segment-travel-time parquet is shared between modes and is
    always built if missing.
    """
    if stop_delay_distribution not in ("gumbel", "discrete"):
        raise ValueError(
            f"stop_delay_distribution must be 'gumbel' or 'discrete', "
            f"got {stop_delay_distribution!r}"
        )

    from models.delay_distributions import (
        build_stop_delays,
        build_stop_delays_gumbel,
        build_segment_travel_times,
    )

    if stop_delay_distribution == "discrete":
        if not (hdfs_exists(spark, cfg["P_STOP_DELAYS"])
                and hdfs_exists(spark, cfg["P_STOP_DELAYS_NODOW"])
                and hdfs_exists(spark, cfg["P_STOP_DELAYS_NOLINE"])):
            print("Building stop_delays (discrete PMF)… (this takes a few minutes)")
            build_stop_delays(
                spark,
                istdaten_parquet_path = cfg["ISTDATEN_PARQUET"],
                output_path           = cfg["P_STOP_DELAYS"],
                fallback_no_dow_path  = cfg["P_STOP_DELAYS_NODOW"],
                fallback_no_line_path = cfg["P_STOP_DELAYS_NOLINE"],
                min_n = 50,
            )
            print("  stop_delays done.")
        else:
            print("stop_delays already exists — skipping.")
    else:  # gumbel
        if not (hdfs_exists(spark, cfg["P_STOP_DELAYS_G"])
                and hdfs_exists(spark, cfg["P_STOP_DELAYS_G_NODOW"])
                and hdfs_exists(spark, cfg["P_STOP_DELAYS_G_NOLINE"])):
            print("Building stop_delays (Gumbel)… (this takes a few minutes)")
            build_stop_delays_gumbel(
                spark,
                istdaten_parquet_path = cfg["ISTDATEN_PARQUET"],
                output_path           = cfg["P_STOP_DELAYS_G"],
                fallback_no_dow_path  = cfg["P_STOP_DELAYS_G_NODOW"],
                fallback_no_line_path = cfg["P_STOP_DELAYS_G_NOLINE"],
                min_n = 50,
            )
            print("  stop_delays_gumbel done.")
        else:
            print("stop_delays_gumbel already exists — skipping.")

    if not hdfs_exists(spark, cfg["P_SEGMENT_TT"]):
        print("Building segment_tt… (this takes a few minutes)")
        build_segment_travel_times(
            spark,
            istdaten_parquet_path = cfg["ISTDATEN_PARQUET"],
            output_path           = cfg["P_SEGMENT_TT"],
            fallback_no_line_path = cfg["P_SEGMENT_TT_NOLINE"],
            min_n        = 50,
            max_stop_gap = 20,
        )
        print("  segment_tt done.")
    else:
        print("segment_tt already exists — skipping.")


# ---------------------------------------------------------------------------
# 6. trip_to_line mapping
# ---------------------------------------------------------------------------

def load_trip_to_line(jp, cfg: dict) -> dict:
    """
    Load (or build) the trip_id -> line_name dict.

    Tries the local cache at /tmp/username/trip_to_line.parquet first,
    then the HDFS path, then queries Trino.

    Returns
    -------
    dict mapping trip_id (str) -> route_short_name (str)
    """
    local = f"/tmp/{pwd.getpwuid(os.getuid()).pw_name}/trip_to_line.parquet"

    if os.path.exists(local):
        df = pd.read_parquet(local)
    elif hdfs_exists_path(cfg["P_TRIP_TO_LINE"]):
        df = pd.read_parquet(cfg["P_TRIP_TO_LINE"])
        df.to_parquet(local, index=False)
    else:
        max_pub = pd.read_sql(
            f"SELECT MAX(pub_date) AS d FROM {jp.sharedns}.sbb_stops", jp.conn
        ).iloc[0, 0]
        sql = f"""
            SELECT t.trip_id, r.route_short_name AS line
            FROM {jp.sharedns}.sbb_trips  t
            JOIN {jp.sharedns}.sbb_routes r
              ON t.route_id = r.route_id AND t.pub_date = r.pub_date
            WHERE t.pub_date = DATE '{max_pub}'
        """
        df = pd.read_sql(sql, jp.conn)
        df.to_parquet(local, index=False)
        print(f"trip_to_line built from Trino ({len(df):,} rows, pub_date={max_pub}).")

    print(f"trip_to_line: {len(df):,} rows loaded.")
    return dict(zip(df["trip_id"], df["line"]))


def hdfs_exists_path(path: str) -> bool:
    """Lightweight HDFS existence check via the hdfs CLI (no SparkSession needed)."""
    result = subprocess.run(
        ["hdfs", "dfs", "-test", "-e", path], capture_output=True
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# 7. Bus-state snapshot
# ---------------------------------------------------------------------------

def build_bus_snapshot(
    spark,
    cfg: dict,
    operating_day: str,
    query_time_s:  int,
    jp,
) -> pd.DataFrame:
    """
    Build (or load) the real-time bus-state snapshot.

    Parameters
    ----------
    spark          : SparkSession
    cfg            : config dict from load_environment()
    operating_day  : 'YYYY-MM-DD' string, e.g. '2026-01-15'
    query_time_s   : seconds-since-midnight representing "now"

    Returns
    -------
    pandas DataFrame with columns [trip_id, bpuic, arr_actual_s]
    """
    import pyspark.sql.functions as F
    from pyspark.sql import Window

    if not hdfs_exists(spark, cfg["P_BUS_SNAPSHOT"]):
        stops_df = pd.read_sql(
        f"SELECT DISTINCT stop_id FROM {jp.userns}.stops", jp.conn
        )
        region_bpuics = [int(x) for x in stops_df["stop_id"].tolist()]
        snap = (
            spark.table("iceberg.sbb.istdaten")
            .filter(F.col("operating_day") == operating_day)
            .filter(F.col("unplanned") == False)
            .filter(F.col("arr_status").isNotNull())
            .filter(F.upper(F.trim(F.col("arr_status"))).isin(["REAL", "GESCHAETZT"]))
            .filter(F.col("bpuic").isin(region_bpuics))
            .filter(F.col("arr_actual").isNotNull())
            .withColumn(
                "arr_actual_s",
                F.hour("arr_actual") * 3600 +
                F.minute("arr_actual") * 60 +
                F.second("arr_actual"),
            )
            .filter(F.col("arr_actual_s") <= query_time_s)
            .withColumn(
                "rn",
                F.row_number().over(
                    Window.partitionBy("trip_id").orderBy(F.desc("arr_actual_s"))
                ),
            )
            .filter(F.col("rn") == 1)
            .select("trip_id", "bpuic", "arr_actual_s")
        )
        snap.write.parquet(cfg["P_BUS_SNAPSHOT"], mode="overwrite")
        print("Bus snapshot built and written to HDFS.")
    else:
        print("Bus snapshot already exists on HDFS, loading.")

    snapshot = spark.read.parquet(cfg["P_BUS_SNAPSHOT"]).toPandas()
    print(f"Snapshot rows: {len(snapshot):,}")
    return snapshot


def make_bus_state_fn(snapshot: pd.DataFrame):
    """
    Wrap the snapshot DataFrame in a callable suitable for path_confidence.

    Returns a function:  bus_state_fn(trip_id, query_time_s) -> (bpuic, arr_actual_s) | None
    """
    idx = snapshot.set_index("trip_id")

    def bus_state_fn(trip_id, query_time_s):
        if trip_id not in idx.index:
            return None
        row = idx.loc[trip_id]
        return int(row["bpuic"]), float(row["arr_actual_s"])

    return bus_state_fn


# ---------------------------------------------------------------------------
# 8. Lookup objects
# ---------------------------------------------------------------------------

def load_lookups(cfg: dict, stop_delay_distribution: str = "discrete"):
    """
    Download delay/travel-time parquets from HDFS and instantiate lookup objects.

    Parameters
    ----------
    cfg                     : dict returned by load_environment()
    stop_delay_distribution : 'gumbel' or 'discrete'
        Which family of stop-delay parquets to back the StopDelayLookup with.
        - 'discrete' : 71-bucket PMF, exact convolution at transfers.
        - 'gumbel'   : Gumbel(mu, beta) parametric, analytic / Monte-Carlo at
                       transfers.

    Returns
    -------
    (StopDelayLookup, SegmentTravelTimeLookup)
    """
    from models import StopDelayLookup, SegmentTravelTimeLookup

    if stop_delay_distribution not in ("gumbel", "discrete"):
        raise ValueError(
            f"stop_delay_distribution must be 'gumbel' or 'discrete', "
            f"got {stop_delay_distribution!r}"
        )

    username = pwd.getpwuid(os.getuid()).pw_name

    # Build the download list lazily — only pull the variant we'll actually use.
    if stop_delay_distribution == "gumbel":
        local_map = {
            cfg["P_STOP_DELAYS_G"]:        f"/tmp/{username}/stop_delays_gumbel.parquet",
            cfg["P_STOP_DELAYS_G_NODOW"]:  f"/tmp/{username}/stop_delays_gumbel_no_dow.parquet",
            cfg["P_STOP_DELAYS_G_NOLINE"]: f"/tmp/{username}/stop_delays_gumbel_no_line.parquet",
        }
    else:
        local_map = {
            cfg["P_STOP_DELAYS"]:        f"/tmp/{username}/stop_delays.parquet",
            cfg["P_STOP_DELAYS_NODOW"]:  f"/tmp/{username}/stop_delays_no_dow.parquet",
            cfg["P_STOP_DELAYS_NOLINE"]: f"/tmp/{username}/stop_delays_no_line.parquet",
        }
    # Segment travel times are shared between modes.
    local_map[cfg["P_SEGMENT_TT"]]        = f"/tmp/{username}/segment_tt.parquet"
    local_map[cfg["P_SEGMENT_TT_NOLINE"]] = f"/tmp/{username}/segment_tt_no_line.parquet"

    for remote, local in local_map.items():
        hdfs_get(remote, local)

    if stop_delay_distribution == "gumbel":
        stop_delays = StopDelayLookup(
            primary_path          = f"/tmp/{username}/stop_delays_gumbel.parquet",
            fallback_no_dow_path  = f"/tmp/{username}/stop_delays_gumbel_no_dow.parquet",
            fallback_no_line_path = f"/tmp/{username}/stop_delays_gumbel_no_line.parquet",
            mode                  = "gumbel",
            min_n = 50,
        )
    else:
        stop_delays = StopDelayLookup(
            primary_path          = f"/tmp/{username}/stop_delays.parquet",
            fallback_no_dow_path  = f"/tmp/{username}/stop_delays_no_dow.parquet",
            fallback_no_line_path = f"/tmp/{username}/stop_delays_no_line.parquet",
            mode                  = "discrete",
            min_n = 50,
        )

    travel_times = SegmentTravelTimeLookup(
        primary_path          = f"/tmp/{username}/segment_tt.parquet",
        fallback_no_line_path = f"/tmp/{username}/segment_tt_no_line.parquet",
        min_n = 50,
    )
    print(f"Lookups loaded (stop_delay_distribution={stop_delay_distribution!r}).")
    return stop_delays, travel_times


# ---------------------------------------------------------------------------
# 9. Feature Importance
# ---------------------------------------------------------------------------

def validate_features(spark, cfg: dict, sample_fraction: float = 0.05):
    """
    Run gradient-boosting + permutation-importance validation of the
    delay-distribution conditioning keys.
    Returns a sorted DataFrame [feature, importance].
    """
    from models.feature_importance import analyze_delay_features_full
    return analyze_delay_features_full(
        spark,
        istdaten_parquet_path = cfg["ISTDATEN_PARQUET"],
        sample_fraction       = sample_fraction,
    )


# ---------------------------------------------------------------------------
# 10. Routing
# ---------------------------------------------------------------------------

def find_routes(
    jp,
    start_id:    str,
    end_id:      str,
    day:         str,
    departs:     str,
    deadline:    str,
    max_walk_m:  int  = 500,
    buffer_steps: tuple = (120, 180, 360),
    slack_steps:  tuple = (0,),
    top_k:        int   = 5,
) -> list[tuple[str, list]]:
    """
    Run route_latest and return a list of (depart_str, events) pairs.

    Parameters
    ----------
    jp          : JourneyPlanner
    start_id    : stop_id of origin
    end_id      : stop_id of destination
    day         : day string, e.g. 'monday'
    departs     : HH:MM departure string (used for display only in latest mode)
    deadline    : HH:MM arrival deadline
    max_walk_m  : maximum walking distance in metres
    buffer_steps: iterable of buffer-second values to explore
    slack_steps : iterable of slack-second values to explore
    top_k       : maximum number of candidate routes to return

    Returns
    -------
    List of (depart_str, event_list) sorted by descending departure time.
    """
    from routing.time_utils import seconds_to_time

    candidate_routes = jp.route_latest(
        start_id     = start_id,
        end_id       = end_id,
        deadline     = deadline,
        day          = day,
        max_walk_m   = max_walk_m,
        buffer_steps = buffer_steps,
        slack_steps  = slack_steps,
        top_k        = top_k
    )

    candidates = []
    for evs in candidate_routes:
        first_board = next(
            (e for e in evs if e[1] is not None and e[2] is not None and e[3] is None),
            None,
        )
        depart_str = seconds_to_time(first_board[0])[:5] if first_board else departs
        candidates.append((depart_str, evs))

    candidates.sort(key=lambda x: x[0], reverse=True)
    print(f"Found {len(candidates)} distinct candidate path(s).")
    return candidates


# ---------------------------------------------------------------------------
# 11. Confidence scoring
# ---------------------------------------------------------------------------

def score_routes(
    candidates:    list[tuple[str, list]],
    stop_delays,
    travel_times,
    trip_to_line:  dict,
    day:           str,
    query_time_s:  int,
    deadline_s:    int,
    bus_state_fn,
    q_threshold:   float = 0.80,
    n_samples:     int   = 10_000,
    seed:          int   = 42,
) -> pd.DataFrame:
    """
    Score every candidate route with path_confidence and return a ranked DataFrame.

    Columns: depart_str, arr_str, n_transfers, overall_p, meets_Q, breakdown
    """
    from models import path_confidence, day_to_dow
    from routing.time_utils import seconds_to_time

    rng  = np.random.default_rng(seed)
    rows = []

    for depart_str, evs in candidates:
        p, br = path_confidence(
            events        = evs,
            stop_delays   = stop_delays,
            travel_times  = travel_times,
            trip_to_line  = trip_to_line,
            dow           = day_to_dow(day),
            query_time_s  = query_time_s,
            deadline_s    = deadline_s,
            buffer_s      = 0,
            bus_state_fn  = bus_state_fn,
            n_samples     = n_samples,
            rng           = rng,
        )
        arr_ts = next(
            (b["sched_arr_ts"] for b in br if b["type"] == "arrival"),
            evs[-1][0] if evs else None,
        )
        rows.append(
            dict(
                depart_str   = depart_str,
                arr_str      = seconds_to_time(arr_ts)[:-3] if arr_ts is not None else None,
                n_transfers  = sum(1 for b in br if b["type"] == "transfer"),
                overall_p    = p,
                meets_Q      = p >= q_threshold,
                breakdown    = br,
            )
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("depart_str", ascending=False)
        .reset_index(drop=True)
    )
    return df


def print_route_summary(
    summary:     pd.DataFrame,
    deadline:    str,
    q_threshold: float,
) -> None:
    """Pretty-print the scored route summary table."""
    display_cols = ["depart_str", "arr_str", "n_transfers", "overall_p", "meets_Q"]
    print(f'\n{"=" * 65}')
    print(f'  Q threshold: {q_threshold:.0%}   deadline: {deadline}')
    print(f'{"=" * 65}')
    print(summary[display_cols].to_string(index=False))
    feasible = summary[summary["meets_Q"]]
    print(f"\nFeasible routes ({len(feasible)} of {len(summary)}, sorted by latest departure):")
    if len(feasible):
        print(feasible[display_cols].to_string(index=False))
    else:
        print("  None meet the confidence threshold.")


def print_breakdown(summary: pd.DataFrame) -> None:
    """Print a per-segment breakdown for every candidate."""
    for _, row in summary.iterrows():
        print(f'\n=== {row["depart_str"]}  total p={row["overall_p"]:.3f} ===')
        for seg in row["breakdown"]:
            if seg["type"] == "transfer":
                gap = seg["board_ts"] - seg["alight_ts"]
                print(
                    f"  transfer @ {seg['alight_stop']}: "
                    f"{seg['incoming_line']} → {seg['outgoing_line']}, "
                    f"gap={gap}s, walk={seg['walk_s']:.0f}s, p={seg['p']:.3f}"
                )
            elif seg["type"] == "arrival":
                cushion = seg["deadline_s"] - seg["sched_arr_ts"]
                print(
                    f"  arrival  @ {seg['final_stop']}: "
                    f"line {seg['final_line']}, cushion={cushion}s, p={seg['p']:.3f}"
                )


# ---------------------------------------------------------------------------
# 12. Visualisation
# ---------------------------------------------------------------------------

def launch_ui(
    jp,
    summary:          pd.DataFrame,
    candidates:       list[tuple[str, list]],
    stop_delays,
    travel_times,
    trip_to_line:     dict,
    day:              str,
    query_time_s:     int,
    deadline_s:       int,
    bus_state_fn,
    q_threshold:      float = 0.80,
    origin_name:      str   = "",
    destination_name: str   = "",
    deadline:         str   = "",
) -> None:
    """
    Build the route summary dict and launch the ipywidget route UI.

    Parameters
    ----------
    origin_name      : human-readable origin stop name (for the title)
    destination_name : human-readable destination stop name (for the title)
    deadline         : HH:MM arrival deadline string (for the title)
    """
    import importlib
    import vizu.route_visualizer as rv
    importlib.reload(rv)

    from models import path_confidence, day_to_dow

    stops_df = pd.read_sql(f"SELECT * FROM {jp.userns}.stops", jp.conn)

    viz_summary = rv.build_summary(
        candidates          = candidates,
        path_confidence_fn  = path_confidence,
        stop_delays         = stop_delays,
        travel_times        = travel_times,
        trip_to_line        = trip_to_line,
        day_to_dow          = day_to_dow,
        day                 = day,
        query_time_s        = query_time_s,
        deadline_s          = deadline_s,
        bus_state_fn        = bus_state_fn,
        q_threshold         = q_threshold,
    )

    rv.launch_route_ui(
        stops            = stops_df,
        summary          = viz_summary,
        trip_to_line     = trip_to_line,
        origin_name      = origin_name,
        destination_name = destination_name,
        deadline         = deadline,
    )


# ---------------------------------------------------------------------------
# 13. Helper function
# ---------------------------------------------------------------------------

def get_holdout_window(spark, holdout_days: int = 30):
    """
    Return (cutoff, max_day) — the evaluation window held out of training.

    Any OPERATING_DAY in `(cutoff, max_day]` is safe to use for the
    real-time bus snapshot: it was not seen by the delay distributions.
    """
    import pyspark.sql.functions as F
    from datetime import timedelta

    max_day = (
        spark.table("iceberg.sbb.istdaten")
        .agg(F.max("operating_day").alias("d"))
        .first()["d"]
    )
    cutoff = max_day - timedelta(days=holdout_days)
    return cutoff, max_day
