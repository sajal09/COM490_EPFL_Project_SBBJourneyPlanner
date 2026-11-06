WITH walk_candidates AS (
    SELECT
        a.stop_id        AS from_stop,
        a.arrival_time   AS from_time,
        s.to_stop        AS to_stop,
        s.weight         AS walk_time
    FROM ${userns}.relevant_trips a
    JOIN ${userns}.walk_edges s ON s.from_stop = a.stop_id
    WHERE s.distance <= ${max_walk_m}
),
raw_walk_edges AS (
    -- Branch 1: to_stop has onward departures — keep earliest
    SELECT
        wc.from_stop,
        wc.from_time,
        wc.to_stop,
        b.departure_time                AS to_time,
        'walk'                          AS edge_type,
        NULL                            AS trip_id,
        b.departure_time - wc.from_time AS weight,
        ROW_NUMBER() OVER (
            PARTITION BY wc.from_stop, wc.from_time, wc.to_stop
            ORDER BY b.departure_time ASC
        ) AS rn
    FROM walk_candidates wc
    JOIN ${userns}.relevant_trips b
      ON b.stop_id = wc.to_stop
     AND b.departure_time >= wc.from_time + wc.walk_time

    UNION ALL

    -- Branch 2: to_stop has no onward departures — terminal walk
    SELECT
        wc.from_stop,
        wc.from_time,
        wc.to_stop,
        wc.from_time + wc.walk_time     AS to_time,
        'walk'                          AS edge_type,
        NULL                            AS trip_id,
        wc.walk_time                    AS weight,
        1                               AS rn
    FROM walk_candidates wc
)
SELECT
    from_stop,
    from_time,
    to_stop,
    to_time,
    edge_type,
    trip_id,
    weight
FROM raw_walk_edges
WHERE rn = 1