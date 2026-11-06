WITH last_arrival AS (
    SELECT
        stop_id,
        arrival_time,
        departure_time,
        trip_id,
        LEAD(arrival_time) OVER (
            PARTITION BY stop_id
            ORDER BY arrival_time
        ) AS next_arrival
    FROM ${userns}.relevant_trips
)
SELECT
    stop_id                AS from_stop,
    arrival_time           AS from_time,
    stop_id                AS to_stop,
    ${arrival_time}        AS to_time,
    0                      AS weight,
    'sink'                 AS edge_type,
    NULL                   AS trip_id
FROM last_arrival
WHERE next_arrival IS NULL
   OR next_arrival > ${arrival_time}
