WITH next_events AS (
    SELECT 
        stop_id,
        arrival_time,
        LEAD(departure_time) OVER (
            PARTITION BY stop_id 
            ORDER BY departure_time ASC
        ) AS next_departure
    FROM ${userns}.relevant_trips
)
SELECT 
    stop_id AS from_stop,
    arrival_time AS from_time,
    stop_id AS to_stop,
    next_departure AS to_time,
    'dwell' AS edge_type,
    (next_departure - arrival_time) AS weight,
    NULL    AS trip_id
FROM next_events
WHERE next_departure IS NOT NULL 
    AND next_departure >= arrival_time
