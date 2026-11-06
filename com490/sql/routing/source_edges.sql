WITH first_departure AS (
    SELECT
        stop_id,
        departure_time,
        arrival_time,
        trip_id,
        ROW_NUMBER() OVER (
            PARTITION BY stop_id
            ORDER BY departure_time
        ) AS rn
    FROM ${userns}.relevant_trips
)
SELECT
    ${source}                              AS from_stop,
    ${departure_time}                      AS from_time,
    ${source}                              AS to_stop,
    departure_time                         AS to_time,
    departure_time - ${departure_time}     AS weight,
    'source'                               AS edge_type,
    trip_id
FROM first_departure
WHERE rn = 1