WITH relevant_trips AS (
    SELECT DISTINCT trip_id
    FROM ${userns}.stop_times
    WHERE ${day} = TRUE
    AND departure_time BETWEEN ${query_time} AND ${window_end}
),
trip_data AS (
    SELECT st.trip_id, st.stop_id, st.stop_sequence,
           st.departure_time, st.arrival_time,
           MIN(st.trip_id) OVER (
               PARTITION BY st.stop_id, st.stop_sequence, 
                            st.departure_time, st.arrival_time
           ) AS canonical_trip_id
    FROM ${userns}.stop_times st
    INNER JOIN relevant_trips rt ON st.trip_id = rt.trip_id
)
SELECT trip_id, stop_id, stop_sequence, departure_time, arrival_time
FROM trip_data
WHERE trip_id = canonical_trip_id
ORDER BY trip_id, stop_sequence