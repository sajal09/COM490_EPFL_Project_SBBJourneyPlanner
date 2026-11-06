SELECT 
    curr.stop_id AS from_stop,
    curr.departure_time AS from_time,
    nxt.stop_id AS to_stop,
    nxt.arrival_time AS to_time,
    'ride' AS edge_type,
    curr.trip_id,
    (nxt.arrival_time - curr.departure_time) AS weight
FROM ${userns}.relevant_trips AS curr
JOIN ${userns}.relevant_trips AS nxt 
    ON  curr.trip_id = nxt.trip_id 
    AND curr.stop_sequence + 1 = nxt.stop_sequence
WHERE curr.stop_id != nxt.stop_id