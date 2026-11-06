SELECT a_stop_id, b_stop_id, distance
FROM ${userns}.stop_to_stop
WHERE distance <= ${max_walk_m}