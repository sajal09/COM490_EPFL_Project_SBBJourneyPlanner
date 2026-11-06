DROP MATERIALIZED VIEW IF EXISTS ${userns}.walk_edges;  
    
CREATE MATERIALIZED VIEW ${userns}.walk_edges AS
    WITH bidirectional AS (
        -- Direction A to B
        SELECT 
            a_stop_id AS from_stop, 
            b_stop_id AS to_stop, 
            distance
        FROM ${userns}.stop_to_stop
        
        UNION ALL
        
        -- Direction B to A
        SELECT 
            b_stop_id AS from_stop, 
            a_stop_id AS to_stop, 
            distance
        FROM ${userns}.stop_to_stop
    )
    SELECT
        from_stop,
        CAST(NULL AS INTEGER) AS from_time,
        to_stop,
        CAST(NULL AS INTEGER) AS to_time,
        'walk' AS edge_type,
        CAST(NULL AS VARCHAR) AS trip_id,
        -- Use 1.2 to avoid integer division truncation
        CAST(distance * 1.2 AS INTEGER) AS weight,
        distance
    FROM bidirectional