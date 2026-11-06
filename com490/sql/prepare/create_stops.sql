DROP MATERIALIZED VIEW IF EXISTS ${userns}.stops;

CREATE MATERIALIZED VIEW ${userns}.stops AS
WITH selected_regions AS (
    SELECT ST_GeomFromBinary(wkb_geometry) AS geom
    FROM ${sharedns}.geo
    WHERE CAST(uuid AS VARCHAR) IN (${region_uuids_sql})
),
candidate_stops AS (
    SELECT
        TRY(CAST(split_part(stop_id, ':', 1) AS BIGINT)) AS stop_id,
        stop_name,
        stop_lat,
        stop_lon,
        location_type,
        parent_station
    FROM ${sharedns}.sbb_stops
    WHERE pub_date = DATE '${max_pub_date}'
      AND split_part(stop_id, ':', 1) LIKE '85%'
),
filtered_stops AS (
    SELECT DISTINCT
        s.stop_id,
        s.stop_name,
        s.stop_lat,
        s.stop_lon
    FROM candidate_stops s
    CROSS JOIN selected_regions r
    WHERE s.stop_id IS NOT NULL
      AND (s.location_type IS NULL OR s.location_type <> 1)
      AND ST_Contains(
            r.geom,
            ST_Point(CAST(s.stop_lon AS DOUBLE), CAST(s.stop_lat AS DOUBLE))
      )
)
SELECT
    stop_id,
    min(stop_name) AS stop_name,
    AVG(stop_lat) AS stop_lat,
    AVG(stop_lon) AS stop_lon
FROM filtered_stops
GROUP BY stop_id;