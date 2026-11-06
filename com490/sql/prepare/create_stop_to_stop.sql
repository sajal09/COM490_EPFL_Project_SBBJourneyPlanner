DROP MATERIALIZED VIEW IF EXISTS ${userns}.stop_to_stop;

CREATE MATERIALIZED VIEW ${userns}.stop_to_stop AS
    SELECT 
        a.stop_id AS a_stop_id,
        b.stop_id AS b_stop_id,
        ST_Distance(
            to_spherical_geography(ST_Point(a.stop_lon, a.stop_lat)),
            to_spherical_geography(ST_Point(b.stop_lon, b.stop_lat))
        ) AS distance
    FROM ${userns}.stops a
    CROSS JOIN ${userns}.stops b
    WHERE a.stop_id < b.stop_id
      AND ST_Distance(
            to_spherical_geography(ST_Point(a.stop_lon, a.stop_lat)),
            to_spherical_geography(ST_Point(b.stop_lon, b.stop_lat))
        ) <= 500