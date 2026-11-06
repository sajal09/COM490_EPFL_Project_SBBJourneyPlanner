DROP TABLE IF EXISTS ${userns}.stop_times;

CREATE TABLE ${userns}.stop_times AS
    WITH regional_stops AS (
        SELECT stop_id
        FROM ${userns}.stops
    ),
    relevant_trips AS (
        SELECT DISTINCT st.trip_id
        FROM ${sharedns}.sbb_stop_times st
        INNER JOIN regional_stops rs
            ON TRY(CAST(split_part(st.stop_id, ':', 1) AS BIGINT)) = rs.stop_id
        WHERE st.pub_date = DATE '${max_pub_date}'
    ),
    parsed_times AS (
        SELECT
            st.trip_id,
            TRY(CONCAT (CAST(split_part(st.trip_id, '.', 3) AS VARCHAR), '.' ,CAST(split_part(st.trip_id, '.', 5) AS VARCHAR))) AS route_id,
            TRY(CAST(split_part(st.stop_id, ':', 1) AS BIGINT)) AS stop_id,
            st.stop_sequence,
            (
                CAST(split_part(st.departure_time, ':', 1) AS INTEGER) * 3600 +
                CAST(split_part(st.departure_time, ':', 2) AS INTEGER) * 60 +
                CAST(split_part(st.departure_time, ':', 3) AS INTEGER)
            ) AS departure_time,
            (
                CAST(split_part(st.arrival_time, ':', 1) AS INTEGER) * 3600 +
                CAST(split_part(st.arrival_time, ':', 2) AS INTEGER) * 60 +
                CAST(split_part(st.arrival_time, ':', 3) AS INTEGER)
            ) AS arrival_time
        FROM ${sharedns}.sbb_stop_times st
        INNER JOIN relevant_trips rt
            ON st.trip_id = rt.trip_id
        WHERE st.pub_date = DATE '${max_pub_date}'
    )
    SELECT
        pt.trip_id,
        pt.route_id,
        pt.stop_id,
        pt.stop_sequence,
        pt.departure_time,
        pt.arrival_time,
        CAST(c.monday    AS BOOLEAN) AS monday,
        CAST(c.tuesday   AS BOOLEAN) AS tuesday,
        CAST(c.wednesday AS BOOLEAN) AS wednesday,
        CAST(c.thursday  AS BOOLEAN) AS thursday,
        CAST(c.friday    AS BOOLEAN) AS friday,
        CAST(c.saturday  AS BOOLEAN) AS saturday,
        CAST(c.sunday    AS BOOLEAN) AS sunday
    FROM parsed_times pt
    INNER JOIN ${sharedns}.sbb_trips tr
        ON pt.trip_id = tr.trip_id
        AND tr.pub_date = DATE '${max_pub_date}'
    INNER JOIN ${sharedns}.sbb_calendar c
        ON tr.service_id = c.service_id
        AND c.pub_date = DATE '${max_pub_date}'
    WHERE pt.stop_id IS NOT NULL