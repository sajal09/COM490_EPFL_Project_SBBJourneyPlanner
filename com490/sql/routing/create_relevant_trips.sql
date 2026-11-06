
DROP TABLE IF EXISTS ${userns}.relevant_trips;

CREATE TABLE ${userns}.relevant_trips AS
    SELECT *
    FROM ${userns}.stop_times
    WHERE ${day} = TRUE
        AND departure_time >= ${query_time}