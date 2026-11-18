# Candidate feature handoff

Use train_candidates.csv.gz for training and trip-grouped validation. It covers all 80 training trips. Use student_test_candidates.csv.gz for development evaluation. It covers nine of the ten student-test trips. Trip 00000000 is missing its .arcs file in the extracted source and is excluded. Its coverage counts are blank.

The search radius is 75 m with up to 10 candidates per GPS fix. Columns are split, trip_id, observation_index, timestamp, arc_id, candidate_rank, distance_m, gps_heading_deg, arc_heading_deg, and heading_error_deg. Blank heading fields indicate unavailable measurements.

Keep trip_id as a string to preserve leading zeros. Join Prajoth's aligned labels on split, trip_id, and observation_index, using the original zero-based GPS row. A candidate is positive when arc_id equals true_arc_id. Fixes without candidates have no feature rows. Candidate miss rate requires aligned labels.
