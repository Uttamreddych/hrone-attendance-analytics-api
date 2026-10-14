# DECISIONS.md

1. **Indexes.**
We created unique index `employees: {emp_code: 1}` to guarantee employee code uniqueness, and `attendance_logs: {emp_code: 1, date: 1}` to guarantee at most one record per employee per day. For query performance, we created `attendance_logs: {date: -1, emp_code: 1}` to satisfy sorted pagination without in-memory sorting, `attendance_logs: {emp_code: 1, punch_in: -1}` for fast punch-out matching, `attendance_logs: {date: 1, late_minutes: 1, emp_code: 1}` for the late leaderboard, and `employees: {joined_on: 1, department: 1}` for headcount scans. We considered indexing individual fields in the embedded `history` array but rejected it because audit entries are always loaded with the parent record and never filtered standalone.

2. **Punch-in race.**
Both concurrent requests pass initial validation and attempt `insert_one()`. MongoDB's unique compound index on `(emp_code, date)` guarantees that only one document is inserted. The second request raises a `DuplicateKeyError`, which our FastAPI handler catches and converts into HTTP 409 (`already punched in for this date`). Exactly one 201 and one 409 are returned.

3. **Ties.**
Our pipeline uses MongoDB's `$setWindowFields` with `$rank` to apply standard competition ranking (1, 2, 2, 4). The `$match: {rank: {$lte: limit}}` filter is applied after ranking. Both tied employees at the cutoff share the qualifying rank and are returned, so the response may contain more than `limit` items. Tied rows are ordered by `emp_code` ascending.

4. **Headcount.**
The pipeline originates from `employees` (matching `joined_on <= end_of_month`), grouping by department to establish baseline headcount before executing a `$lookup` against `attendance_logs`. Employees with zero logs still count because the base stream is the employee roster, not the attendance records.

5. **One thing you would change for 100x the data.**
We would maintain an `attendance_rollups` collection updated incrementally at punch-out and regularization, allowing monthly summaries and leaderboard queries to read pre-aggregated totals via simple index seeks rather than running real-time aggregations across millions of documents.
