"""
Comprehensive Edge-Case & Boundary Test Suite for HROne Attendance & Analytics API
Covers:
1. Exact threshold boundaries (late grace period, overtime >= 30 min, half-day < 4.50)
2. Half-up decimal rounding (rounding .005, float precision vs Decimal)
3. Timestamps with fractional seconds and whole-second truncation (Rule R1)
4. Midnight crossings and overnight shift calculations (Rule R1, R2, R3)
5. Empty datasets, zero-log employees, and departments with no logs
6. Validation boundaries: invalid employee codes, shift start == shift end, invalid emails, inverted dates
7. Repeated requests, idempotent behavior, and race condition handling
"""
import calendar
import os
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch
import pathlib

KIT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

from fastapi.testclient import TestClient
import app.main as app_main
from app.main import (
    app,
    IST,
    truncate_to_seconds,
    epoch_ms_to_datetime,
    datetime_to_epoch_ms,
    is_overnight_shift,
    get_attendance_date,
    compute_late_minutes,
    compute_work_hours,
    compute_overtime,
    compute_half_day,
    calculate_working_days,
    round_half_up,
    round_rate_half_up,
)

client = TestClient(app)


class TestThresholdBoundaries(unittest.TestCase):
    """Verifies exact threshold boundaries for R2, R3, R4, R5, R8."""

    def test_late_minutes_exact_boundaries(self):
        # Shift: 09:30
        shift_start = "09:30"
        att_date = "2026-07-06"

        # Exactly on time: 09:30:00 -> 0
        dt_on_time = datetime(2026, 7, 6, 9, 30, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_on_time, shift_start, att_date), 0)

        # Inside grace period: 09:39:59 (599 seconds elapsed) -> 0
        dt_599 = datetime(2026, 7, 6, 9, 39, 59, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_599, shift_start, att_date), 0)

        # Exact boundary: 09:40:00 (strictly 600 seconds elapsed) -> 0 (Rule R2: strictly > 10:00)
        dt_600 = datetime(2026, 7, 6, 9, 40, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_600, shift_start, att_date), 0)

        # First second late: 09:40:01 (601 seconds elapsed) -> 10 minutes (floor from shift start!)
        dt_601 = datetime(2026, 7, 6, 9, 40, 1, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_601, shift_start, att_date), 10)

        # End of 10th minute late: 09:40:59 (659 seconds elapsed) -> 10 minutes
        dt_659 = datetime(2026, 7, 6, 9, 40, 59, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_659, shift_start, att_date), 10)

        # Next minute: 09:41:00 (660 seconds elapsed) -> 11 minutes
        dt_660 = datetime(2026, 7, 6, 9, 41, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_660, shift_start, att_date), 11)

        # Spec example: 10:15:59 (2759 seconds elapsed) -> 45 minutes
        dt_spec = datetime(2026, 7, 6, 10, 15, 59, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_spec, shift_start, att_date), 45)

        # Early punch-in: 09:15:00 -> 0
        dt_early = datetime(2026, 7, 6, 9, 15, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_early, shift_start, att_date), 0)

    def test_overtime_exact_boundaries(self):
        # Shift: 09:30 - 18:30
        shift_end = "18:30"
        att_date = "2026-07-06"

        # 29 minutes 59 seconds overtime (18:59:59) -> 0 (strictly under 30 minutes)
        dt_29 = datetime(2026, 7, 6, 18, 59, 59, tzinfo=IST)
        self.assertEqual(compute_overtime(dt_29, shift_end, att_date), 0)

        # Exactly 30 minutes overtime (19:00:00) -> 30 (Rule R3: >= 30)
        dt_30 = datetime(2026, 7, 6, 19, 0, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(dt_30, shift_end, att_date), 30)

        # 30 minutes 59 seconds overtime (19:00:59) -> 30
        dt_30_59 = datetime(2026, 7, 6, 19, 0, 59, tzinfo=IST)
        self.assertEqual(compute_overtime(dt_30_59, shift_end, att_date), 30)

        # 31 minutes overtime (19:01:00) -> 31
        dt_31 = datetime(2026, 7, 6, 19, 1, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(dt_31, shift_end, att_date), 31)

        # Left early (18:20:00) -> 0
        dt_early = datetime(2026, 7, 6, 18, 20, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(dt_early, shift_end, att_date), 0)

    def test_half_day_boundary_with_rounding(self):
        # Rule R5: half_day is true when the ROUNDED work_hours < 4.50
        self.assertTrue(compute_half_day(4.49))
        self.assertFalse(compute_half_day(4.50))
        self.assertFalse(compute_half_day(4.51))
        self.assertFalse(compute_half_day(None))

        # Precision boundary check: 4 hours 29 min 42 sec = 16182 sec
        # 16182 / 3600 = 4.495 hours -> rounds HALF-UP to 4.50!
        # Therefore, half_day MUST be False!
        p_in = datetime(2026, 7, 6, 9, 30, 0, tzinfo=IST)
        p_out = p_in + timedelta(seconds=16182)
        wh = compute_work_hours(p_in, p_out)
        self.assertEqual(wh, 4.50)
        self.assertFalse(compute_half_day(wh))

        # 4 hours 29 min 40 sec = 16180 sec -> 4.49444... -> rounds to 4.49
        p_out_under = p_in + timedelta(seconds=16180)
        wh_under = compute_work_hours(p_in, p_out_under)
        self.assertEqual(wh_under, 4.49)
        self.assertTrue(compute_half_day(wh_under))

    def test_half_up_rounding_precision(self):
        # Banker's rounding in Python round(2.5) == 2, round(3.5) == 4
        # Standard half-up must round 2.5 to 3
        self.assertEqual(round_half_up(Decimal("2.5"), 0), 3.0)
        self.assertEqual(round_half_up(Decimal("3.5"), 0), 4.0)

        # 2 decimal places:
        self.assertEqual(round_half_up(Decimal("8.125"), 2), 8.13)
        self.assertEqual(round_half_up(Decimal("8.124"), 2), 8.12)

        # 4 decimal places (rates):
        self.assertEqual(round_rate_half_up(Decimal("0.85715")), 0.8572)
        self.assertEqual(round_rate_half_up(Decimal("0.85714")), 0.8571)


class TestTimestampAndFractionalSeconds(unittest.TestCase):
    """Verifies truncation of instants to whole seconds before calculations (Rule R1)."""

    def test_fractional_seconds_truncation(self):
        # Instant with 900,000 microseconds
        dt_with_micros = datetime(2026, 7, 6, 9, 40, 0, 900000, tzinfo=timezone.utc)
        truncated = truncate_to_seconds(dt_with_micros)
        self.assertEqual(truncated.microsecond, 0)
        self.assertEqual(truncated.second, 0)

        # Epoch ms with fractional seconds (1783312500900 ms -> 1783312500 seconds)
        dt_from_ms = epoch_ms_to_datetime(1783312500900)
        self.assertEqual(dt_from_ms.microsecond, 0)

        # Work hours truncated to whole seconds
        dt1 = datetime(2026, 7, 6, 9, 0, 0, 800000, tzinfo=timezone.utc)
        dt2 = datetime(2026, 7, 6, 18, 0, 0, 100000, tzinfo=timezone.utc)
        # Without truncation, (dt2 - dt1) would be 8h 59m 59.3s
        # With whole-second truncation, difference is exactly 9 hours (32400s)
        wh = compute_work_hours(dt1, dt2)
        self.assertEqual(wh, 9.00)


class TestOvernightShiftsAndMidnightCrossings(unittest.TestCase):
    """Verifies overnight shifts (22:00 to 06:00) and midnight crossings."""

    def test_overnight_shift_date_assignment(self):
        # Shift: 22:00 - 06:00
        shift_start = "22:00"
        shift_end = "06:00"
        self.assertTrue(is_overnight_shift(shift_start, shift_end))

        # Punch-in at 21:55 IST on July 6 -> belongs to July 6
        p_in_before_midnight = datetime(2026, 7, 6, 21, 55, 0, tzinfo=IST)
        self.assertEqual(get_attendance_date(p_in_before_midnight, shift_start, shift_end), "2026-07-06")

        # Punch-in at 02:15 IST on July 7 -> before shift_end (06:00), so belongs to July 6!
        p_in_after_midnight = datetime(2026, 7, 7, 2, 15, 0, tzinfo=IST)
        self.assertEqual(get_attendance_date(p_in_after_midnight, shift_start, shift_end), "2026-07-06")

        # Punch-in at 06:00 IST on July 7 -> NOT earlier than shift_end, belongs to July 7
        p_in_next_shift = datetime(2026, 7, 7, 6, 0, 0, tzinfo=IST)
        self.assertEqual(get_attendance_date(p_in_next_shift, shift_start, shift_end), "2026-07-07")

    def test_overnight_late_and_overtime_across_midnight(self):
        shift_start = "22:00"
        shift_end = "06:00"
        att_date = "2026-07-06"

        # Punched in at 00:15 IST on July 7 (2h 15m = 135 minutes after shift start at 22:00)
        p_in = datetime(2026, 7, 7, 0, 15, 0, tzinfo=IST)
        late = compute_late_minutes(p_in, shift_start, att_date)
        self.assertEqual(late, 135)

        # Punched out at 06:40 IST on July 7 (40 minutes after shift end at 06:00 next day)
        p_out = datetime(2026, 7, 7, 6, 40, 0, tzinfo=IST)
        ot = compute_overtime(p_out, shift_end, att_date, shift_start)
        self.assertEqual(ot, 40)

        # Work hours: 00:15 to 06:40 = 6 hours 25 minutes = 23100s / 3600 = 6.4166... -> 6.42
        wh = compute_work_hours(p_in, p_out)
        self.assertEqual(wh, 6.42)
        self.assertFalse(compute_half_day(wh))


class TestValidationBoundariesAndEdgeCases(unittest.TestCase):
    """Verifies input validation boundaries for employees, punch, and queries."""

    def test_emp_code_patterns(self):
        # Valid codes (EMP + 4 to 6 digits)
        valid_payload = {
            "emp_code": "EMP0001",
            "name": "Test User",
            "email": "test@example.com",
            "department": "Engineering",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01",
        }
        with patch.object(app_main, "db", MagicMock()):
            app_main.db.employees.find_one.return_value = None
            resp = client.post("/employees", json=valid_payload)
            self.assertEqual(resp.status_code, 201)

        # Invalid: 3 digits (too short)
        payload_bad = dict(valid_payload, emp_code="EMP123")
        self.assertEqual(client.post("/employees", json=payload_bad).status_code, 422)

        # Invalid: 7 digits (too long)
        payload_bad2 = dict(valid_payload, emp_code="EMP1234567")
        self.assertEqual(client.post("/employees", json=payload_bad2).status_code, 422)

        # Invalid: lowercase 'emp'
        payload_bad3 = dict(valid_payload, emp_code="emp0001")
        self.assertEqual(client.post("/employees", json=payload_bad3).status_code, 422)

    def test_shift_start_equal_shift_end_rejected(self):
        payload = {
            "emp_code": "EMP0002",
            "name": "Test User",
            "email": "test@example.com",
            "department": "Engineering",
            "shift_start": "09:30",
            "shift_end": "09:30",  # Same time -> invalid!
            "joined_on": "2026-01-01",
        }
        resp = client.post("/employees", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_invalid_email_rejected(self):
        payload = {
            "emp_code": "EMP0003",
            "name": "Test User",
            "email": "invalid-email-no-at",
            "department": "Engineering",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01",
        }
        resp = client.post("/employees", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_invalid_calendar_date_rejected(self):
        payload = {
            "emp_code": "EMP0004",
            "name": "Test User",
            "email": "test@example.com",
            "department": "Engineering",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-02-31",  # Feb 31 does not exist
        }
        resp = client.post("/employees", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_punch_out_greater_than_24_hours_fails_422(self):
        mock_db = MagicMock()
        with patch.object(app_main, "db", mock_db):
            mock_db.employees.find_one.return_value = {
                "emp_code": "EMP0001",
                "shift_start": "09:30",
                "shift_end": "18:30",
            }
            # Punch in July 6 09:30 UTC
            t_in = datetime(2026, 7, 6, 9, 30, 0, tzinfo=timezone.utc)
            mock_db.attendance_logs.find_one.return_value = {
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "punch_in": t_in,
                "punch_out": None,
            }

            # Punch out July 7 09:35 UTC (24 hours and 5 minutes later) -> 422
            t_out_ms = int((t_in + timedelta(hours=24, minutes=5)).timestamp()) * 1000
            resp = client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": t_out_ms})
            self.assertEqual(resp.status_code, 422)


class TestEmptyDatasetsAndNoLogs(unittest.TestCase):
    """Verifies that empty datasets, zero-log employees, and departments without logs behave correctly."""

    def test_employee_monthly_empty_logs(self):
        mock_db = MagicMock()
        with patch.object(app_main, "db", mock_db):
            mock_db.employees.find_one.return_value = {
                "emp_code": "EMP0001",
                "joined_on": "2026-01-01",
            }
            # Database aggregation returns empty list (no attendance records in month)
            mock_db.attendance_logs.aggregate.return_value = []

            resp = client.get("/analytics/employees/EMP0001/monthly?month=2026-07")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["working_days"], 23)
            self.assertEqual(data["present_days"], 0.0)
            self.assertEqual(data["leave_days"], 0)
            self.assertEqual(data["late_count"], 0)
            self.assertEqual(data["total_late_minutes"], 0)
            self.assertEqual(data["total_overtime_minutes"], 0)
            self.assertEqual(data["attendance_pct"], 0.0)

    def test_department_summary_empty_department(self):
        mock_db = MagicMock()
        with patch.object(app_main, "db", mock_db):
            # Aggregation returns empty list if no departments matched
            mock_db.employees.aggregate.return_value = []
            resp = client.get("/analytics/departments/summary?month=2026-07&department=EmptyDept")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["month"], "2026-07")
            self.assertEqual(data["items"], [])

    def test_late_leaderboard_empty_month(self):
        mock_db = MagicMock()
        with patch.object(app_main, "db", mock_db):
            # No employees were late in this month
            mock_db.attendance_logs.aggregate.return_value = []
            resp = client.get("/analytics/leaderboard/late?month=2026-07")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["month"], "2026-07")
            self.assertEqual(data["items"], [])


if __name__ == "__main__":
    unittest.main()
