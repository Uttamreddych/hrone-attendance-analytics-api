"""
Test suite for Step 1 and Step 2 implementation.
Validates:
- Timezone and whole-second truncation (R1)
- Epoch-millisecond conversions (R1)
- Overnight shift detection & attendance date calculation (R1)
- Late minutes calculation with strict 10-minute grace (R2)
- Overtime calculation with >= 30-minute threshold & overnight support (R3)
- Work hours calculation with HALF-UP rounding (R4)
- Half day calculation (R5)
- /health endpoint error handling when MongoDB is offline
"""
import sys
import unittest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pathlib

KIT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

from app.main import (
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
    round_half_up,
    app,
)
from fastapi.testclient import TestClient


class TestStep1And2(unittest.TestCase):

    def test_round_half_up(self):
        # Python's built-in round(2.555, 2) is 2.55 (banker's rounding).
        # Half-up rounding MUST round 2.555 up to 2.56.
        self.assertEqual(round_half_up(Decimal("2.555"), 2), 2.56)
        self.assertEqual(round_half_up(Decimal("2.554"), 2), 2.55)
        self.assertEqual(round_half_up(Decimal("4.495"), 2), 4.50)
        self.assertEqual(round_half_up(Decimal("0.12345"), 4), 0.1235)
        self.assertEqual(round_half_up(Decimal("0.12344"), 4), 0.1234)

    def test_truncate_to_seconds(self):
        dt = datetime(2026, 7, 6, 9, 40, 0, 950000, tzinfo=IST)
        truncated = truncate_to_seconds(dt)
        self.assertEqual(truncated.microsecond, 0)
        self.assertEqual(truncated.second, 0)
        self.assertEqual(truncated.minute, 40)

    def test_epoch_ms_conversions(self):
        # 2026-07-06T03:58:00Z = 1783310280000 ms
        ms_in = 1783310280999  # with sub-second millis
        dt = epoch_ms_to_datetime(ms_in)
        self.assertEqual(dt.microsecond, 0)
        self.assertEqual(dt.tzinfo, timezone.utc)
        ms_out = datetime_to_epoch_ms(dt)
        self.assertEqual(ms_out, 1783310280000)
        self.assertIsNone(datetime_to_epoch_ms(None))

    def test_is_overnight_shift(self):
        # Regular shift: 09:30 to 18:30 -> False
        self.assertFalse(is_overnight_shift("09:30", "18:30"))
        # Overnight shifts:
        self.assertTrue(is_overnight_shift("22:00", "06:00"))
        self.assertTrue(is_overnight_shift("20:00", "04:00"))
        self.assertTrue(is_overnight_shift("08:00", "08:00"))

    def test_attendance_date_regular(self):
        # Regular shift 09:30-18:30. Punch-in at 09:28 IST on 2026-07-06
        dt = datetime(2026, 7, 6, 9, 28, 0, tzinfo=IST)
        date_str = get_attendance_date(dt, "09:30", "18:30")
        self.assertEqual(date_str, "2026-07-06")

    def test_attendance_date_overnight(self):
        # Overnight shift 22:00 - 06:00
        # Case A: Punched in before midnight on 2026-07-06 at 21:55 IST
        dt1 = datetime(2026, 7, 6, 21, 55, 0, tzinfo=IST)
        self.assertEqual(get_attendance_date(dt1, "22:00", "06:00"), "2026-07-06")

        # Case B: Punched in after midnight on 2026-07-07 at 05:30 IST
        # Earlier than shift_end (06:00), belongs to previous day (2026-07-06)
        dt2 = datetime(2026, 7, 7, 5, 30, 0, tzinfo=IST)
        self.assertEqual(get_attendance_date(dt2, "22:00", "06:00"), "2026-07-06")

        # Case C: Punched in at or after shift_end (e.g. 06:15 IST on 2026-07-07)
        # Belongs to 2026-07-07
        dt3 = datetime(2026, 7, 7, 6, 15, 0, tzinfo=IST)
        self.assertEqual(get_attendance_date(dt3, "22:00", "06:00"), "2026-07-07")

    def test_late_minutes_r2_spec_examples(self):
        # Shift 09:30
        # 1. 09:40:00 -> 0 (exactly on 10 min grace)
        dt1 = datetime(2026, 7, 6, 9, 40, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt1, "09:30", "2026-07-06"), 0)

        # 2. 09:40:01 -> 10 (strictly more than 10:00 -> floor from shift_start is 10)
        dt2 = datetime(2026, 7, 6, 9, 40, 1, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt2, "09:30", "2026-07-06"), 10)

        # 3. 10:15:59 -> 45
        dt3 = datetime(2026, 7, 6, 10, 15, 59, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt3, "09:30", "2026-07-06"), 45)

        # 4. Early arrival (09:15) -> 0
        dt4 = datetime(2026, 7, 6, 9, 15, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt4, "09:30", "2026-07-06"), 0)

        # 5. Sample 2 from dataset: 10:05 IST -> 35 late minutes
        dt5 = datetime(2026, 7, 6, 10, 5, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt5, "09:30", "2026-07-06"), 35)

    def test_overtime_r3(self):
        # Shift 09:30 - 18:30 on 2026-07-06
        # Case A: Sample 1: punch out at 18:35 (5 min overtime, < 30) -> 0
        out1 = datetime(2026, 7, 6, 18, 35, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(out1, "18:30", "2026-07-06", "09:30"), 0)

        # Case B: Exactly 29 minutes after shift_end -> 0
        out2 = datetime(2026, 7, 6, 18, 59, 59, tzinfo=IST)
        self.assertEqual(compute_overtime(out2, "18:30", "2026-07-06", "09:30"), 0)

        # Case C: Exactly 30 minutes after shift_end (19:00:00) -> 30
        out3 = datetime(2026, 7, 6, 19, 0, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(out3, "18:30", "2026-07-06", "09:30"), 30)

        # Case D: Sample 2: punch out at 19:10 (40 min overtime) -> 40
        out4 = datetime(2026, 7, 6, 19, 10, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(out4, "18:30", "2026-07-06", "09:30"), 40)

        # Case E: Sample 7: Overnight shift 22:00 - 06:00 on 2026-07-06.
        # Punch-out at 06:40 IST on 2026-07-07 -> 40 minutes overtime
        out5 = datetime(2026, 7, 7, 6, 40, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(out5, "06:00", "2026-07-06", "22:00"), 40)

    def test_work_hours_r4(self):
        # Sample 1: 09:28 to 18:35 -> 32820 seconds -> 9.116666... -> 9.12
        p_in1 = datetime(2026, 7, 6, 9, 28, 0, tzinfo=IST)
        p_out1 = datetime(2026, 7, 6, 18, 35, 0, tzinfo=IST)
        self.assertEqual(compute_work_hours(p_in1, p_out1), 9.12)

        # Sample 2: 10:05 to 19:10 -> 32700 seconds -> 9.083333... -> 9.08
        p_in2 = datetime(2026, 7, 6, 10, 5, 0, tzinfo=IST)
        p_out2 = datetime(2026, 7, 6, 19, 10, 0, tzinfo=IST)
        self.assertEqual(compute_work_hours(p_in2, p_out2), 9.08)

        # Sample 4: 09:30 to 13:00 (3.5 hours)
        p_in4 = datetime(2026, 7, 7, 9, 30, 0, tzinfo=IST)
        p_out4 = datetime(2026, 7, 7, 13, 0, 0, tzinfo=IST)
        self.assertEqual(compute_work_hours(p_in4, p_out4), 3.5)

        # Sample 7: 21:55 (July 6) to 06:40 (July 7) -> 8.75 hours
        p_in7 = datetime(2026, 7, 6, 21, 55, 0, tzinfo=IST)
        p_out7 = datetime(2026, 7, 7, 6, 40, 0, tzinfo=IST)
        self.assertEqual(compute_work_hours(p_in7, p_out7), 8.75)

    def test_half_day_r5(self):
        self.assertTrue(compute_half_day(3.50))
        self.assertTrue(compute_half_day(4.49))
        self.assertFalse(compute_half_day(4.50))
        self.assertFalse(compute_half_day(9.12))
        self.assertFalse(compute_half_day(None))

    def test_health_when_mongodb_down(self):
        with TestClient(app) as client:
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 503)
            self.assertEqual(resp.json(), {"detail": "Database unavailable"})

    def test_health_when_mongodb_up(self):
        from unittest.mock import patch
        from pymongo.database import Database
        with patch.object(Database, "command", return_value={"ok": 1}):
            with TestClient(app) as test_client:
                resp = test_client.get("/health")
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
