"""
Unit tests for Step 4:
1. POST /attendance/punch-out
2. PATCH /attendance/{emp_code}/{date}

Uses mocked MongoDB collections via app.main.db so tests run cleanly
and independently without requiring an active MongoDB connection.
"""
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pathlib

KIT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import app.main as app_main
from app.main import app, IST
from fastapi.testclient import TestClient


class TestStep4Endpoints(unittest.TestCase):

    def setUp(self):
        self.mock_db = MagicMock()
        self.db_patcher = patch.object(app_main, "db", self.mock_db)
        self.db_patcher.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.db_patcher.stop()

    # =========================================================================
    # 1. POST /attendance/punch-out
    # =========================================================================
    def test_punch_out_success_regular(self):
        # EMP0001: 09:30 - 18:30 shift
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        # Punched in at 09:28 IST (03:58 UTC) on 2026-07-06 = 1783310280000 ms
        p_in_dt = datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": p_in_dt,
            "punch_out": None,
            "work_hours": None,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }

        # Punch out at 18:35 IST (13:05 UTC) on 2026-07-06 = 1783343100000 ms
        p_out_dt = datetime(2026, 7, 6, 13, 5, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one_and_update.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": p_in_dt,
            "punch_out": p_out_dt,
            "work_hours": 9.12,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }

        resp = self.client.post(
            "/attendance/punch-out",
            json={"emp_code": "EMP0001", "punched_at": 1783343100000},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["emp_code"], "EMP0001")
        self.assertEqual(data["date"], "2026-07-06")
        self.assertEqual(data["punch_in"], 1783310280000)
        self.assertEqual(data["punch_out"], 1783343100000)
        self.assertEqual(data["work_hours"], 9.12)
        self.assertEqual(data["overtime_minutes"], 0)
        self.assertFalse(data["half_day"])
        self.assertEqual(data["history"], [])

    def test_punch_out_overtime_and_overnight(self):
        # EMP0005: overnight shift 22:00 - 06:00
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0005",
            "shift_start": "22:00",
            "shift_end": "06:00",
        }
        # Punched in at 21:55 IST on 2026-07-06 (16:25 UTC)
        p_in_dt = datetime(2026, 7, 6, 16, 25, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0005",
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": p_in_dt,
            "punch_out": None,
            "work_hours": None,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }

        # Punch out at 06:40 IST on 2026-07-07 (01:10 UTC = 1783386600000 ms)
        # 40 min overtime (>= 30 threshold)
        p_out_dt = datetime(2026, 7, 7, 1, 10, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one_and_update.return_value = {
            "emp_code": "EMP0005",
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": p_in_dt,
            "punch_out": p_out_dt,
            "work_hours": 8.75,
            "late_minutes": 0,
            "overtime_minutes": 40,
            "half_day": False,
            "history": [],
        }

        resp = self.client.post(
            "/attendance/punch-out",
            json={"emp_code": "EMP0005", "punched_at": 1783386600000},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["work_hours"], 8.75)
        self.assertEqual(data["overtime_minutes"], 40)

    def test_punch_out_unknown_employee(self):
        self.mock_db.employees.find_one.return_value = None
        resp = self.client.post("/attendance/punch-out", json={"emp_code": "UNKNOWN"})
        self.assertEqual(resp.status_code, 404)

    def test_punch_out_missing_punch_in_record(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        self.mock_db.attendance_logs.find_one.return_value = None
        resp = self.client.post("/attendance/punch-out", json={"emp_code": "EMP0001"})
        self.assertEqual(resp.status_code, 404)

    def test_punch_out_already_punched_out(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        p_in_dt = datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc)
        p_out_dt = datetime(2026, 7, 6, 13, 5, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": p_in_dt,
            "punch_out": p_out_dt,
        }
        resp = self.client.post("/attendance/punch-out", json={"emp_code": "EMP0001"})
        self.assertEqual(resp.status_code, 409)

    def test_punch_out_concurrent_conflict(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        p_in_dt = datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": p_in_dt,
            "punch_out": None,
        }
        # Simulate race where another request completed punch-out first
        self.mock_db.attendance_logs.find_one_and_update.return_value = None
        resp = self.client.post(
            "/attendance/punch-out",
            json={"emp_code": "EMP0001", "punched_at": 1783343100000},
        )
        self.assertEqual(resp.status_code, 409)

    def test_punch_out_invalid_timestamps(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        p_in_dt = datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": p_in_dt,
            "punch_out": None,
        }

        # 1. punched_at <= punch_in (03:58 UTC = 1783310280000 ms)
        resp1 = self.client.post(
            "/attendance/punch-out",
            json={"emp_code": "EMP0001", "punched_at": 1783310280000},
        )
        self.assertEqual(resp1.status_code, 422)

        # 2. punched_at > punch_in + 24 hours (86401 seconds later)
        resp2 = self.client.post(
            "/attendance/punch-out",
            json={"emp_code": "EMP0001", "punched_at": 1783310280000 + 86401000},
        )
        self.assertEqual(resp2.status_code, 422)

        # 3. punched_at below 100000000000 (looks like seconds)
        resp3 = self.client.post(
            "/attendance/punch-out",
            json={"emp_code": "EMP0001", "punched_at": 1783343100},
        )
        self.assertEqual(resp3.status_code, 422)

    # =========================================================================
    # 2. PATCH /attendance/{emp_code}/{date}
    # =========================================================================
    def test_regularize_punch_in_success(self):
        # EMP0001: shift 09:30 - 18:30
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        # Initially punched in at 10:05 IST (04:35 UTC), punch-out at 18:30 IST (13:00 UTC)
        old_p_in = datetime(2026, 7, 7, 4, 35, 0, tzinfo=timezone.utc)
        p_out = datetime(2026, 7, 7, 13, 0, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-07",
            "status": "PRESENT",
            "punch_in": old_p_in,
            "punch_out": p_out,
            "work_hours": 8.42,
            "late_minutes": 35,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }

        # Correct punch-in to 09:28 IST (03:58 UTC = 1783396680000 ms)
        new_p_in = datetime(2026, 7, 7, 3, 58, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one_and_update.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-07",
            "status": "PRESENT",
            "punch_in": new_p_in,
            "punch_out": p_out,
            "work_hours": 9.03,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [
                {
                    "at": datetime(2026, 7, 8, 5, 0, 0, tzinfo=timezone.utc),
                    "by": "hr.admin",
                    "reason": "biometric glitch",
                    "changes": {
                        "punch_in": {"from": old_p_in, "to": new_p_in},
                        "late_minutes": {"from": 35, "to": 0},
                        "work_hours": {"from": 8.42, "to": 9.03},
                    },
                }
            ],
        }

        payload = {
            "punch_in": 1783396680000,
            "reason": "biometric glitch",
            "regularized_by": "hr.admin",
        }
        resp = self.client.patch("/attendance/EMP0001/2026-07-07", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["late_minutes"], 0)
        self.assertEqual(data["work_hours"], 9.03)
        self.assertEqual(len(data["history"]), 1)
        entry = data["history"][0]
        self.assertEqual(entry["by"], "hr.admin")
        self.assertEqual(entry["reason"], "biometric glitch")
        self.assertIn("punch_in", entry["changes"])
        self.assertEqual(entry["changes"]["punch_in"]["to"], 1783396680000)

    def test_regularize_status_to_leave_clears_punches(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        p_in = datetime(2026, 7, 7, 3, 58, 0, tzinfo=timezone.utc)
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-07",
            "status": "PRESENT",
            "punch_in": p_in,
            "punch_out": None,
            "work_hours": None,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }

        self.mock_db.attendance_logs.find_one_and_update.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-07",
            "status": "LEAVE",
            "punch_in": None,
            "punch_out": None,
            "work_hours": None,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [
                {
                    "at": datetime(2026, 7, 8, 5, 0, 0, tzinfo=timezone.utc),
                    "by": "manager",
                    "reason": "approved sick leave",
                    "changes": {
                        "status": {"from": "PRESENT", "to": "LEAVE"},
                        "punch_in": {"from": p_in, "to": None},
                    },
                }
            ],
        }

        resp = self.client.patch(
            "/attendance/EMP0001/2026-07-07",
            json={"status": "LEAVE", "reason": "approved sick leave", "regularized_by": "manager"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "LEAVE")
        self.assertIsNone(data["punch_in"])

    def test_regularize_leave_with_punch_in_fails_422(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-07",
            "status": "PRESENT",
        }
        # Attempt to set LEAVE while supplying punch_in
        resp = self.client.patch(
            "/attendance/EMP0001/2026-07-07",
            json={
                "status": "LEAVE",
                "punch_in": 1783396680000,
                "reason": "should fail",
                "regularized_by": "manager",
            },
        )
        self.assertEqual(resp.status_code, 422)

    def test_regularize_changes_nothing_fails_422(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-07",
            "status": "PRESENT",
            "punch_in": datetime(2026, 7, 7, 3, 58, 0, tzinfo=timezone.utc),
            "punch_out": None,
            "work_hours": None,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }

        # Sending identical status PRESENT and nothing else
        resp = self.client.patch(
            "/attendance/EMP0001/2026-07-07",
            json={
                "status": "PRESENT",
                "reason": "no actual change",
                "regularized_by": "manager",
            },
        )
        self.assertEqual(resp.status_code, 422)

    def test_regularize_missing_resource(self):
        # 1. Missing employee
        self.mock_db.employees.find_one.return_value = None
        resp1 = self.client.patch(
            "/attendance/UNKNOWN/2026-07-07",
            json={"reason": "test reason", "regularized_by": "admin"},
        )
        self.assertEqual(resp1.status_code, 404)

        # 2. Missing record
        self.mock_db.employees.find_one.return_value = {"emp_code": "EMP0001"}
        self.mock_db.attendance_logs.find_one.return_value = None
        resp2 = self.client.patch(
            "/attendance/EMP0001/2026-07-07",
            json={"reason": "test reason", "regularized_by": "admin"},
        )
        self.assertEqual(resp2.status_code, 404)

    def test_regularize_concurrent_modification_conflict(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        self.mock_db.attendance_logs.find_one.return_value = {
            "emp_code": "EMP0001",
            "date": "2026-07-07",
            "status": "PRESENT",
            "punch_in": datetime(2026, 7, 7, 3, 58, 0, tzinfo=timezone.utc),
            "history": [],
        }
        # Simulate race where another correction committed first, causing find_one_and_update to return None
        self.mock_db.attendance_logs.find_one_and_update.return_value = None

        resp = self.client.patch(
            "/attendance/EMP0001/2026-07-07",
            json={"status": "WFH", "reason": "remote work", "regularized_by": "manager"},
        )
        self.assertEqual(resp.status_code, 409)


if __name__ == "__main__":
    unittest.main()
