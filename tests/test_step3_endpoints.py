"""
Unit tests for Step 3:
1. POST /employees
2. GET /employees
3. POST /attendance/punch-in
4. GET /attendance

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
from pymongo.errors import DuplicateKeyError


class TestStep3Endpoints(unittest.TestCase):

    def setUp(self):
        self.mock_db = MagicMock()
        self.db_patcher = patch.object(app_main, "db", self.mock_db)
        self.db_patcher.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.db_patcher.stop()

    # =========================================================================
    # 1. POST /employees
    # =========================================================================
    def test_create_employee_success(self):
        self.mock_db.employees.find_one.return_value = None
        self.mock_db.employees.insert_one.return_value = MagicMock(inserted_id="fake_id")

        payload = {
            "emp_code": "EMP0001",
            "name": "Asha Rao",
            "email": "asha@example.com",
            "department": "Engineering",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-05",
        }
        resp = self.client.post("/employees", json=payload)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["emp_code"], "EMP0001")
        self.assertEqual(data["name"], "Asha Rao")
        self.assertEqual(data["email"], "asha@example.com")
        self.assertEqual(data["department"], "Engineering")
        self.assertEqual(data["shift_start"], "09:30")
        self.assertEqual(data["shift_end"], "18:30")
        self.assertEqual(data["joined_on"], "2026-01-05")
        self.assertIsInstance(data["created_at"], int)
        self.assertNotIn("_id", data)
        self.assertNotIn("id", data)

    def test_create_employee_duplicate_precheck(self):
        self.mock_db.employees.find_one.return_value = {"emp_code": "EMP0001"}

        payload = {
            "emp_code": "EMP0001",
            "name": "Duplicate",
            "email": "dup@example.com",
            "department": "Engineering",
            "joined_on": "2026-01-05",
        }
        resp = self.client.post("/employees", json=payload)
        self.assertEqual(resp.status_code, 409)

    def test_create_employee_duplicate_race_condition(self):
        self.mock_db.employees.find_one.return_value = None
        self.mock_db.employees.insert_one.side_effect = DuplicateKeyError("E11000 duplicate key")

        payload = {
            "emp_code": "EMP0001",
            "name": "Duplicate Race",
            "email": "dup@example.com",
            "department": "Engineering",
            "joined_on": "2026-01-05",
        }
        resp = self.client.post("/employees", json=payload)
        self.assertEqual(resp.status_code, 409)

    def test_create_employee_validation_errors(self):
        # 1. Invalid emp_code format (must be EMP followed by 4-6 digits)
        resp1 = self.client.post(
            "/employees",
            json={
                "emp_code": "INVALID",
                "name": "Test",
                "email": "test@example.com",
                "department": "Engineering",
                "joined_on": "2026-01-05",
            },
        )
        self.assertEqual(resp1.status_code, 422)

        # 2. shift_start == shift_end (must differ)
        resp2 = self.client.post(
            "/employees",
            json={
                "emp_code": "EMP0001",
                "name": "Test",
                "email": "test@example.com",
                "department": "Engineering",
                "shift_start": "09:30",
                "shift_end": "09:30",
                "joined_on": "2026-01-05",
            },
        )
        self.assertEqual(resp2.status_code, 422)

        # 3. Invalid email format
        resp3 = self.client.post(
            "/employees",
            json={
                "emp_code": "EMP0001",
                "name": "Test",
                "email": "not-an-email",
                "department": "Engineering",
                "joined_on": "2026-01-05",
            },
        )
        self.assertEqual(resp3.status_code, 422)

        # 4. Invalid joined_on date (e.g. Feb 30)
        resp4 = self.client.post(
            "/employees",
            json={
                "emp_code": "EMP0001",
                "name": "Test",
                "email": "test@example.com",
                "department": "Engineering",
                "joined_on": "2026-02-30",
            },
        )
        self.assertEqual(resp4.status_code, 422)

    # =========================================================================
    # 2. GET /employees
    # =========================================================================
    def test_list_employees_success(self):
        mock_records = [
            {
                "emp_code": "EMP0001",
                "name": "Asha Rao",
                "email": "asha@example.com",
                "department": "Engineering",
                "shift_start": "09:30",
                "shift_end": "18:30",
                "joined_on": "2026-01-05",
                "created_at": datetime(2026, 1, 5, 4, 30, 0, tzinfo=timezone.utc),
            }
        ]

        chain = MagicMock()
        chain.sort.return_value = chain
        chain.skip.return_value = chain
        chain.limit.return_value = mock_records

        self.mock_db.employees.count_documents.return_value = 1
        self.mock_db.employees.find.return_value = chain

        resp = self.client.get("/employees?department=Engineering&page=1&page_size=20")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["page_size"], 20)
        self.assertEqual(len(data["items"]), 1)
        emp = data["items"][0]
        self.assertEqual(emp["emp_code"], "EMP0001")
        self.assertEqual(emp["created_at"], 1767587400000)
        self.assertNotIn("_id", emp)

        self.mock_db.employees.count_documents.assert_called_once_with({"department": "Engineering"})
        self.mock_db.employees.find.assert_called_once_with({"department": "Engineering"}, {"_id": 0})
        chain.sort.assert_called_once_with("emp_code", 1)
        chain.skip.assert_called_once_with(0)
        chain.limit.assert_called_once_with(20)

    def test_list_employees_pagination_validation(self):
        resp_zero_page = self.client.get("/employees?page=0")
        self.assertEqual(resp_zero_page.status_code, 422)

        resp_oversize = self.client.get("/employees?page_size=101")
        self.assertEqual(resp_oversize.status_code, 422)

    # =========================================================================
    # 3. POST /attendance/punch-in
    # =========================================================================
    def test_punch_in_success_regular_shift(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        self.mock_db.attendance_logs.find_one.return_value = None
        self.mock_db.attendance_logs.insert_one.return_value = MagicMock(inserted_id="fake_att_id")

        payload = {
            "emp_code": "EMP0001",
            "punched_at": 1783311001000,
            "status": "PRESENT",
        }
        resp = self.client.post("/attendance/punch-in", json=payload)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["emp_code"], "EMP0001")
        self.assertEqual(data["date"], "2026-07-06")
        self.assertEqual(data["status"], "PRESENT")
        self.assertEqual(data["punch_in"], 1783311001000)
        self.assertIsNone(data["punch_out"])
        self.assertIsNone(data["work_hours"])
        self.assertEqual(data["late_minutes"], 10)
        self.assertEqual(data["overtime_minutes"], 0)
        self.assertFalse(data["half_day"])
        self.assertEqual(data["history"], [])
        self.assertNotIn("_id", data)
        self.assertNotIn("id", data)

    def test_punch_in_overnight_shift(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0005",
            "shift_start": "22:00",
            "shift_end": "06:00",
        }
        self.mock_db.attendance_logs.find_one.return_value = None
        self.mock_db.attendance_logs.insert_one.return_value = MagicMock(inserted_id="fake_att_id")

        payload = {
            "emp_code": "EMP0005",
            "punched_at": 1783382400000,
            "status": "PRESENT",
        }
        resp = self.client.post("/attendance/punch-in", json=payload)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["emp_code"], "EMP0005")
        self.assertEqual(data["date"], "2026-07-06")

    def test_punch_in_unknown_employee(self):
        self.mock_db.employees.find_one.return_value = None
        resp = self.client.post("/attendance/punch-in", json={"emp_code": "UNKNOWN"})
        self.assertEqual(resp.status_code, 404)

    def test_punch_in_duplicate_precheck(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        self.mock_db.attendance_logs.find_one.return_value = {"emp_code": "EMP0001", "date": "2026-07-06"}
        resp = self.client.post("/attendance/punch-in", json={"emp_code": "EMP0001"})
        self.assertEqual(resp.status_code, 409)

    def test_punch_in_duplicate_race_condition(self):
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "shift_start": "09:30",
            "shift_end": "18:30",
        }
        self.mock_db.attendance_logs.find_one.return_value = None
        self.mock_db.attendance_logs.insert_one.side_effect = DuplicateKeyError("E11000 duplicate key")

        resp = self.client.post("/attendance/punch-in", json={"emp_code": "EMP0001"})
        self.assertEqual(resp.status_code, 409)

    def test_punch_in_invalid_payloads(self):
        # 1. Invalid status (ABSENT is not allowed at punch in)
        resp1 = self.client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "status": "ABSENT"})
        self.assertEqual(resp1.status_code, 422)

        # 2. Punched_at looks like seconds (< 100000000000)
        resp2 = self.client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": 1783310401})
        self.assertEqual(resp2.status_code, 422)

    # =========================================================================
    # 4. GET /attendance
    # =========================================================================
    def test_list_attendance_success(self):
        mock_records = [
            {
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc),
                "punch_out": datetime(2026, 7, 6, 13, 5, 0, tzinfo=timezone.utc),
                "work_hours": 9.12,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
                "history": [],
            }
        ]

        chain = MagicMock()
        chain.sort.return_value = chain
        chain.skip.return_value = chain
        chain.limit.return_value = mock_records

        self.mock_db.attendance_logs.count_documents.return_value = 1
        self.mock_db.attendance_logs.find.return_value = chain

        resp = self.client.get(
            "/attendance?emp_code=EMP0001&date_from=2026-07-01&date_to=2026-07-31&status=PRESENT&page=1&page_size=20"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["page_size"], 20)
        self.assertEqual(len(data["items"]), 1)
        rec = data["items"][0]
        self.assertEqual(rec["emp_code"], "EMP0001")
        self.assertEqual(rec["date"], "2026-07-06")
        self.assertEqual(rec["punch_in"], 1783310280000)
        self.assertEqual(rec["punch_out"], 1783343100000)
        self.assertEqual(rec["work_hours"], 9.12)
        self.assertNotIn("_id", rec)
        self.assertNotIn("id", rec)

        self.mock_db.attendance_logs.count_documents.assert_called_once_with({
            "emp_code": "EMP0001",
            "date": {"$gte": "2026-07-01", "$lte": "2026-07-31"},
            "status": "PRESENT",
        })
        self.mock_db.attendance_logs.find.assert_called_once_with({
            "emp_code": "EMP0001",
            "date": {"$gte": "2026-07-01", "$lte": "2026-07-31"},
            "status": "PRESENT",
        }, {"_id": 0})
        chain.sort.assert_called_once_with([("date", -1), ("emp_code", 1)])
        chain.skip.assert_called_once_with(0)
        chain.limit.assert_called_once_with(20)

    def test_list_attendance_validations(self):
        # 1. date_from > date_to -> 422
        resp1 = self.client.get("/attendance?date_from=2026-07-10&date_to=2026-07-05")
        self.assertEqual(resp1.status_code, 422)

        # 2. Invalid date format
        resp2 = self.client.get("/attendance?date_from=2026-13-45")
        self.assertEqual(resp2.status_code, 422)

        # 3. Invalid status
        resp3 = self.client.get("/attendance?status=INVALID_STATUS")
        self.assertEqual(resp3.status_code, 422)

        # 4. Invalid pagination (page < 1)
        resp4 = self.client.get("/attendance?page=0")
        self.assertEqual(resp4.status_code, 422)


if __name__ == "__main__":
    unittest.main()
