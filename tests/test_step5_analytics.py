"""
Unit & Integration Tests for HROne Analytics & Explain Endpoints
Covers:
- Pipeline Builders and aggregation stages verification
- GET /analytics/employees/{emp_code}/monthly
- GET /analytics/departments/summary
- GET /analytics/leaderboard/late
- GET /analytics/departments/{department}/trend
- GET /admin/explain/{endpoint}
- Both mocked unit tests and live MongoDB integration tests
"""
import os
import sys
import unittest
from datetime import datetime, timezone
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
    build_employee_monthly_pipeline,
    build_department_summary_pipeline,
    build_late_leaderboard_pipeline,
    build_department_trend_pipeline,
    calculate_working_days,
    round_rate_half_up,
)

client = TestClient(app)


# =========================================================================== #
# 1. Pipeline Stage Verification Unit Tests
# =========================================================================== #
class TestPipelineBuilders(unittest.TestCase):
    def test_employee_monthly_pipeline_structure(self):
        pipe = build_employee_monthly_pipeline("EMP0001", "2026-07")
        self.assertIsInstance(pipe, list)
        self.assertEqual(len(pipe), 3)

        # Stage 0: $match
        self.assertEqual(pipe[0]["$match"]["emp_code"], "EMP0001")
        self.assertEqual(pipe[0]["$match"]["date"]["$gte"], "2026-07-01")
        self.assertEqual(pipe[0]["$match"]["date"]["$lte"], "2026-07-31")

        # Stage 1: $project includes is_weekday and is_present_status
        self.assertIn("is_weekday", pipe[1]["$project"])
        self.assertIn("is_present_status", pipe[1]["$project"])

        # Stage 2: $group groups by None with expected metrics
        group = pipe[2]["$group"]
        self.assertIsNone(group["_id"])
        self.assertIn("present_days", group)
        self.assertIn("leave_days", group)
        self.assertIn("late_count", group)
        self.assertIn("total_late_minutes", group)
        self.assertIn("total_overtime_minutes", group)

    def test_department_summary_pipeline_structure(self):
        # Without department filter
        pipe = build_department_summary_pipeline("2026-07")
        self.assertIsInstance(pipe, list)
        self.assertEqual(pipe[0]["$match"]["joined_on"]["$lte"], "2026-07-31")
        self.assertNotIn("department", pipe[0]["$match"])

        # Verify $group, $lookup, $project, $sort
        stages = [list(s.keys())[0] for s in pipe]
        self.assertEqual(stages, ["$match", "$group", "$lookup", "$project", "$project", "$sort"])

        # With department filter
        pipe_dept = build_department_summary_pipeline("2026-07", "Engineering")
        self.assertEqual(pipe_dept[0]["$match"]["department"], "Engineering")

    def test_late_leaderboard_pipeline_structure(self):
        pipe = build_late_leaderboard_pipeline("2026-07", limit=10, department="Sales")
        stages = [list(s.keys())[0] for s in pipe]
        self.assertIn("$setWindowFields", stages)
        self.assertIn("$match", stages)
        self.assertIn("$sort", stages)

        # Verify competition ranking with $rank
        window_stage = next(s for s in pipe if "$setWindowFields" in s)
        self.assertEqual(window_stage["$setWindowFields"]["output"]["rank"], {"$rank": {}})

        # Verify post-rank limit matching
        rank_match = next(s for s in pipe if "$match" in s and "rank" in s["$match"])
        self.assertEqual(rank_match["$match"]["rank"], {"$lte": 10})

    def test_department_trend_pipeline_structure(self):
        pipe = build_department_trend_pipeline("Engineering", "2026-07-01", "2026-07-07")
        stages = [list(s.keys())[0] for s in pipe]
        self.assertIn("$densify", stages)
        self.assertIn("$unionWith", stages)
        self.assertIn("$setWindowFields", stages)

        # Verify densify stage
        densify_stage = next(s for s in pipe if "$densify" in s)
        self.assertEqual(densify_stage["$densify"]["field"], "day_date")
        self.assertEqual(densify_stage["$densify"]["range"]["unit"], "day")

        # Verify 7-day moving average window
        window_stage = next(s for s in pipe if "$setWindowFields" in s)
        self.assertEqual(
            window_stage["$setWindowFields"]["output"]["moving_avg_raw"]["window"]["documents"],
            [-6, "current"],
        )


# =========================================================================== #
# 2. Mocked Unit Tests for Analytics Endpoints
# =========================================================================== #
class TestAnalyticsEndpointsMocked(unittest.TestCase):
    """Unit tests executing against a mocked PyMongo database."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.patcher = patch.object(app_main, "db", self.mock_db)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    # --- GET /analytics/employees/{emp_code}/monthly ---
    def test_employee_monthly_normal(self):
        # Employee Asha Rao joined on 2026-01-05
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0001",
            "name": "Asha Rao",
            "department": "Engineering",
            "joined_on": "2026-01-05",
        }
        # July 2026 has 23 working days (Mon-Fri)
        self.mock_db.attendance_logs.aggregate.return_value = [
            {
                "present_days": 20.5,
                "leave_days": 2,
                "late_count": 3,
                "total_late_minutes": 45,
                "total_overtime_minutes": 60,
            }
        ]

        resp = client.get("/analytics/employees/EMP0001/monthly?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["emp_code"], "EMP0001")
        self.assertEqual(data["month"], "2026-07")
        self.assertEqual(data["working_days"], 23)
        self.assertEqual(data["present_days"], 20.5)
        self.assertEqual(data["leave_days"], 2)
        self.assertEqual(data["late_count"], 3)
        self.assertEqual(data["total_late_minutes"], 45)
        self.assertEqual(data["total_overtime_minutes"], 60)
        # 20.5 / 23 * 100 = 89.1304... -> 89.13
        self.assertEqual(data["attendance_pct"], 89.13)

    def test_employee_monthly_mid_month_joiner(self):
        # Employee joins on July 20, 2026 (a Monday)
        # Total working days in July from July 20: 10 working days
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0002",
            "joined_on": "2026-07-20",
        }
        self.mock_db.attendance_logs.aggregate.return_value = [
            {
                "present_days": 9.0,
                "leave_days": 1,
                "late_count": 0,
                "total_late_minutes": 0,
                "total_overtime_minutes": 0,
            }
        ]
        resp = client.get("/analytics/employees/EMP0002/monthly?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["working_days"], 10)
        self.assertEqual(data["present_days"], 9.0)
        self.assertEqual(data["attendance_pct"], 90.0)

    def test_employee_monthly_zero_working_days(self):
        # Joined next month
        self.mock_db.employees.find_one.return_value = {
            "emp_code": "EMP0003",
            "joined_on": "2026-08-01",
        }
        self.mock_db.attendance_logs.aggregate.return_value = []
        resp = client.get("/analytics/employees/EMP0003/monthly?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["working_days"], 0)
        self.assertIsNone(data["attendance_pct"])

    def test_employee_monthly_not_found(self):
        self.mock_db.employees.find_one.return_value = None
        resp = client.get("/analytics/employees/EMP9999/monthly?month=2026-07")
        self.assertEqual(resp.status_code, 404)

    def test_employee_monthly_invalid_month(self):
        resp = client.get("/analytics/employees/EMP0001/monthly?month=2026-13")
        self.assertEqual(resp.status_code, 422)

    # --- GET /analytics/departments/summary ---
    def test_department_summary_normal(self):
        self.mock_db.employees.aggregate.return_value = [
            {
                "department": "Engineering",
                "headcount": 10,
                "present_days": 185.5,
                "avg_work_hours": 8.754,
                "late_count": 5,
                "total_late_minutes": 120,
                "leave_count": 8,
                "on_duty_count": 2,
            },
            {
                "department": "HR",
                "headcount": 2,
                "present_days": 0.0,
                "avg_work_hours": None,
                "late_count": 0,
                "total_late_minutes": 0,
                "leave_count": 0,
                "on_duty_count": 0,
            },
        ]
        resp = client.get("/analytics/departments/summary?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["month"], "2026-07")
        items = data["items"]
        self.assertEqual(len(items), 2)

        eng = items[0]
        self.assertEqual(eng["department"], "Engineering")
        self.assertEqual(eng["headcount"], 10)
        self.assertEqual(eng["present_days"], 185.5)
        self.assertEqual(eng["avg_work_hours"], 8.75)  # 8.754 rounded to 8.75
        self.assertEqual(eng["late_count"], 5)
        self.assertEqual(eng["total_late_minutes"], 120)
        self.assertEqual(eng["leave_count"], 8)
        self.assertEqual(eng["on_duty_count"], 2)

        hr = items[1]
        self.assertEqual(hr["department"], "HR")
        self.assertEqual(hr["headcount"], 2)
        self.assertEqual(hr["present_days"], 0.0)
        self.assertIsNone(hr["avg_work_hours"])

    def test_department_summary_invalid_month(self):
        resp = client.get("/analytics/departments/summary?month=invalid")
        self.assertEqual(resp.status_code, 422)

    # --- GET /analytics/leaderboard/late ---
    def test_late_leaderboard_competition_ranking_and_ties(self):
        # Two employees tie at rank 2. Limit is 2. Both must be included!
        self.mock_db.attendance_logs.aggregate.return_value = [
            {
                "rank": 1,
                "emp_code": "EMP0001",
                "name": "Alice",
                "department": "Sales",
                "total_late_minutes": 120,
                "late_count": 4,
            },
            {
                "rank": 2,
                "emp_code": "EMP0002",
                "name": "Bob",
                "department": "Sales",
                "total_late_minutes": 90,
                "late_count": 3,
            },
            {
                "rank": 2,
                "emp_code": "EMP0003",
                "name": "Charlie",
                "department": "Sales",
                "total_late_minutes": 90,
                "late_count": 3,
            },
        ]
        resp = client.get("/analytics/leaderboard/late?month=2026-07&limit=2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["month"], "2026-07")
        items = data["items"]
        # Limit was 2, but rank 2 ties, so 3 items returned!
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["rank"], 1)
        self.assertEqual(items[1]["rank"], 2)
        self.assertEqual(items[2]["rank"], 2)
        self.assertEqual(items[1]["emp_code"], "EMP0002")
        self.assertEqual(items[2]["emp_code"], "EMP0003")

    def test_late_leaderboard_validation(self):
        resp = client.get("/analytics/leaderboard/late?month=2026-07&limit=0")
        self.assertEqual(resp.status_code, 422)
        resp2 = client.get("/analytics/leaderboard/late?month=2026-07&limit=51")
        self.assertEqual(resp2.status_code, 422)

    # --- GET /analytics/departments/{department}/trend ---
    def test_department_trend_normal(self):
        self.mock_db.employees.find_one.return_value = {"department": "Engineering"}
        self.mock_db.attendance_logs.aggregate.return_value = [
            {
                "date": "2026-07-01",
                "is_working_day": True,
                "headcount": 5,
                "present_count": 4.5,
                "late_count": 1,
                "attendance_rate": 0.9,
                "moving_avg_raw": 0.9,
            },
            {
                "date": "2026-07-02",
                "is_working_day": True,
                "headcount": 5,
                "present_count": 5.0,
                "late_count": 0,
                "attendance_rate": 1.0,
                "moving_avg_raw": 0.95,
            },
            {
                "date": "2026-07-04",  # Saturday
                "is_working_day": False,
                "headcount": 5,
                "present_count": 0.0,
                "late_count": 0,
                "attendance_rate": None,
                "moving_avg_raw": 0.95,
            },
        ]

        resp = client.get("/analytics/departments/Engineering/trend?from=2026-07-01&to=2026-07-04")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["department"], "Engineering")
        items = data["items"]
        self.assertEqual(len(items), 3)

        self.assertEqual(items[0]["date"], "2026-07-01")
        self.assertTrue(items[0]["is_working_day"])
        self.assertEqual(items[0]["attendance_rate"], 0.9)
        self.assertEqual(items[0]["moving_avg_7d"], 0.9)

        self.assertEqual(items[2]["date"], "2026-07-04")
        self.assertFalse(items[2]["is_working_day"])
        self.assertIsNone(items[2]["attendance_rate"])
        self.assertEqual(items[2]["moving_avg_7d"], 0.95)

    def test_department_trend_errors(self):
        self.mock_db.employees.find_one.return_value = None
        # 404 unknown department
        resp = client.get("/analytics/departments/NonExistent/trend?from=2026-07-01&to=2026-07-05")
        self.assertEqual(resp.status_code, 404)

        # 422 to < from
        self.mock_db.employees.find_one.return_value = {"department": "Engineering"}
        resp = client.get("/analytics/departments/Engineering/trend?from=2026-07-10&to=2026-07-05")
        self.assertEqual(resp.status_code, 422)

        # 422 range > 92 days
        resp = client.get("/analytics/departments/Engineering/trend?from=2026-01-01&to=2026-04-15")
        self.assertEqual(resp.status_code, 422)

    # --- GET /admin/explain/{endpoint} ---
    def test_admin_explain_endpoint(self):
        # 1. explain employee_monthly
        self.mock_db.command.return_value = {
            "stages": [
                {"$cursor": {"queryPlanner": {"winningPlan": {"stage": "IXSCAN"}}}}
            ]
        }
        resp = client.get("/admin/explain/employee_monthly?emp_code=EMP0001&month=2026-07")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["endpoint"], "employee_monthly")
        self.assertEqual(data["collection"], "attendance_logs")
        self.assertIn("stages", data["explain"])

        # 2. explain department_summary
        resp2 = client.get("/admin/explain/department_summary?month=2026-07")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.json()["collection"], "employees")

        # 3. explain attendance_list
        mock_cursor = MagicMock()
        mock_cursor.explain.return_value = {"queryPlanner": {"winningPlan": {"stage": "IXSCAN"}}}
        self.mock_db.attendance_logs.find.return_value.sort.return_value.skip.return_value.limit.return_value = mock_cursor
        resp3 = client.get("/admin/explain/attendance_list?emp_code=EMP0001")
        self.assertEqual(resp3.status_code, 200)
        self.assertEqual(resp3.json()["collection"], "attendance_logs")

        # 4. explain invalid endpoint
        resp4 = client.get("/admin/explain/unknown_endpoint")
        self.assertEqual(resp4.status_code, 422)

        # 5. explain missing required parameters
        resp5 = client.get("/admin/explain/employee_monthly")
        self.assertEqual(resp5.status_code, 422)


# =========================================================================== #
# 3. Integration Tests Against Real/Configured MongoDB
# =========================================================================== #
class TestAnalyticsLiveIntegration(unittest.TestCase):
    """
    Live integration test that connects to MONGO_URI if available.
    If MongoDB is offline/unreachable, tests cleanly skip and report.
    """
    is_live = False

    @classmethod
    def setUpClass(cls):
        try:
            # Check if live MongoDB answers
            app_main.client.admin.command("ping")
            cls.is_live = True
        except Exception:
            cls.is_live = False

    def setUp(self):
        if not self.is_live:
            self.skipTest("Live MongoDB not available on MONGO_URI; skipping live integration tests.")

    def test_live_aggregation_pipeline(self):
        # Insert temporary records into live test DB and execute real aggregation
        test_emp = {
            "emp_code": "EMPTST1",
            "name": "Integration Test",
            "email": "test@example.com",
            "department": "QA",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-07-01",
            "created_at": datetime.now(timezone.utc),
        }
        test_log = {
            "emp_code": "EMPTST1",
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": datetime(2026, 7, 6, 4, 15, tzinfo=timezone.utc),
            "punch_out": datetime(2026, 7, 6, 13, 15, tzinfo=timezone.utc),
            "work_hours": 9.0,
            "late_minutes": 15,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }
        app_main.db.employees.update_one({"emp_code": "EMPTST1"}, {"$set": test_emp}, upsert=True)
        app_main.db.attendance_logs.update_one(
            {"emp_code": "EMPTST1", "date": "2026-07-06"}, {"$set": test_log}, upsert=True
        )

        try:
            resp = client.get("/analytics/employees/EMPTST1/monthly?month=2026-07")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["emp_code"], "EMPTST1")
            self.assertEqual(data["present_days"], 1.0)
            self.assertEqual(data["late_count"], 1)
            self.assertEqual(data["total_late_minutes"], 15)
        finally:
            app_main.db.employees.delete_one({"emp_code": "EMPTST1"})
            app_main.db.attendance_logs.delete_one({"emp_code": "EMPTST1", "date": "2026-07-06"})


if __name__ == "__main__":
    unittest.main()
