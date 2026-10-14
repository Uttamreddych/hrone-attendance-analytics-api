# Employee Attendance & Analytics API

A REST API built with Python, FastAPI, and MongoDB for employee attendance tracking and analytics.

## Features

- Create and list employees with pagination and filtering.
- Record punch-in and punch-out with attendance business rules.
- Correct attendance records with a reason and audit history.
- Generate employee, department, late-arrival, and attendance-trend analytics.
- Use MongoDB aggregation pipelines for analytics.
- Create database indexes to support uniqueness and query performance.
- Provide a health endpoint and an endpoint for explaining query plans.

## Technology Stack

- Python 3.11+
- FastAPI
- PyMongo
- MongoDB
- Pydantic
- unittest

## Requirements

- Python 3.11 or newer
- MongoDB running locally or a reachable MongoDB instance

## Setup

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create your local environment configuration:

```powershell
Copy-Item .env.example .env
```

Configure `MONGO_URI` and `MONGO_DB` in `.env` if needed.

Start the API:

```powershell
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` to explore the API.

## Running Tests

Run the test suite with:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

In my local test run, 70 tests passed and 1 test was skipped. The skipped test requires a running MongoDB instance. Live database integration and production performance have not been verified in this environment.

## Configuration

| Variable | Default |
|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` |
| `MONGO_DB` | `attendance_db` |

Do not commit `.env` files, credentials, or other secrets.

## Project Documentation

- `REVIEW.md` — review of identified implementation issues.
- `DECISIONS.md` — technical decisions and trade-offs.
- `tests/` — automated tests.

## Limitations

Live MongoDB integration tests require an available MongoDB server. Results from a local test run do not establish production readiness or performance at scale.
