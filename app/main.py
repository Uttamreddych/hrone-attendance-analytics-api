"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience)
"""
import calendar
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError, DuplicateKeyError

# Load local .env if present. Real environment variables take priority.
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

# tz_aware=True ensures datetimes from MongoDB are timezone-aware UTC.
# serverSelectionTimeoutMS=2000 ensures health check fails quickly if DB is unreachable.
client = MongoClient(MONGO_URI, tz_aware=True, serverSelectionTimeoutMS=2000)
db = client[MONGO_DB]

IST = timezone(timedelta(hours=5, minutes=30))


# --------------------------------------------------------------------------- #
# Database Startup & Indexes
# --------------------------------------------------------------------------- #
def ensure_indexes():
    """Create MongoDB indexes idempotently at startup."""
    try:
        # employees collection
        db.employees.create_index([("emp_code", 1)], unique=True)
        db.employees.create_index([("department", 1), ("joined_on", 1)])
        db.employees.create_index([("joined_on", 1), ("department", 1)])

        # attendance_logs collection
        db.attendance_logs.create_index([("emp_code", 1), ("date", 1)], unique=True)
        db.attendance_logs.create_index([("date", -1), ("emp_code", 1)])
        db.attendance_logs.create_index([("emp_code", 1), ("punch_in", -1)])
        db.attendance_logs.create_index([("date", 1), ("emp_code", 1)])
        db.attendance_logs.create_index([("date", 1), ("late_minutes", 1), ("emp_code", 1)])
    except PyMongoError:
        # If MongoDB is unreachable at startup, allow app to boot so /health can report 503
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure indexes exist
    ensure_indexes()
    yield
    # Shutdown: cleanly close client connection
    client.close()


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers: Timezone, Epoch Milliseconds, and Calculations
# --------------------------------------------------------------------------- #
def round_half_up(val: Decimal, decimals: int = 2) -> float:
    """
    Rounds a Decimal value using standard HALF-UP rounding (Rule R4, R8).
    Avoids Python's built-in float round-half-to-even behavior.
    """
    fmt = "0." + "0" * decimals if decimals > 0 else "0"
    return float(val.quantize(Decimal(fmt), rounding=ROUND_HALF_UP))


def truncate_to_seconds(dt: datetime) -> datetime:
    """
    Truncates an instant to whole seconds before calculations (Rule R1).
    For example: 09:40:00.900 counts as 09:40:00.
    """
    return dt.replace(microsecond=0)


def epoch_ms_to_datetime(ms: int) -> datetime:
    """
    Converts epoch milliseconds to a UTC datetime truncated to whole seconds (Rule R1).
    """
    seconds = ms // 1000  # Truncate fractional seconds
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def datetime_to_epoch_ms(dt: Optional[datetime]) -> Optional[int]:
    """
    Converts a datetime instant to epoch milliseconds. Returns None if dt is None.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1000


def is_overnight_shift(shift_start: str, shift_end: str) -> bool:
    """
    Returns True if shift_end <= shift_start (e.g. 22:00 to 06:00, Rule R1).
    """
    start_h, start_m = map(int, shift_start.split(":"))
    end_h, end_m = map(int, shift_end.split(":"))
    return (end_h, end_m) <= (start_h, start_m)


def get_attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """
    Determines the IST attendance date (YYYY-MM-DD) for a punch-in (Rule R1).
    Overnight shifts: If punch_in IST time is earlier than shift_end,
    it belongs to the shift that started the previous calendar day.
    """
    punch_in_ist = truncate_to_seconds(punch_in.astimezone(IST))
    cal_date = punch_in_ist.date()

    if is_overnight_shift(shift_start, shift_end):
        end_h, end_m = map(int, shift_end.split(":"))
        if (punch_in_ist.hour, punch_in_ist.minute, punch_in_ist.second) < (end_h, end_m, 0):
            cal_date = cal_date - timedelta(days=1)

    return cal_date.isoformat()


def compute_late_minutes(
    punch_in: datetime, shift_start: str, attendance_date: Optional[str] = None
) -> int:
    """
    Computes late minutes with a 10-minute grace period (Rule R2).
    - Late only if strictly more than 10 minutes (> 600s) after shift start.
    - When late, late_minutes = whole minutes (floor) since shift start.
    - Otherwise 0.
    """
    punch_in_ist = truncate_to_seconds(punch_in.astimezone(IST))
    if attendance_date is None:
        attendance_date = punch_in_ist.date().isoformat()

    start_h, start_m = map(int, shift_start.split(":"))
    shift_start_dt = datetime.fromisoformat(attendance_date).replace(
        hour=start_h, minute=start_m, second=0, microsecond=0, tzinfo=IST
    )

    elapsed_seconds = int((punch_in_ist - shift_start_dt).total_seconds())
    # Strict 10-minute grace period (600 seconds)
    if elapsed_seconds > 600:
        return elapsed_seconds // 60
    return 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """
    Computes work hours (Rule R4):
    (punch_out - punch_in) in seconds / 3600, rounded to 2 decimals HALF-UP.
    Instants are truncated to whole seconds before calculation.
    """
    p_in = truncate_to_seconds(punch_in)
    p_out = truncate_to_seconds(punch_out)
    diff_seconds = Decimal(str(int((p_out - p_in).total_seconds())))
    hours = diff_seconds / Decimal("3600")
    return round_half_up(hours, 2)


def compute_overtime(
    punch_out: datetime, shift_end: str, attendance_date: str, shift_start: str = "09:30"
) -> int:
    """
    Computes overtime minutes (Rule R3):
    Whole minutes (floor) between shift_end and punch_out, counted only if >= 30; otherwise 0.
    For overnight shifts, shift_end falls on the day after attendance_date.
    """
    punch_out_ist = truncate_to_seconds(punch_out.astimezone(IST))
    end_date = datetime.fromisoformat(attendance_date).date()
    if is_overnight_shift(shift_start, shift_end):
        end_date = end_date + timedelta(days=1)

    end_h, end_m = map(int, shift_end.split(":"))
    shift_end_dt = datetime(
        end_date.year, end_date.month, end_date.day, end_h, end_m, 0, tzinfo=IST
    )

    diff_seconds = int((punch_out_ist - shift_end_dt).total_seconds())
    minutes = diff_seconds // 60
    if minutes >= 30:
        return int(minutes)
    return 0


def compute_half_day(work_hours: Optional[float]) -> bool:
    """
    Computes half_day boolean (Rule R5):
    True when rounded work_hours < 4.50.
    """
    if work_hours is None:
        return False
    return work_hours < 4.50


def round_rate_half_up(val: Decimal) -> float:
    """Rounds a rate value to 4 decimal places HALF-UP (Rule R8)."""
    return round_half_up(val, 4)


def calculate_working_days(joined_on_str: str, year: int, month: int) -> int:
    """
    Counts Monday-Friday working days in the month from max(start_of_month, joined_on)
    to end_of_month (Rule R7). Returns 0 if employee joined after the end of the month.
    """
    last_day = calendar.monthrange(year, month)[1]
    month_start = date(year, month, 1)
    month_end = date(year, month, last_day)

    joined_on = datetime.strptime(joined_on_str, "%Y-%m-%d").date()
    if joined_on > month_end:
        return 0

    effective_start = max(month_start, joined_on)
    days_count = (month_end - effective_start).days + 1
    return sum(
        1 for d in range(days_count) if (effective_start + timedelta(days=d)).weekday() < 5
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class StatusEnum(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    LEAVE = "LEAVE"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class PresenceStatusEnum(str, Enum):
    PRESENT = "PRESENT"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class EmployeeIn(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field("18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def validate_joined_on(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("joined_on must be a valid date in YYYY-MM-DD format")
        return v

    @model_validator(mode="after")
    def validate_shift_times(self) -> "EmployeeIn":
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(None, ge=100000000000, le=4102444800000, strict=True)
    status: PresenceStatusEnum = PresenceStatusEnum.PRESENT


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(None, ge=100000000000, le=4102444800000, strict=True)


class RegularizeIn(BaseModel):
    status: Optional[StatusEnum] = None
    punch_in: Optional[int] = Field(None, ge=100000000000, le=4102444800000, strict=True)
    punch_out: Optional[int] = Field(None, ge=100000000000, le=4102444800000, strict=True)
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# Serialization Helpers
# --------------------------------------------------------------------------- #
def serialize_employee(doc: dict) -> dict:
    """Formats an employee document according to the OpenAPI Employee schema."""
    created_at_val = doc.get("created_at")
    if isinstance(created_at_val, datetime):
        created_at_ms = datetime_to_epoch_ms(created_at_val)
    elif isinstance(created_at_val, int):
        created_at_ms = created_at_val
    else:
        created_at_ms = None

    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": created_at_ms,
    }


def serialize_history_entry(entry: dict) -> dict:
    """Formats a single audit history entry."""
    at_val = entry.get("at")
    if isinstance(at_val, datetime):
        at_ms = datetime_to_epoch_ms(at_val)
    elif isinstance(at_val, int):
        at_ms = at_val
    else:
        at_ms = None

    changes = {}
    for field, change in entry.get("changes", {}).items():
        c_from = change.get("from")
        c_to = change.get("to")
        if field in ("punch_in", "punch_out"):
            if isinstance(c_from, datetime):
                c_from = datetime_to_epoch_ms(c_from)
            if isinstance(c_to, datetime):
                c_to = datetime_to_epoch_ms(c_to)
        changes[field] = {"from": c_from, "to": c_to}

    return {
        "at": at_ms,
        "by": entry.get("by"),
        "reason": entry.get("reason"),
        "changes": changes,
    }


def serialize_attendance_record(doc: dict) -> dict:
    """Formats an attendance document according to the OpenAPI AttendanceRecord schema."""
    raw_history = doc.get("history")
    if raw_history is None:
        history = []
    else:
        history = [serialize_history_entry(h) for h in raw_history]

    punch_in_val = doc.get("punch_in")
    punch_in_ms = datetime_to_epoch_ms(punch_in_val) if isinstance(punch_in_val, datetime) else punch_in_val

    punch_out_val = doc.get("punch_out")
    punch_out_ms = datetime_to_epoch_ms(punch_out_val) if isinstance(punch_out_val, datetime) else punch_out_val

    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": punch_in_ms,
        "punch_out": punch_out_ms,
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Endpoints provided & fixed
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    now_utc = truncate_to_seconds(datetime.now(timezone.utc))
    doc = body.model_dump()
    doc["created_at"] = now_utc

    if db.employees.find_one({"emp_code": body.emp_code}):
        raise HTTPException(status_code=409, detail="emp_code already exists")

    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists")

    return serialize_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {}
    if department:
        q["department"] = department

    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    cursor = (
        db.employees.find(q, {"_id": 0})
        .sort("emp_code", 1)
        .skip(skip)
        .limit(page_size)
    )

    items = [serialize_employee(d) for d in cursor]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if body.punched_at is not None:
        punch_in_dt = epoch_ms_to_datetime(body.punched_at)
    else:
        punch_in_dt = truncate_to_seconds(datetime.now(timezone.utc))

    attendance_date = get_attendance_date(
        punch_in_dt, emp["shift_start"], emp["shift_end"]
    )
    late_minutes = compute_late_minutes(
        punch_in_dt, emp["shift_start"], attendance_date
    )

    status_str = (
        body.status.value if isinstance(body.status, Enum) else body.status
    )
    doc = {
        "emp_code": body.emp_code,
        "date": attendance_date,
        "status": status_str,
        "punch_in": punch_in_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_minutes,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    if db.attendance_logs.find_one({"emp_code": body.emp_code, "date": attendance_date}):
        raise HTTPException(status_code=409, detail="already punched in for this date")

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="already punched in for this date")

    return serialize_attendance_record(doc)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[StatusEnum] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from:
        try:
            datetime.strptime(date_from, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=422, detail="date_from must be a valid date in YYYY-MM-DD format"
            )
    if date_to:
        try:
            datetime.strptime(date_to, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=422, detail="date_to must be a valid date in YYYY-MM-DD format"
            )
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=422, detail="date_from must not be greater than date_to"
        )

    q = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status.value if isinstance(status, Enum) else status

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    cursor = (
        db.attendance_logs.find(q, {"_id": 0})
        .sort([("date", -1), ("emp_code", 1)])
        .skip(skip)
        .limit(page_size)
    )

    items = [serialize_attendance_record(d) for d in cursor]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# --------------------------------------------------------------------------- #
# Punch-Out and Regularization Endpoints
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-out", status_code=200)
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if body.punched_at is not None:
        punch_out_dt = epoch_ms_to_datetime(body.punched_at)
    else:
        punch_out_dt = truncate_to_seconds(datetime.now(timezone.utc))

    # Find the employee's most recent record whose punch_in is at or before punched_at
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": punch_out_dt}},
        sort=[("punch_in", -1)],
    )
    if not record or record.get("punch_in") is None:
        raise HTTPException(status_code=404, detail="No matching punch-in record found")

    # If that record is already punched out -> 409
    if record.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="Record is already punched out")

    # Time difference validation: punched_at <= punch_in or > 24 hours -> 422
    punch_in_dt = record["punch_in"]
    if punch_in_dt.tzinfo is None:
        punch_in_dt = punch_in_dt.replace(tzinfo=timezone.utc)

    diff_seconds = int((punch_out_dt - punch_in_dt).total_seconds())
    if diff_seconds <= 0:
        raise HTTPException(status_code=422, detail="punched_at must be strictly after punch_in")
    if diff_seconds > 86400:
        raise HTTPException(status_code=422, detail="punched_at cannot be more than 24 hours after punch_in")

    # Compute derived fields (R3 - R5)
    work_hours = compute_work_hours(punch_in_dt, punch_out_dt)
    overtime_minutes = compute_overtime(
        punch_out_dt, emp["shift_end"], record["date"], emp["shift_start"]
    )
    half_day = compute_half_day(work_hours)

    # Concurrency safe atomic update: two simultaneous punch-outs yield exactly one 200, rest 409
    updated_record = db.attendance_logs.find_one_and_update(
        {
            "emp_code": body.emp_code,
            "date": record["date"],
            "punch_out": None,
        },
        {
            "$set": {
                "punch_out": punch_out_dt,
                "work_hours": work_hours,
                "overtime_minutes": overtime_minutes,
                "half_day": half_day,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not updated_record:
        raise HTTPException(status_code=409, detail="Record is already punched out")

    return serialize_attendance_record(updated_record)


@app.patch("/attendance/{emp_code}/{date}", status_code=200)
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="date parameter must be in YYYY-MM-DD format")

    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not record:
        raise HTTPException(status_code=404, detail="Attendance record not found for this date")

    current_status = record.get("status")
    current_punch_in = record.get("punch_in")
    if current_punch_in and current_punch_in.tzinfo is None:
        current_punch_in = current_punch_in.replace(tzinfo=timezone.utc)

    current_punch_out = record.get("punch_out")
    if current_punch_out and current_punch_out.tzinfo is None:
        current_punch_out = current_punch_out.replace(tzinfo=timezone.utc)

    # Status determination
    if body.status is not None:
        target_status = body.status.value if isinstance(body.status, Enum) else body.status
    else:
        target_status = current_status

    # Rule: setting status ABSENT or LEAVE clears both punch times;
    # supplying punch_in or punch_out with it is 422
    if target_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(
                status_code=422,
                detail="supplying punch_in or punch_out when setting status to ABSENT or LEAVE is not allowed",
            )
        target_punch_in = None
        target_punch_out = None
    else:
        # Presence status requires a punch_in (supplied or already stored)
        if body.punch_in is not None:
            target_punch_in = epoch_ms_to_datetime(body.punch_in)
        else:
            target_punch_in = current_punch_in

        if target_punch_in is None:
            raise HTTPException(status_code=422, detail="a presence status requires a punch_in")

        if body.punch_out is not None:
            target_punch_out = epoch_ms_to_datetime(body.punch_out)
        else:
            target_punch_out = current_punch_out

        # punch_in must stay on the record's attendance date (R1)
        computed_date = get_attendance_date(target_punch_in, emp["shift_start"], emp["shift_end"])
        if computed_date != date:
            raise HTTPException(status_code=422, detail="punch_in must stay on the record's attendance date")

        # punch_out must be after punch_in and within 24 hours
        if target_punch_out is not None:
            diff_sec = (target_punch_out - target_punch_in).total_seconds()
            if diff_sec <= 0:
                raise HTTPException(status_code=422, detail="punch_out must be strictly after punch_in")
            if diff_sec > 86400:
                raise HTTPException(status_code=422, detail="punch_out must be within 24 hours of punch_in")

    # Recompute derived fields (R2 - R5)
    if target_status in ("ABSENT", "LEAVE"):
        target_work_hours = None
        target_late_minutes = 0
        target_overtime_minutes = 0
        target_half_day = False
    else:
        target_late_minutes = compute_late_minutes(target_punch_in, emp["shift_start"], date)
        if target_punch_out is not None:
            target_work_hours = compute_work_hours(target_punch_in, target_punch_out)
            target_overtime_minutes = compute_overtime(
                target_punch_out, emp["shift_end"], date, emp["shift_start"]
            )
            target_half_day = compute_half_day(target_work_hours)
        else:
            target_work_hours = None
            target_overtime_minutes = 0
            target_half_day = False

    # Compute changes: only fields that actually changed
    changes_db = {}
    if target_status != current_status:
        changes_db["status"] = {"from": current_status, "to": target_status}
    if target_punch_in != current_punch_in:
        changes_db["punch_in"] = {"from": current_punch_in, "to": target_punch_in}
    if target_punch_out != current_punch_out:
        changes_db["punch_out"] = {"from": current_punch_out, "to": target_punch_out}
    if target_work_hours != record.get("work_hours"):
        changes_db["work_hours"] = {"from": record.get("work_hours"), "to": target_work_hours}
    if target_late_minutes != record.get("late_minutes", 0):
        changes_db["late_minutes"] = {"from": record.get("late_minutes", 0), "to": target_late_minutes}
    if target_overtime_minutes != record.get("overtime_minutes", 0):
        changes_db["overtime_minutes"] = {
            "from": record.get("overtime_minutes", 0),
            "to": target_overtime_minutes,
        }
    if target_half_day != record.get("half_day", False):
        changes_db["half_day"] = {"from": record.get("half_day", False), "to": target_half_day}

    # Contract: a request that changes nothing is 422
    if not changes_db:
        raise HTTPException(status_code=422, detail="Request changes nothing")

    # History entry creation (at is BSON datetime truncated to seconds)
    now_utc = truncate_to_seconds(datetime.now(timezone.utc))
    history_entry = {
        "at": now_utc,
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes_db,
    }

    # Optimistic lock on history array to prevent concurrent edits from overwriting each other
    existing_history = record.get("history", [])
    update_filter = {
        "emp_code": emp_code,
        "date": date,
        "history": existing_history,
    }
    update_doc = {
        "$set": {
            "status": target_status,
            "punch_in": target_punch_in,
            "punch_out": target_punch_out,
            "work_hours": target_work_hours,
            "late_minutes": target_late_minutes,
            "overtime_minutes": target_overtime_minutes,
            "half_day": target_half_day,
        },
        "$push": {
            "history": history_entry,
        },
    }

    updated_record = db.attendance_logs.find_one_and_update(
        update_filter,
        update_doc,
        return_document=ReturnDocument.AFTER,
    )
    if not updated_record:
        raise HTTPException(
            status_code=409,
            detail="Concurrent modification conflict: record was modified concurrently",
        )

    return serialize_attendance_record(updated_record)


# --------------------------------------------------------------------------- #
# Analytics Pipeline Builders
# --------------------------------------------------------------------------- #
def build_employee_monthly_pipeline(emp_code: str, month: str) -> list:
    """Builds MongoDB aggregation pipeline for monthly attendance summary of an employee."""
    year, month_num = map(int, month.split("-"))
    last_day = calendar.monthrange(year, month_num)[1]
    start_date_str = f"{month}-01"
    end_date_str = f"{month}-{last_day:02d}"

    return [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": start_date_str, "$lte": end_date_str},
            }
        },
        {
            "$project": {
                "date": 1,
                "status": 1,
                "late_minutes": 1,
                "overtime_minutes": 1,
                "half_day": 1,
                "is_weekday": {
                    "$in": [
                        {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                        [2, 3, 4, 5, 6],  # Mon=2 to Fri=6
                    ]
                },
                "is_present_status": {
                    "$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]
                },
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {"$and": ["$is_present_status", "$is_weekday"]},
                            {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
    ]


def build_department_summary_pipeline(
    month: str, department: Optional[str] = None
) -> list:
    """Builds cross-collection MongoDB aggregation pipeline for department monthly summary."""
    year, month_num = map(int, month.split("-"))
    last_day = calendar.monthrange(year, month_num)[1]
    start_date_str = f"{month}-01"
    end_date_str = f"{month}-{last_day:02d}"

    match_stage = {"joined_on": {"$lte": end_date_str}}
    if department:
        match_stage["department"] = department

    return [
        {"$match": match_stage},
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "emp_codes": {"$push": "$emp_code"},
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"emp_codes": "$emp_codes"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$in": ["$emp_code", "$$emp_codes"]},
                                    {"$gte": ["$date", start_date_str]},
                                    {"$lte": ["$date", end_date_str]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "status": 1,
                            "work_hours": 1,
                            "late_minutes": 1,
                            "half_day": 1,
                            "is_weekday": {
                                "$in": [
                                    {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                                    [2, 3, 4, 5, 6],
                                ]
                            },
                            "is_present_status": {
                                "$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]
                            },
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {
            "$project": {
                "department": "$_id",
                "headcount": 1,
                "present_days": {
                    "$sum": {
                        "$map": {
                            "input": "$logs",
                            "as": "l",
                            "in": {
                                "$cond": [
                                    {"$and": ["$$l.is_present_status", "$$l.is_weekday"]},
                                    {"$cond": [{"$ifNull": ["$$l.half_day", False]}, 0.5, 1.0]},
                                    0.0,
                                ]
                            },
                        }
                    }
                },
                "work_hours_list": {
                    "$filter": {
                        "input": "$logs",
                        "as": "l",
                        "cond": {
                            "$and": [
                                "$$l.is_present_status",
                                {"$ne": ["$$l.work_hours", None]},
                            ]
                        },
                    }
                },
                "late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$gt": [{"$ifNull": ["$$l.late_minutes", 0]}, 0]},
                        }
                    }
                },
                "total_late_minutes": {
                    "$sum": {
                        "$map": {
                            "input": "$logs",
                            "as": "l",
                            "in": {"$ifNull": ["$$l.late_minutes", 0]},
                        }
                    }
                },
                "leave_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "LEAVE"]},
                        }
                    }
                },
                "on_duty_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "ON_DUTY"]},
                        }
                    }
                },
            }
        },
        {
            "$project": {
                "department": 1,
                "headcount": 1,
                "present_days": 1,
                "avg_work_hours": {"$avg": "$work_hours_list.work_hours"},
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$sort": {"department": 1}},
    ]


def build_late_leaderboard_pipeline(
    month: str, limit: int = 10, department: Optional[str] = None
) -> list:
    """Builds MongoDB aggregation pipeline for late leaderboard using standard competition ranking."""
    year, month_num = map(int, month.split("-"))
    last_day = calendar.monthrange(year, month_num)[1]
    start_date_str = f"{month}-01"
    end_date_str = f"{month}-{last_day:02d}"

    pipeline = [
        {
            "$match": {
                "date": {"$gte": start_date_str, "$lte": end_date_str},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]

    if department:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline.extend([
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {
                    "rank": {"$rank": {}}
                },
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ])
    return pipeline


def build_department_trend_pipeline(
    department: str, from_date: str, to_date: str
) -> list:
    """Builds MongoDB aggregation pipeline for daily department attendance trend with gap filling & 7d moving avg."""
    from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_next_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    return [
        {"$match": {"date": {"$gte": from_date, "$lte": to_date}}},
        {
            "$lookup": {
                "from": "employees",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
        {"$match": {"emp.department": department}},
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                            {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "date_str": "$_id",
                "day_date": {"$dateFromString": {"dateString": "$_id"}},
                "present_count": 1,
                "late_count": 1,
            }
        },
        # Ensure at least one seed document exists so $densify covers [from_dt, to_next_dt]
        {
            "$unionWith": {
                "coll": "employees",
                "pipeline": [
                    {"$match": {"department": department}},
                    {"$limit": 1},
                    {
                        "$project": {
                            "date_str": {"$literal": from_date},
                            "day_date": {"$literal": from_dt},
                            "present_count": {"$literal": 0.0},
                            "late_count": {"$literal": 0},
                        }
                    },
                ],
            }
        },
        {
            "$group": {
                "_id": "$day_date",
                "present_count": {"$sum": "$present_count"},
                "late_count": {"$sum": "$late_count"},
            }
        },
        {
            "$project": {
                "day_date": "$_id",
                "present_count": 1,
                "late_count": 1,
            }
        },
        {
            "$densify": {
                "field": "day_date",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [from_dt, to_next_dt],
                },
            }
        },
        {
            "$project": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$day_date"}},
                "present_count": {"$ifNull": ["$present_count", 0.0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
                "day_of_week": {"$dayOfWeek": "$day_date"},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"cur_date": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$cur_date"]},
                                ]
                            }
                        }
                    },
                    {"$count": "count"},
                ],
                "as": "hc",
            }
        },
        {
            "$project": {
                "date": 1,
                "is_working_day": {"$in": ["$day_of_week", [2, 3, 4, 5, 6]]},
                "headcount": {"$ifNull": [{"$arrayElemAt": ["$hc.count", 0]}, 0]},
                "present_count": 1,
                "late_count": 1,
            }
        },
        {
            "$project": {
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": {
                    "$cond": [
                        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                        {"$divide": ["$present_count", "$headcount"]},
                        None,
                    ]
                },
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "moving_avg_raw": {
                        "$avg": "$attendance_rate",
                        "window": {
                            "documents": [-6, "current"],
                        },
                    }
                },
            }
        },
        {"$sort": {"date": 1}},
    ]


def clean_bson_for_json(val: Any) -> Any:
    """Recursively formats BSON values (e.g. from explain plans) into clean JSON-serializable types."""
    if isinstance(val, dict):
        return {k: clean_bson_for_json(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [clean_bson_for_json(v) for v in val]
    elif isinstance(val, (datetime, date)):
        return val.isoformat()
    elif hasattr(val, "__str__") and type(val).__name__ in ("ObjectId", "Decimal128", "Int64"):
        if type(val).__name__ == "Int64":
            return int(val)
        elif type(val).__name__ == "Decimal128":
            return float(str(val))
        return str(val)
    return val


# --------------------------------------------------------------------------- #
# Analytics Endpoints
# --------------------------------------------------------------------------- #
@app.get("/analytics/employees/{emp_code}/monthly", status_code=200)
def employee_monthly(
    emp_code: str,
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    year, month_num = map(int, month.split("-"))
    working_days = calculate_working_days(emp["joined_on"], year, month_num)
    pipeline = build_employee_monthly_pipeline(emp_code, month)

    agg_res = list(db.attendance_logs.aggregate(pipeline))
    if agg_res:
        row = agg_res[0]
        present_days = float(row.get("present_days", 0.0))
        leave_days = int(row.get("leave_days", 0))
        late_count = int(row.get("late_count", 0))
        total_late_minutes = int(row.get("total_late_minutes", 0))
        total_overtime_minutes = int(row.get("total_overtime_minutes", 0))
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late_minutes = 0
        total_overtime_minutes = 0

    if working_days == 0:
        attendance_pct = None
    else:
        pct = (Decimal(str(present_days)) / Decimal(str(working_days))) * Decimal("100")
        attendance_pct = round_half_up(pct, 2)

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late_minutes,
        "total_overtime_minutes": total_overtime_minutes,
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary", status_code=200)
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
):
    pipeline = build_department_summary_pipeline(month, department)
    raw_items = list(db.employees.aggregate(pipeline))
    items = []
    for item in raw_items:
        avg_wh = item.get("avg_work_hours")
        if avg_wh is not None:
            avg_wh = round_half_up(Decimal(str(avg_wh)), 2)
        items.append({
            "department": item["department"],
            "headcount": int(item["headcount"]),
            "present_days": float(item["present_days"]),
            "avg_work_hours": avg_wh,
            "late_count": int(item["late_count"]),
            "total_late_minutes": int(item.get("total_late_minutes") or 0),
            "leave_count": int(item["leave_count"]),
            "on_duty_count": int(item["on_duty_count"]),
        })

    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late", status_code=200)
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = Query(None),
):
    pipeline = build_late_leaderboard_pipeline(month, limit, department)
    raw_items = list(db.attendance_logs.aggregate(pipeline))
    items = []
    for item in raw_items:
        items.append({
            "rank": int(item["rank"]),
            "emp_code": item["emp_code"],
            "name": item["name"],
            "department": item["department"],
            "total_late_minutes": int(item["total_late_minutes"]),
            "late_count": int(item["late_count"]),
        })

    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend", status_code=200)
def department_trend(
    department: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
):
    try:
        d_from = datetime.strptime(from_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="from parameter must be in YYYY-MM-DD format")

    try:
        d_to = datetime.strptime(to_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="to parameter must be in YYYY-MM-DD format")

    if d_to < d_from:
        raise HTTPException(status_code=422, detail="to date must not be earlier than from date")

    if (d_to - d_from).days + 1 > 92:
        raise HTTPException(status_code=422, detail="date range must not exceed 92 days")

    # Check if department exists in employees
    if not db.employees.find_one({"department": department}):
        raise HTTPException(status_code=404, detail="Department not found")

    pipeline = build_department_trend_pipeline(department, from_date, to_date)
    raw_items = list(db.attendance_logs.aggregate(pipeline))
    items = []
    for item in raw_items:
        rate = item.get("attendance_rate")
        if rate is not None:
            rate = round_rate_half_up(Decimal(str(rate)))

        m_avg = item.get("moving_avg_raw")
        if m_avg is not None:
            m_avg = round_rate_half_up(Decimal(str(m_avg)))

        items.append({
            "date": item["date"],
            "is_working_day": bool(item["is_working_day"]),
            "headcount": int(item["headcount"]),
            "present_count": float(item["present_count"]),
            "late_count": int(item["late_count"]),
            "attendance_rate": rate,
            "moving_avg_7d": m_avg,
        })

    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Admin / Explain Endpoint
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}", status_code=200)
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[StatusEnum] = None,
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    valid_endpoints = [
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    ]
    if endpoint not in valid_endpoints:
        raise HTTPException(status_code=422, detail="Invalid endpoint name")

    if endpoint == "attendance_list":
        if date_from:
            try:
                datetime.strptime(date_from, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=422, detail="date_from must be a valid date in YYYY-MM-DD format"
                )
        if date_to:
            try:
                datetime.strptime(date_to, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=422, detail="date_to must be a valid date in YYYY-MM-DD format"
                )
        if date_from and date_to and date_from > date_to:
            raise HTTPException(
                status_code=422, detail="date_from must not be greater than date_to"
            )

        q = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status.value if isinstance(status, Enum) else status

        skip = (page - 1) * page_size
        cursor = (
            db.attendance_logs.find(q)
            .sort([("date", -1), ("emp_code", 1)])
            .skip(skip)
            .limit(page_size)
        )
        explain_doc = cursor.explain()
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": clean_bson_for_json(explain_doc),
        }

    elif endpoint == "employee_monthly":
        if not emp_code:
            raise HTTPException(status_code=422, detail="emp_code is required for employee_monthly explain")
        if not month:
            raise HTTPException(status_code=422, detail="month is required for employee_monthly explain")
        pipeline = build_employee_monthly_pipeline(emp_code, month)
        explain_doc = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": clean_bson_for_json(explain_doc),
        }

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(status_code=422, detail="month is required for department_summary explain")
        pipeline = build_department_summary_pipeline(month, department)
        explain_doc = db.command(
            "explain",
            {
                "aggregate": "employees",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "employees",
            "explain": clean_bson_for_json(explain_doc),
        }

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(status_code=422, detail="month is required for late_leaderboard explain")
        pipeline = build_late_leaderboard_pipeline(month, limit, department)
        explain_doc = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": clean_bson_for_json(explain_doc),
        }

    elif endpoint == "department_trend":
        if not department:
            raise HTTPException(status_code=422, detail="department is required for department_trend explain")
        if not from_date or not to_date:
            raise HTTPException(status_code=422, detail="from and to are required for department_trend explain")
        pipeline = build_department_trend_pipeline(department, from_date, to_date)
        explain_doc = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": clean_bson_for_json(explain_doc),
        }

