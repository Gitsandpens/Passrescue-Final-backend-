"""Run history. PostgreSQL when DATABASE_URL is set, SQLite otherwise. Never blocks planning."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from uuid import uuid4
from fastapi.encoders import jsonable_encoder
from sqlalchemy import create_engine, text

url = os.getenv("DATABASE_URL", "sqlite:///passrescue.db")
for prefix in ("postgres://", "postgresql://"):
    if url.startswith(prefix):
        url = "postgresql+psycopg2://" + url[len(prefix):]
db = create_engine(url, pool_pre_ping=True)
try:
    with db.begin() as c:
        c.execute(text("create table if not exists runs (id varchar(40) primary key, created varchar(40), kind varchar(20), summary text)"))
except Exception as e:
    print("Run history unavailable:", e, flush=True)


def name():
    return {"postgresql": "PostgreSQL", "sqlite": "SQLite"}.get(db.dialect.name, db.dialect.name)


def save_run(result):
    try:
        rec = result["recovery"]
        summary = jsonable_encoder(dict(metrics=result["metrics"], baseline_metrics=result["baseline_metrics"],
                                        solver=result["solver"], recovery=rec,
                                        violations=result["validation"]["violation_count"], runtime_s=result["runtime_s"]))
        with db.begin() as c:
            c.execute(text("insert into runs (id, created, kind, summary) values (:i, :c, :k, :s)"),
                      dict(i=uuid4().hex, c=datetime.now(timezone.utc).isoformat(), k="recovery" if rec["is_recovery"] else "plan",
                           s=json.dumps(summary)))
    except Exception as e:
        print("Run not saved:", e, flush=True)


def recent(limit=20):
    with db.begin() as c:
        rows = c.execute(text("select id, created, kind, summary from runs order by created desc limit :n"), dict(n=limit)).all()
    return [dict(id=r[0], created=r[1], kind=r[2], summary=json.loads(r[3])) for r in rows]
