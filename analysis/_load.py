"""일별 시간대 입출차 자료를 읽습니다(공공데이터포털 15138735, 2024년 7월 31일치)."""
import os, json, datetime as dt
import numpy as np
from _style import DATA, ROOT

def load_days():
    rows = []
    for line in open(os.path.join(DATA, "daily_hourly.txt"), encoding="utf-8"):
        d, i, o, w = line.strip().split("|")
        rows.append({"date": d, "in": np.array(i.split(), int), "out": np.array(o.split(), int),
                     "dwell": np.array(w.split(), int)})
    concerts = json.load(open(os.path.join(ROOT, "data", "aggregates.json"), encoding="utf-8"))["concerts_2024_07"]["dates"]
    for r in rows:
        day = dt.date.fromisoformat(r["date"])
        r["weekend"] = int(day.weekday() >= 5)
        r["concert"] = int(r["date"] in concerts)
        r["n_concert"] = concerts.get(r["date"], 0)
        r["dow"] = "월화수목금토일"[day.weekday()]
    assert len(rows) == 31
    return rows
