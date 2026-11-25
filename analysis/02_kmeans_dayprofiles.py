"""
02. 일별 출차 시간 패턴 군집 분석 (k-means)

목적
- 31일의 24시간 출차 분포를 비율로 바꾼 뒤 k-means로 묶어, 공원의 하루가 몇 가지 유형으로 운영되는지 확인합니다.
- 공연이 있던 날(7/1, 7/12, 7/20)이 별도 유형으로 묶이는지, 묶이지 않는다면 어느 시간대에서 차이가 나는지 확인합니다.
방법
- 각 날의 시간대별 출차 수를 그날 전체 출차 수로 나눈 비율(합계 1)을 입력으로 씁니다.
- k는 2~5 중 실루엣 계수가 가장 높은 값으로 고릅니다. 초기값 영향을 줄이려고 n_init=50, random_state=0으로 고정합니다.
"""
import json, os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from _style import *
from _load import load_days

days = load_days()
X = np.array([d["out"] / d["out"].sum() for d in days])

scores = {}
models = {}
for k in range(2, 6):
    km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
    scores[k] = float(silhouette_score(X, km.labels_))
    models[k] = km
best_k = max(scores, key=scores.get)
km = models[best_k]
lab = km.labels_

# 군집 이름: 주말 비율로 정렬
order = sorted(range(best_k), key=lambda c: np.mean([d["weekend"] for d, l in zip(days, lab) if l == c]))
name = {}
for rank, c in enumerate(order):
    members = [d for d, l in zip(days, lab) if l == c]
    wk = sum(d["weekend"] for d in members)
    name[c] = f"유형 {chr(65+rank)} ({len(members)}일, 주말 {wk}일)"

# 공연일과 같은 군집 평일의 차이: 21시 이후 출차 비율
late_share = X[:, 21:].sum(axis=1)
summary = []
for d, l, ls in zip(days, lab, late_share):
    summary.append({"date": d["date"], "dow": d["dow"], "cluster": name[l], "weekend": d["weekend"],
                    "concert": d["concert"], "late21_share": round(float(ls), 4),
                    "dist_to_centroid": round(float(np.linalg.norm(X[days.index(d)] - km.cluster_centers_[l])), 4)})
concert_rows = [s for s in summary if s["concert"]]
for s in concert_rows:
    peers = [t for t in summary if t["cluster"] == s["cluster"] and not t["concert"]]
    s["peer_late21_share_mean"] = round(float(np.mean([t["late21_share"] for t in peers])), 4)
    s["peer_dist_mean"] = round(float(np.mean([t["dist_to_centroid"] for t in peers])), 4)

# 그림 1: 군집 중심 프로파일과 공연일
cols = [NAVY, GOLD, SAGE, BLUE, SUB]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
hrs = np.arange(24)
for rank, c in enumerate(order):
    ax.plot(hrs, km.cluster_centers_[c] * 100, color=cols[rank], lw=2.4, label=name[c])
for d in days:
    if d["concert"]:
        i = days.index(d)
        ax.plot(hrs, X[i] * 100, color=ROSE, lw=1.3, ls="--", alpha=0.9)
ax.plot([], [], color=ROSE, ls="--", label="공연일 (7/1, 7/12, 7/20)")
ax.axvspan(20.5, 23.5, color=ROSE, alpha=0.07)
ax.text(22, ax.get_ylim()[1] * 0.92, "21시 이후", ha="center", color=ROSE, fontsize=9)
ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("출차 시각"); ax.set_ylabel("그날 출차 중 비율 (%)")
ax.set_title(f"출차 패턴 {best_k}개 유형: 공연일 3일 중 2일은 21시 이후 비율이 높음")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax2 = axes[1]
ks = list(scores); vs = [scores[k] for k in ks]
ax2.bar([str(k) for k in ks], vs, color=[ROSE if k == best_k else "#d9d2c6" for k in ks], width=0.55)
for x, v in zip(range(len(ks)), vs):
    ax2.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, color=NAVY)
ax2.set_title("k 선택: 실루엣 계수"); ax2.set_xlabel("군집 수 k"); ax2.grid(axis="x", visible=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
source(fig, "공공데이터포털 15138735 올림픽공원 입출차 정보(2024.7, 215,528건), 15149169 공연정보. k-means 분석은 연구자 작성")
save(fig, "02a_kmeans_profiles.png")

# 그림 2: 달력 형태로 날짜별 유형
fig, ax = plt.subplots(figsize=(9, 3.6))
import datetime as dt
for d, l in zip(days, lab):
    day = dt.date.fromisoformat(d["date"])
    week = (day.day + 0 - 1 + 0) // 7
    col = day.weekday(); row = (day.day - 1 + dt.date(2024, 7, 1).weekday()) // 7
    rank = order.index(l)
    ax.add_patch(plt.Rectangle((col, -row), 0.92, 0.85, color=cols[rank], alpha=0.85))
    ax.text(col + 0.46, -row + 0.5, str(day.day), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    if d["concert"]:
        ax.add_patch(plt.Rectangle((col, -row), 0.92, 0.85, fill=False, ec=ROSE, lw=3))
for i, w in enumerate("월화수목금토일"):
    ax.text(i + 0.46, 1.1, w, ha="center", color=SUB)
ax.set_xlim(-0.1, 7); ax.set_ylim(-4.3, 1.5); ax.axis("off")
for rank, c in enumerate(order):
    ax.scatter([], [], color=cols[rank], marker="s", s=80, label=name[c])
ax.scatter([], [], facecolor="none", edgecolor=ROSE, marker="s", s=80, lw=2, label="공연일")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
ax.set_title("2024년 7월 날짜별 출차 패턴 유형")
source(fig, "공공데이터포털 15138735, 15149169. 연구자 분석")
save(fig, "02b_kmeans_calendar.png")

out = {"silhouette": {k: round(v, 4) for k, v in scores.items()}, "best_k": best_k,
       "clusters": {name[c]: [d["date"][5:] for d, l in zip(days, lab) if l == c] for c in order},
       "concert_days": concert_rows}
json.dump(out, open(os.path.join(FIG, "02_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
