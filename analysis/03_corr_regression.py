"""
03. 늦은 출차의 상관과 회귀 분석

목적
- 21시 이후 출차 대수가 공연 여부, 주말 여부와 어떤 관계를 보이는지 확인합니다.
- 공원 운영 여정(정산, 매점, 안내 인력의 운영 시간)이 늦은 퇴장 수요와 맞는지 판단하는 근거로 씁니다.
주의
- 자료는 2024년 7월 31일(n=31)뿐이고 공연일은 3일입니다. 결과는 상관 관계이며 인과를 뜻하지 않습니다.
- 표본이 작아 이분산에 강한 HC3 표준오차를 쓰고, 공연일 하나씩을 빼고 다시 추정해 결과가 한 날에 좌우되는지 확인합니다.
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from _style import *
from _load import load_days

days = load_days()
df = pd.DataFrame([{
    "date": d["date"][5:], "dow": d["dow"], "concert": d["concert"], "weekend": d["weekend"],
    "total_in": int(d["in"].sum()), "late_exit21": int(d["out"][21:].sum()),
    "early_in5_7": int(d["in"][5:7].sum()),
    "short_dwell_share": d["dwell"][0] / d["dwell"].sum(),
    "late_share": d["out"][21:].sum() / d["out"].sum(),
} for d in days])

# 정의: 그날 21시 00분부터 23시 59분 사이에 출차한 대수(출차 시각 기준).
# 일별 자료의 시간대 합계가 집계 자료(aggregates.json)의 평일, 주말 합계와 같은지 확인합니다.
inout = json.load(open(os.path.join(ROOT, "data", "aggregates.json"), encoding="utf-8"))["inout"]
for kind, flag in (("weekday", 0), ("weekend", 1)):
    sub = [d for d in days if d["weekend"] == flag]
    assert [int(x) for x in sum(d["out"] for d in sub)] == inout[kind]["out"], kind
    assert [int(x) for x in sum(d["in"] for d in sub)] == inout[kind]["in"], kind

vars_ = ["late_exit21", "total_in", "early_in5_7", "short_dwell_share", "concert", "weekend"]
labels = {"late_exit21": "21시 이후 출차", "total_in": "하루 입차", "early_in5_7": "5~7시 입차",
          "short_dwell_share": "30분 미만 체류 비율", "concert": "공연일", "weekend": "주말"}
corr = df[vars_].corr(method="pearson")
spear = df[vars_].corr(method="spearman")

m1 = smf.ols("late_exit21 ~ concert + weekend", df).fit(cov_type="HC3")
m2 = smf.ols("late_exit21 ~ concert + weekend + total_in", df).fit(cov_type="HC3")
m3 = smf.ols("late_share ~ concert + weekend", df).fit(cov_type="HC3")
loo = {}
for cd in df.loc[df.concert == 1, "date"]:
    f = smf.ols("late_exit21 ~ concert + weekend", df[df.date != cd]).fit(cov_type="HC3")
    loo[cd] = {"b_concert": round(f.params["concert"], 1), "p": round(f.pvalues["concert"], 4)}

def tidy(m):
    ci = m.conf_int()
    return {k: {"b": round(m.params[k], 4), "se_hc3": round(m.bse[k], 4), "p": round(m.pvalues[k], 4),
                "ci95": [round(ci.loc[k, 0], 4), round(ci.loc[k, 1], 4)]} for k in m.params.index} | {
        "n": int(m.nobs), "r2": round(m.rsquared, 3), "adj_r2": round(m.rsquared_adj, 3)}

# 그림 1: 상관 행렬
fig, ax = plt.subplots(figsize=(6.8, 5.6))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("rb", [BLUE, "#ffffff", ROSE])
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
names = [labels[v] for v in vars_]
ax.set_xticks(range(len(vars_))); ax.set_xticklabels(names, rotation=35, ha="right")
ax.set_yticks(range(len(vars_))); ax.set_yticklabels(names)
ax.grid(False)
for i in range(len(vars_)):
    for j in range(len(vars_)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(corr.values[i, j]) > 0.6 else NAVY)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("일별 지표 사이의 피어슨 상관 (n=31일)")
fig.tight_layout(rect=(0, 0.04, 1, 1))
source(fig, "공공데이터포털 15138735, 15149169. 연구자 분석. 상관은 인과를 뜻하지 않음")
save(fig, "03a_correlation.png")

# 그림 2: 21시 이후 출차와 회귀 계수
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2, 1]})
ax = axes[0]
colors = [ROSE if c else (GOLD if w else NAVY) for c, w in zip(df.concert, df.weekend)]
ax.bar(df.date, df.late_exit21, color=colors, width=0.7)
med = df.late_exit21.median()
ax.axhline(med, color=SUB, lw=1, ls="--"); ax.text(30.6, med + 15, f"중앙값 {med:.0f}", ha="right", color=SUB, fontsize=9)
for r in df.itertuples():
    if r.concert:
        ax.text(r.Index, r.late_exit21 + 20, f"{r.late_exit21:,}", ha="center", color=ROSE, fontsize=9, fontweight="bold")
ax.set_xticks(range(0, 31, 2)); ax.set_xticklabels(df.date[::2], rotation=45, fontsize=8)
ax.set_title("날짜별 21시 이후 출차 대수"); ax.grid(axis="x", visible=False)
for lab_, c in [("평일", NAVY), ("주말", GOLD), ("공연일", ROSE)]:
    ax.bar([0], [0], color=c, label=lab_)
ax.legend(frameon=False, loc="upper left")
ax = axes[1]
terms = ["concert", "weekend"]
b = [m1.params[t] for t in terms]; ci = m1.conf_int().loc[terms]
ax.errorbar(b, [1, 0], xerr=[np.array(b) - ci[0].values, ci[1].values - np.array(b)], fmt="o",
            color=NAVY, ecolor=SUB, capsize=4, ms=8)
ax.axvline(0, color=SUB, lw=1)
ax.set_yticks([1, 0]); ax.set_yticklabels(["공연일", "주말"])
for y, t in zip([1, 0], terms):
    ax.text(m1.params[t], y + 0.18, f"{m1.params[t]:+.0f}대 (p={m1.pvalues[t]:.3f})", ha="center", fontsize=9, color=NAVY)
ax.set_ylim(-0.6, 1.6)
ax.set_title("OLS 계수와 95% 신뢰구간 (HC3)"); ax.set_xlabel("21시 이후 출차 대수 차이")
fig.tight_layout(rect=(0, 0.05, 1, 1))
source(fig, f"공공데이터포털 15138735, 15149169. 연구자 분석. n=31, R²={m1.rsquared:.2f}. 공연일은 3일뿐이므로 상관 관계로만 해석")
save(fig, "03b_late_exit_regression.png")

out = {"pearson": corr.round(3).to_dict(), "spearman_late_exit21": spear["late_exit21"].round(3).to_dict(),
       "m1_late_exit21": tidy(m1), "m2_with_total_in": tidy(m2), "m3_late_share": tidy(m3),
       "leave_one_concert_out": loo, "late_exit21_median": float(med),
       "concert_days": df[df.concert == 1][["date", "dow", "late_exit21", "late_share"]].to_dict("records")}
json.dump(out, open(os.path.join(FIG, "03_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: out[k] for k in ["m1_late_exit21", "m2_with_total_in", "m3_late_share", "leave_one_concert_out", "spearman_late_exit21", "concert_days"]}, ensure_ascii=False, indent=1, default=float))
