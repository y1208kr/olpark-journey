"""
01. FAQ 텍스트 분석: 여정 단계별 TF-IDF, 워드클라우드, 키워드 동시출현 네트워크

목적
- 연역적으로 정한 고객 여정 7단계(+비방문객)에 FAQ를 배정한 뒤,
  단계마다 방문객이 어떤 단어로 질문하는지 확인합니다.
- 단어 사이의 동시출현 관계로, 여러 단계에 걸쳐 함께 묻는 주제를 찾습니다.

입력
- data/faq_text.tsv : 공공데이터포털 15089962 '자주 묻는 질문정보' 원문(번호, 제목, 내용)
- ../data/faq_coded.csv : 문항별 여정 단계 코딩(연구자 코딩)
분석 범위: 올림픽공원 37문항(미사경정공원 12문항 제외)
"""
import csv, os, json, itertools, re
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from _style import *

STAGES = ["계획", "이동", "주차와 입장", "이용", "관람", "퇴장과 정산", "사후", "비방문객"]
STOP = {"올림픽공원", "공원", "경우", "때", "수", "등", "곳", "및", "문의", "참고", "부탁",
        "관련", "내", "중", "이점", "바로가기", "홈페이지", "시", "분", "개", "번", "가지", "정도",
        "고객", "관심", "다음", "해당", "가능", "여부", "위", "점",
        "이용", "진행", "gt", "lt", "nbsp", "이상", "올림픽"}

text = {r[0]: (r[1], r[2]) for r in csv.reader(open(os.path.join(DATA, "faq_text.tsv"), encoding="utf-8"), delimiter="\t")}
coded = [r for r in csv.DictReader(open(os.path.join(ROOT, "data", "faq_coded.csv"), encoding="utf-8")) if r["park"] == "올림픽공원"]
assert len(coded) == 37, len(coded)

kiwi = Kiwi()
# 시설 고유명이 두 단어로 쪼개지지 않도록 사용자 사전에 등록합니다.
for w in ["올림픽공원", "올림픽홀", "장미광장", "평화의광장", "체조경기장", "핸드볼경기장",
          "몽촌토성", "몽촌해자", "유실물센터", "올림픽공원역", "몽촌토성역", "티켓링크"]:
    kiwi.add_user_word(w, "NNP")
def norm(s):
    s = re.sub(r"&[a-z]+;", " ", s)
    return s.replace("평화의 광장", "평화의광장").replace("장미 광장", "장미광장")
def nouns(s):
    out = []
    for t in kiwi.tokenize(s):
        if t.tag in ("NNG", "NNP", "SL") and len(t.form) >= 2 and t.form not in STOP:
            out.append(t.form)
    return out

docs = []
for r in coded:
    title, content = text[r["id"]]
    toks = nouns(norm(title + " " + content))
    docs.append({"id": r["id"], "stage": r["stage"], "toks": toks})

# 1) 단계별 TF-IDF: 같은 단계의 문항을 하나의 문서로 묶고, 단계 사이에서 구별되는 단어를 찾습니다.
stage_docs = {s: [] for s in STAGES}
for d in docs:
    stage_docs[d["stage"]].extend(d["toks"])
present = [s for s in STAGES if stage_docs[s]]
vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, norm="l2")
X = vec.fit_transform([stage_docs[s] for s in present])
terms = np.array(vec.get_feature_names_out())
top = {}
for i, s in enumerate(present):
    row = X[i].toarray().ravel()
    idx = row.argsort()[::-1][:5]
    top[s] = [(terms[j], round(float(row[j]), 3)) for j in idx if row[j] > 0]

fig, axes = plt.subplots(2, 4, figsize=(13, 5.8))
for ax, s in zip(axes.ravel(), STAGES):
    items = top.get(s, [])
    n = sum(1 for d in docs if d["stage"] == s)
    if items:
        w = [t for t, _ in items][::-1]; v = [x for _, x in items][::-1]
        col = ROSE if s in ("이용", "주차와 입장", "계획") else NAVY
        ax.barh(w, v, color=col, height=0.6)
        ax.set_xlim(0, max(v) * 1.25)
        for y, x in enumerate(v):
            ax.text(x + max(v) * 0.03, y, f"{x:.2f}", va="center", fontsize=8, color=SUB)
    ax.set_title(f"{s}  (n={n})", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=7)
fig.suptitle("여정 단계별 FAQ 핵심어 (TF-IDF 상위 5개)", x=0.01, ha="left", fontsize=14, fontweight="bold", color=NAVY)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
source(fig, "공공데이터포털 15089962 한국체육산업개발 자주 묻는 질문정보, 올림픽공원 37문항. 단계 코딩과 분석은 연구자 작성")
save(fig, "01a_tfidf_by_stage.png")

# 2) 워드클라우드: 전체 빈도를 한 장으로 보여 줍니다(보조 자료).
freq = Counter(t for d in docs for t in d["toks"])
def color_func(word, **kw):
    return ROSE if freq[word] >= 8 else (NAVY if freq[word] >= 4 else SUB)
wc = WordCloud(font_path=FONT_PATH, width=1400, height=700, background_color="white",
               prefer_horizontal=1.0, max_words=70, color_func=color_func, random_state=7)
wc.generate_from_frequencies(freq)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
ax.set_title("올림픽공원 FAQ 명사 빈도 워드클라우드 (37문항)")
source(fig, "공공데이터포털 15089962, 명사 추출 kiwipiepy. 진한 분홍은 8회 이상, 남색은 4~7회")
save(fig, "01b_wordcloud.png")

# 3) 동시출현 네트워크: 같은 문항에 함께 나온 명사 쌍을 연결합니다.
MIN_FREQ = 3
keep = {t for t, c in freq.items() if c >= MIN_FREQ}
edges = Counter()
for d in docs:
    u = sorted(set(d["toks"]) & keep)
    for a, b in itertools.combinations(u, 2):
        edges[(a, b)] += 1
G = nx.Graph()
for (a, b), w in edges.items():
    if w >= 2:
        G.add_edge(a, b, weight=w)
# 단어의 주 단계: 그 단어가 가장 많이 나온 단계
word_stage = {}
for t in G.nodes:
    c = Counter(d["stage"] for d in docs if t in d["toks"])
    word_stage[t] = c.most_common(1)[0][0]
    G.nodes[t]["stages"] = len(c)
deg = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G, weight=None)
comm = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))

palette = {"계획": BLUE, "이동": SAGE, "주차와 입장": ROSE, "이용": GOLD, "관람": "#8a6fb0",
           "퇴장과 정산": "#a94c64", "사후": SUB, "비방문객": "#999999"}
pos = nx.spring_layout(G, k=0.9, seed=3, weight="weight")
fig, ax = plt.subplots(figsize=(11, 7.5))
ws = [G[u][v]["weight"] for u, v in G.edges]
nx.draw_networkx_edges(G, pos, ax=ax, width=[0.6 + w * 0.5 for w in ws], edge_color="#d9d2c6", alpha=0.9)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[200 + 2600 * btw[n] + 40 * freq[n] for n in G.nodes],
                       node_color=[palette[word_stage[n]] for n in G.nodes], alpha=0.9, linewidths=1, edgecolors="white")
nx.draw_networkx_labels(G, pos, ax=ax, font_family=plt.rcParams["font.family"], font_size=9, font_color=NAVY)
ax.axis("off")
for s, c in palette.items():
    if s in word_stage.values():
        ax.scatter([], [], color=c, s=60, label=s)
ax.legend(title="단어가 가장 많이 나온 단계", loc="lower left", frameon=False, fontsize=8, title_fontsize=8)
ax.set_title("FAQ 키워드 동시출현 네트워크 (원 크기: 매개 중심성과 빈도)")
source(fig, f"공공데이터포털 15089962. 3회 이상 나온 명사, 2개 이상 문항에서 함께 나온 쌍만 연결. 노드 {G.number_of_nodes()}개, 연결 {G.number_of_edges()}개")
save(fig, "01c_cooccurrence_network.png")

result = {
    "n_docs": len(docs),
    "tfidf_top5_by_stage": top,
    "top_freq": freq.most_common(15),
    "network": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                "density": round(nx.density(G), 3),
                "top_betweenness": sorted(((k, round(v, 3)) for k, v in btw.items()), key=lambda x: -x[1])[:8],
                "top_degree": sorted(((k, round(v, 3)) for k, v in deg.items()), key=lambda x: -x[1])[:8],
                "multi_stage_words": sorted(((n, G.nodes[n]["stages"]) for n in G.nodes if G.nodes[n]["stages"] >= 3), key=lambda x: -x[1]),
                "communities": [sorted(c) for c in comm]},
}
json.dump(result, open(os.path.join(FIG, "01_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(result, ensure_ascii=False, indent=1))
