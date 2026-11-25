"""공통 그래프 설정: 웹 페이지와 같은 색과 한글 글꼴을 씁니다."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
ROOT = os.path.dirname(HERE)
os.makedirs(FIG, exist_ok=True)

FONT_CANDIDATES = [
    os.path.expanduser("~/.fonts/AnaKR-Regular.ttf"),
    os.path.expanduser("~/.fonts/NotoSansKR-ko-400.ttf"),
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
if FONT_PATH:
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
bold = os.path.expanduser("~/.fonts/AnaKR-Bold.ttf")
if os.path.exists(bold):
    fm.fontManager.addfont(bold)
plt.rcParams["axes.unicode_minus"] = False

NAVY, ROSE, GOLD, SAGE, BLUE, SUB, LINE, BG = (
    "#14213d", "#c8566a", "#c79a5b", "#6f9a7e", "#2c5fbf", "#5d6678", "#e7e1d8", "#ffffff")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": BG,
    "axes.facecolor": BG, "axes.edgecolor": LINE, "axes.labelcolor": NAVY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#efe9df", "grid.linewidth": 0.8,
    "xtick.color": SUB, "ytick.color": SUB, "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "axes.titlelocation": "left",
})


def source(fig, text):
    fig.text(0.01, 0.01, "출처: " + text, fontsize=7.5, color=SUB, ha="left", va="bottom")


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path
