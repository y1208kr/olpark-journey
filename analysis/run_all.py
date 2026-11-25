"""세 분석을 차례로 실행하고 그림과 결과 파일을 figures/ 폴더에 저장합니다."""
import runpy, os
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
for f in ["01_text_tfidf_network.py", "02_kmeans_dayprofiles.py", "03_corr_regression.py"]:
    print("=== 실행:", f)
    runpy.run_path(f, run_name="__main__")
