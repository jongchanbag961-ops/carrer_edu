import pandas as pd
import numpy as np
import streamlit as st
import os
import traceback

st.set_page_config(page_title="학교 진단 리포트(프로토타입)", layout="wide")

# --- Load CSV (debug friendly) ---
st.write("✅ 앱이 실행되었습니다. CSV 로딩을 시작합니다.")
st.write("현재 폴더 파일:", os.listdir("."))

try:
    df = pd.read_csv("FINAL_학교진단.csv", encoding="utf-8-sig")
    st.write("✅ CSV 로딩 성공. 행/열:", df.shape)
except Exception:
    st.error("❌ CSV 로딩 실패")
    st.code(traceback.format_exc())
    st.stop()

# --- Basic cleaning (safe numeric conversion) ---
num_cols = [
    "진학률(실제)", "진학률(예측)", "잔차(실제-예측)", "SEI", "ASI", "재학생수",
    "동아리참여율_Z","동아리다양성_Z","방과후수강강도_Z","방과후공급밀도_Z","특색도입개수_Z"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

st.title("학교 진단 리포트(프로토타입)")
st.caption("FINAL_학교진단 기반: 학교별 지표/유형/처방/벤치마크 추천을 한 화면에서 확인")

# --- Sidebar filters ---
st.sidebar.header("학교 선택")
q = st.sidebar.text_input("학교명 검색", "")
f = df.copy()
if q.strip():
    f = f[f["학교명"].astype(str).str.contains(q.strip(), na=False)]

if "구" in f.columns:
    gu = st.sidebar.multiselect("구", sorted(f["구"].dropna().unique().tolist()))
    if gu:
        f = f[f["구"].isin(gu)]

if "설립구분" in f.columns:
    est = st.sidebar.multiselect("설립구분", sorted(f["설립구분"].dropna().unique().tolist()))
    if est:
        f = f[f["설립구분"].isin(est)]

sel = st.sidebar.selectbox(
    "학교",
    f["학교명"].astype(str).tolist() if len(f) else df["학교명"].astype(str).tolist()
)
row = df[df["학교명"].astype(str) == str(sel)].iloc[0]
# --- KPI row ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("진학률(실제)", f'{row.get("진학률(실제)", np.nan):.1%}' if pd.notna(row.get("진학률(실제)", np.nan)) else "-")
c2.metric("진학률(예측)", f'{row.get("진학률(예측)", np.nan):.1%}' if pd.notna(row.get("진학률(예측)", np.nan)) else "-")
c3.metric("잔차(실제-예측)", f'{row.get("잔차(실제-예측)", np.nan):.1%}' if pd.notna(row.get("잔차(실제-예측)", np.nan)) else "-")
c4.metric("SEI", f'{row.get("SEI", np.nan):.2f}' if pd.notna(row.get("SEI", np.nan)) else "-")
c5.metric("ASI", f'{row.get("ASI", np.nan):.2f}' if pd.notna(row.get("ASI", np.nan)) else "-")

# --- Summary line ---
st.info(
    f'**유형:** {row.get("유형","")}  |  '
    f'**우선개선영역:** {row.get("우선개선영역","")}  |  '
    f'**강점요인:** {row.get("강점요인","")}'
)

left, right = st.columns([1, 1])

with left:
    st.subheader("활동지원(Z) 프로필")
    z_cols = ["동아리참여율_Z","동아리다양성_Z","방과후수강강도_Z","방과후공급밀도_Z","특색도입개수_Z"]
    z_map = {
        "동아리참여율_Z":"동아리참여",
        "동아리다양성_Z":"동아리다양성",
        "방과후수강강도_Z":"방과후수강",
        "방과후공급밀도_Z":"방과후공급",
        "특색도입개수_Z":"특색도입",
    }
    z = {z_map[c]: float(row.get(c, 0.0)) for c in z_cols if c in df.columns}
    st.bar_chart(pd.Series(z))

    st.subheader("기본 정보")
    st.write({
        "school_ID": row.get("school_ID",""),
        "구": row.get("구",""),
        "설립구분": row.get("설립구분",""),
        "재학생수": int(row.get("재학생수", 0)) if pd.notna(row.get("재학생수", np.nan)) else ""
    })

with right:
    st.subheader("AI 리포트(템플릿)")
    rec = row.get("맞춤처방(1줄)", row.get("맞춤처방",""))
    msg = row.get("성공요인(1줄)", row.get("성공요인",""))
    st.markdown(
        f"""
**1) 요약 진단**
- 학교: {row.get('학교명','')} ({row.get('구','')}, {row.get('설립구분','')})
- 진학률: {row.get('진학률(실제)', np.nan):.1%} (예측 {row.get('진학률(예측)', np.nan):.1%}, 잔차 {row.get('잔차(실제-예측)', np.nan):.1%})
- 유형: {row.get('유형','')}

**2) 우선 개선**
- 개선영역: {row.get('우선개선영역','')}
- 처방: {rec}

**3) 강점/벤치마크 관점**
- 강점요인: {row.get('강점요인','')}
- 시사점: {msg}
"""
    )
# --- Benchmark recommendation ---
st.markdown("---")
st.subheader("벤치마크 추천(유사학교 3개)")

cand = df.copy()
if "구" in df.columns and "설립구분" in df.columns:
    same = df[
        (df["구"] == row.get("구","")) &
        (df["설립구분"] == row.get("설립구분","")) &
        (df["school_ID"] != row.get("school_ID",""))
    ].copy()
    cand = same if len(same) else df[df["school_ID"] != row.get("school_ID","")].copy()

# 규모 ±30% 필터 (가능하면 적용)
if "재학생수" in cand.columns and pd.notna(row.get("재학생수", np.nan)):
    size = float(row.get("재학생수", 1.0)) if float(row.get("재학생수", 0)) else 1.0
    cand["size_ratio"] = np.abs(cand["재학생수"] - size) / size
    cand = cand[cand["size_ratio"] <= 0.30].copy()

cand = cand.sort_values("잔차(실제-예측)", ascending=False).head(3)

show_cols = ["학교명","구","설립구분","진학률(실제)","잔차(실제-예측)","강점요인"]
show_cols = [c for c in show_cols if c in cand.columns]
st.dataframe(cand[show_cols], use_container_width=True)

# --- Download report ---
st.markdown("---")
st.subheader("리포트 다운로드")
report_txt = f"""[학교 진단 리포트]
학교: {row.get('학교명','')} ({row.get('구','')}, {row.get('설립구분','')})
진학률: {row.get('진학률(실제)', np.nan):.1%} (예측 {row.get('진학률(예측)', np.nan):.1%}, 잔차 {row.get('잔차(실제-예측)', np.nan):.1%})
SEI: {row.get('SEI', np.nan):.2f} | ASI: {row.get('ASI', np.nan):.2f}
유형: {row.get('유형','')}

우선개선영역: {row.get('우선개선영역','')}
처방: {rec}

강점요인: {row.get('강점요인','')}
성공요인: {msg}
"""
st.download_button(
    label="리포트 TXT 다운로드",
    data=report_txt.encode("utf-8"),
    file_name=f"{row.get('학교명','학교')}_리포트.txt",
    mime="text/plain"
)
