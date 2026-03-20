import pandas as pd
import numpy as np
import streamlit as st
import os
import traceback

st.set_page_config(page_title="학교 진단 리포트(프로토타입)", layout="wide")

st.write("✅ 앱이 실행되었습니다. CSV 로딩을 시작합니다.")
st.write("현재 폴더 파일:", os.listdir("."))

try:
    df = pd.read_csv("FINAL_학교진단.csv", encoding="utf-8-sig")
    st.write("✅ CSV 로딩 성공. 행/열:", df.shape)
    st.write("컬럼 목록:", list(df.columns))
except Exception:
    st.error("❌ CSV 로딩 실패")
    st.code(traceback.format_exc())
    st.stop()

st.title("학교 진단 리포트(프로토타입)")

sel = st.selectbox("학교 선택", df["학교명"].astype(str).tolist())
row = df[df["학교명"] == sel].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("진학률(실제)", f'{row["진학률(실제)"]:.1%}')
c2.metric("진학률(예측)", f'{row["진학률(예측)"]:.1%}')
c3.metric("잔차", f'{row["잔차(실제-예측)"]:.1%}')
