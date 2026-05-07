import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="학생맞춤형 진로유형 진단", layout="wide")
st.title("학생맞춤형 진로유형 진단")
st.caption("학생이 직접 응답하면 진로유형과 지원 방향을 안내합니다. 학교코드를 입력하면 학교 진로환경을 반영한 추가 제안을 제공합니다.")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "맞춤형진로_분석용.xlsx"
MODEL_PATH = BASE_DIR / "rf_final_classweight.pkl"
IMPUTER_PATH = BASE_DIR / "imputer.pkl"
FEATURES_PATH = BASE_DIR / "keep_features.pkl"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"

# -----------------------------
# 파일 불러오기
# -----------------------------
@st.cache_resource
def load_model_objects():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    keep_features = joblib.load(FEATURES_PATH)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    label_map = {int(k): v for k, v in label_map.items()}
    return model, imputer, keep_features, label_map

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH, sheet_name="analysis_ready")

@st.cache_data
def load_school_environment():
    school_df = pd.read_excel(DATA_PATH, sheet_name="1차_학교_환경")
    school_df["Y16AID"] = pd.to_numeric(school_df["Y16AID"], errors="coerce").astype("Int64")
    return school_df

model, imputer, keep_features, label_map = load_model_objects()
df = load_data()
school_df = load_school_environment()

# -----------------------------
# 지역 코드
# -----------------------------
location_map = {
    "서울": 1,
    "부산": 2,
    "대구": 3,
    "인천": 4,
    "광주": 5,
    "대전": 6,
    "울산": 7,
    "경기": 8,
    "강원": 9,
    "충북": 10,
    "충남/세종": 11,
    "전북": 12,
    "전남": 13,
    "경북": 14,
    "경남": 15,
    "제주": 16,
}
location_name_by_code = {v: k for k, v in location_map.items()}

# -----------------------------
# 지역 자동 매핑 변수
# -----------------------------
region_features = [
    "지역_전체교원수",
    "지역_전체교원1인당학생수",
    "지역_고등학교수",
    "지역_일반교사수",
    "지역_전문상담교사수",
    "지역_일반교사1인당학생수",
    "학교당_전문상담교사수",
]

# -----------------------------
# 학교코드 자동 매핑 변수
# -----------------------------
school_features = [
    "학교특색사업지수",
    "학교진로인프라지수",
    "학교진로운영지수",
    "학교진로연계활동지수",
    "학교진로환경종합지수",
]

# -----------------------------
# 학생이 직접 응답할 변수
# -----------------------------
manual_features = [
    "진로방향성_졸업직후계획",
    "진로방향성_교육수준",
    "진로방향성_직업결정",
    "부모진로대화",
    "부모교육기대지수",
    "취업자격준비활동",
    "진로준비종합지수",
    "재학중근로경험",
]

# -----------------------------
# 기본값: 전체 중앙값
# -----------------------------
default_values = {}
for col in keep_features:
    if col in df.columns:
        default_values[col] = pd.to_numeric(df[col], errors="coerce").median()

# -----------------------------
# 지역별 대표값 테이블
# -----------------------------
region_summary = (
    df.groupby("LOCATION")[region_features]
    .median(numeric_only=True)
    .reset_index()
)

def get_region_values(location_code: int):
    row = region_summary[region_summary["LOCATION"] == location_code]
    if row.empty:
        return {col: default_values.get(col, 0) for col in region_features}
    row = row.iloc[0]
    return {col: row[col] for col in region_features}

def normalize_school_code(raw_code: str):
    """사용자 입력 학교코드를 정수형으로 변환한다. 빈칸이면 None."""
    if raw_code is None:
        return None
    code = str(raw_code).strip()
    if code == "":
        return None
    # 사용자가 221276.0처럼 입력한 경우도 처리
    try:
        return int(float(code))
    except ValueError:
        return None

def get_school_values(school_code):
    """학교코드에 해당하는 학교환경 변수와 학교 LOCATION을 반환한다."""
    if school_code is None:
        return None, None

    row = school_df[school_df["Y16AID"] == school_code]
    if row.empty:
        return None, None

    row = row.iloc[0]
    values = {}
    for col in school_features:
        if col in row.index:
            values[col] = pd.to_numeric(row[col], errors="coerce")
        else:
            values[col] = default_values.get(col, 0)

    school_location = row.get("LOCATION", None)
    if pd.notna(school_location):
        school_location = int(school_location)

    return values, school_location

def classify_school_environment(school_values: dict):
    """학교진로환경종합지수 기준으로 학교 진로환경 수준을 분류한다."""
    if not school_values:
        return None, None

    score = school_values.get("학교진로환경종합지수", None)
    if score is None or pd.isna(score):
        return None, None

    q1 = pd.to_numeric(school_df["학교진로환경종합지수"], errors="coerce").quantile(0.25)
    q3 = pd.to_numeric(school_df["학교진로환경종합지수"], errors="coerce").quantile(0.75)

    if score <= q1:
        return "낮음", score
    if score >= q3:
        return "높음", score
    return "보통", score

def build_input_row(user_values: dict):
    row = {}
    for col in keep_features:
        row[col] = default_values.get(col, 0)
    for key, value in user_values.items():
        row[key] = value
    return pd.DataFrame([row], columns=keep_features)

def get_school_based_recommendations(predicted_label: str, school_level: str):
    """학교 진로환경 수준과 예측 유형을 함께 고려한 추가 제안."""
    if school_level is None:
        return []

    common_high = [
        "학교 내 진로상담, 진로체험, 동아리, 연계 프로그램을 우선적으로 확인해보세요.",
        "담당 교사와 상담하면서 학교 안에서 활용 가능한 프로그램을 구체적인 일정으로 연결해보세요.",
    ]
    common_mid = [
        "교내 진로 프로그램을 기본으로 활용하되, 부족한 부분은 커리어넷이나 외부 진로정보로 보완해보세요.",
        "학교 상담과 외부 탐색 활동을 함께 병행하면 진로 선택지를 더 넓힐 수 있습니다.",
    ]
    common_low = [
        "학교 내부 자원만으로 부족할 수 있으므로 커리어넷, 지역 진로체험처, 고용센터, 직업교육기관 등 외부 자원을 함께 찾아보세요.",
        "교사 또는 상담자에게 외부 연계 프로그램 정보를 요청해보는 것이 좋습니다.",
    ]

    type_specific = {
        "4년제 대학 진학": {
            "높음": ["교내 진학상담과 전공탐색 프로그램을 활용해 학과 선택을 구체화해보세요."],
            "보통": ["관심 전공 정보를 학교 상담과 온라인 전공탐색 자료로 함께 비교해보세요."],
            "낮음": ["대학·전공 정보는 커리어넷, 대학알리미, 학과 정보 사이트 등 외부 자료를 적극 활용해보세요."],
        },
        "전문대/직업교육/훈련기관": {
            "높음": ["학교의 진로체험·직업교육 연계 활동을 통해 관심 직무를 직접 확인해보세요."],
            "보통": ["학교 프로그램과 함께 전문대·직업교육기관 정보를 비교해보세요."],
            "낮음": ["지역 직업교육기관, 고용센터, 자격증 과정 등 외부 실습 중심 자원을 우선 탐색해보세요."],
        },
        "취업": {
            "높음": ["학교의 취업상담, 현장실습, 이력서·면접 지원 프로그램을 적극 활용해보세요."],
            "보통": ["학교 상담과 외부 취업정보를 함께 활용해 직무와 자격 준비를 점검해보세요."],
            "낮음": ["고용센터, 직업훈련포털, 지역 취업지원기관을 통해 이력서·면접·자격 준비를 보완해보세요."],
        },
        "미진학·미취업·기타": {
            "높음": ["학교 내 상담과 진로체험 프로그램을 먼저 연결해 진로탐색의 출발점을 만들어보세요."],
            "보통": ["교내 상담을 시작점으로 삼고, 외부 진로검사와 체험활동을 함께 활용해보세요."],
            "낮음": ["교내 자원만으로는 탐색 기회가 부족할 수 있으므로 외부 상담기관과 진로체험처 연계를 우선 고려해보세요."],
        },
    }

    if school_level == "높음":
        base = common_high
    elif school_level == "낮음":
        base = common_low
    else:
        base = common_mid

    extra = type_specific.get(predicted_label, {}).get(school_level, [])
    return extra + base

# -----------------------------
# 기본 정보 입력
# -----------------------------
st.subheader("1. 기본 정보 입력")

info_col1, info_col2 = st.columns(2)

with info_col1:
    selected_region_name = st.selectbox("지역을 선택하세요", list(location_map.keys()))
    selected_location = location_map[selected_region_name]
    region_auto_values = get_region_values(selected_location)

with info_col2:
    school_code_input = st.text_input(
        "학교코드가 있다면 입력하세요",
        placeholder="예: 221276",
        help="담당 교사 또는 상담교사가 안내한 학교코드를 입력하면 학교 진로환경 정보를 반영한 추가 제안을 제공합니다. 입력하지 않아도 기본 진단은 가능합니다.",
    )

normalized_school_code = normalize_school_code(school_code_input)
school_auto_values, school_location = get_school_values(normalized_school_code)
school_level, school_env_score = classify_school_environment(school_auto_values)

if school_code_input.strip() == "":
    st.info("학교코드를 입력하지 않아도 기본 진단은 가능합니다. 이 경우 학교 관련 변수는 전체 자료의 대표값으로 처리됩니다.")
elif school_auto_values is None:
    st.warning("입력한 학교코드를 찾을 수 없습니다. 학교코드를 확인해주세요. 학교환경 정보는 반영하지 않고 기본 진단을 진행합니다.")
else:
    st.success("학교코드가 확인되었습니다. 학교 진로환경 정보를 함께 반영합니다.")
    if school_location is not None and school_location != selected_location:
        school_region_name = location_name_by_code.get(school_location, f"코드 {school_location}")
        st.warning(
            f"입력한 학교코드의 지역({school_region_name})과 선택한 지역({selected_region_name})이 다릅니다. "
            "지역 변수는 선택한 지역 기준으로, 학교환경 변수는 입력한 학교코드 기준으로 반영합니다."
        )

# -----------------------------
# 자기진단 설문
# -----------------------------
st.subheader("2. 자기진단 설문")
col1, col2 = st.columns(2)

with col1:
    career_plan = st.selectbox(
        "졸업 직후 계획에 가장 가까운 것을 선택하세요",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "진학 중심",
            2: "취업·직업교육 중심",
            3: "기타·미정",
        }[x],
    )

    edu_level = st.selectbox(
        "희망하는 교육수준에 가장 가까운 것을 선택하세요",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "대학(4년제) 중심",
            2: "전문·실무교육 중심",
            3: "기타·미정",
        }[x],
    )

    job_decision = st.selectbox(
        "희망 직업이 어느 정도 정해져 있나요?",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x}점",
    )

    parent_talk = st.selectbox(
        "부모님과 진로 대화를 얼마나 자주 하나요?",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x}점",
    )

with col2:
    parent_expect = st.selectbox(
        "부모님의 교육 기대 수준은 어느 정도라고 느끼나요?",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x}점",
    )

    job_prep = st.selectbox(
        "취업·자격 준비를 하고 있나요?",
        options=[0, 1],
        format_func=lambda x: "예" if x == 1 else "아니오",
    )

    career_ready = st.slider(
        "현재 진로준비 수준은 어느 정도인가요?",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
    )

    work_exp = st.selectbox(
        "재학 중 근로경험이 있나요?",
        options=[0, 1],
        format_func=lambda x: "예" if x == 1 else "아니오",
    )

# -----------------------------
# 사용자 입력 정리
# -----------------------------
user_values = {
    "진로방향성_졸업직후계획": career_plan,
    "진로방향성_교육수준": edu_level,
    "진로방향성_직업결정": job_decision,
    "부모진로대화": parent_talk,
    "부모교육기대지수": parent_expect,
    "취업자격준비활동": job_prep,
    "진로준비종합지수": career_ready,
    "재학중근로경험": work_exp,
}

# 지역 변수는 항상 선택 지역 기준으로 반영
user_values.update(region_auto_values)

# 학교코드가 유효한 경우에만 학교환경 변수 반영
school_info_applied = school_auto_values is not None
if school_info_applied:
    user_values.update(school_auto_values)

input_df = build_input_row(user_values)

st.subheader("3. 입력값 확인")
preview_df = pd.DataFrame([user_values]).T.reset_index()
preview_df.columns = ["변수명", "입력값"]
st.dataframe(preview_df, use_container_width=True)

if school_info_applied:
    st.caption("학교명은 표시하지 않으며, 학교코드는 학교 진로환경 정보를 연결하기 위한 내부값으로만 활용됩니다.")
else:
    st.caption("학교코드를 입력하지 않았거나 유효하지 않아 학교환경 변수는 전체 자료의 대표값으로 처리됩니다.")

# -----------------------------
# 예측 실행
# -----------------------------
if st.button("진단 결과 보기"):
    X_input = pd.DataFrame(
        imputer.transform(input_df),
        columns=keep_features,
    )

    pred_class = int(model.predict(X_input)[0])
    pred_proba = model.predict_proba(X_input)[0]

    proba_df = pd.DataFrame({
        "class": model.classes_,
        "class_label": [label_map.get(int(c), str(c)) for c in model.classes_],
        "probability": pred_proba,
    }).sort_values("probability", ascending=False).reset_index(drop=True)

    proba_df["probability_percent"] = (proba_df["probability"] * 100).round(2)

    predicted_label = label_map.get(pred_class, pred_class)

    st.subheader("4. 진단 결과")
    st.success(f"당신에게 가장 가까운 진로유형: {predicted_label}")

    if school_info_applied:
        if school_level is not None:
            st.info(f"입력한 학교코드의 학교 진로환경 정보를 반영했습니다. 학교 진로환경 수준: {school_level}")
        else:
            st.info("입력한 학교코드의 학교 진로환경 정보를 반영했습니다.")
    else:
        st.info("학교코드가 반영되지 않아 학생 응답 정보와 선택한 지역 정보를 중심으로 진단했습니다.")

    st.subheader("5. 유형별 확률")
    st.dataframe(
        proba_df[["class_label", "probability_percent"]],
        use_container_width=True,
    )

    chart_df = proba_df.set_index("class_label")[["probability_percent"]]
    st.bar_chart(chart_df)

    st.subheader("6. 추천 지원 방향")

    prescriptions = {
        "4년제 대학 진학": [
            "희망 전공과 학과 정보를 구체적으로 비교해보세요.",
            "대학 진학 후 필요한 학업 계획을 미리 세워보세요.",
            "관심 전공과 연결되는 체험활동이나 탐색 활동을 늘려보세요.",
        ],
        "전문대/직업교육/훈련기관": [
            "실습 중심 학과와 직업교육 과정을 함께 비교해보세요.",
            "관심 분야와 관련된 자격증·실무 역량 준비 계획을 세워보세요.",
            "현장체험형 진로활동을 통해 적합성을 점검해보세요.",
        ],
        "취업": [
            "희망 직무와 관련된 자격·포트폴리오 준비를 시작해보세요.",
            "이력서·면접 등 취업 준비의 기본 단계를 점검해보세요.",
            "현장실습이나 직무체험 기회를 적극적으로 찾아보세요.",
        ],
        "미진학·미취업·기타": [
            "진로 방향을 먼저 구체화할 수 있도록 자기이해 활동을 해보세요.",
            "진로상담이나 진로검사를 통해 선택지를 넓혀보세요.",
            "부모님 또는 교사와 진로 대화를 늘려보는 것이 도움이 됩니다.",
        ],
    }

    default_prescription = [
        "진로정보 탐색을 조금 더 해보세요.",
        "관심 분야에 대한 체험 기회를 늘려보세요.",
        "필요하면 진로상담을 받아보세요.",
    ]

    for item in prescriptions.get(predicted_label, default_prescription):
        st.write(f"- {item}")

    if school_info_applied:
        st.subheader("7. 학교 자원 활용 제안")
        school_recommendations = get_school_based_recommendations(predicted_label, school_level)
        if school_recommendations:
            for item in school_recommendations:
                st.write(f"- {item}")
        else:
            st.write("- 학교환경 정보가 반영되었지만, 세부 수준을 분류할 수 없어 기본 추천 방향을 우선 참고하세요.")
    else:
        st.subheader("7. 학교 자원 활용 제안")
        st.write("- 학교코드를 입력하면 학교 진로환경 정보를 반영한 추가 제안을 확인할 수 있습니다.")
        st.write("- 학교코드를 모르는 경우 담당 교사 또는 상담교사에게 문의해보세요.")
