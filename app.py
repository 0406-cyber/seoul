import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import requests
import base64
import json
from io import BytesIO
from streamlit_gsheets import GSheetsConnection
import traceback
import re

# -----------------------------------------------------------------------------
# [중요] Google Gemini / Gemma API 설정
# -----------------------------------------------------------------------------
API_KEY = st.secrets["GEMINI_API_KEY"]
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# -----------------------------------------------------------------------------
# 1. 데이터베이스 설정 및 로그 함수
# -----------------------------------------------------------------------------
def get_connection():
    return st.connection("gsheets", type=GSheetsConnection)

def log_error(error_message, current_user="unknown"):
    try:
        conn = get_connection()
        try:
            logs_df = conn.read(worksheet="logs", ttl=0).dropna(how="all")
        except Exception:
            logs_df = pd.DataFrame(columns=["timestamp", "username", "error_message"])
            
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_log = pd.DataFrame([{
            "timestamp": now,
            "username": current_user,
            "error_message": str(error_message)
        }])
        
        updated_logs = pd.concat([logs_df, new_log], ignore_index=True)
        conn.update(worksheet="logs", data=updated_logs)
    except Exception as e:
        st.sidebar.error(f"로그 기록 실패: {e}")

def login_user(username):
    conn = get_connection()
    users_df = conn.read(worksheet="users", ttl=0).dropna(how="all")

    if username not in users_df['username'].values:
        new_user = pd.DataFrame([{"username": username, "login_count": 1, "total_points": 0}])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
    else:
        users_df.loc[users_df['username'] == username, 'login_count'] += 1

    conn.update(worksheet="users", data=users_df)

def update_user_points(username, points):
    conn = get_connection()
    users_df = conn.read(worksheet="users", ttl=0).dropna(how="all")
    users_df.loc[users_df['username'] == username, 'total_points'] += points
    conn.update(worksheet="users", data=users_df)

def save_usage(username, elec, gas, co2):
    conn = get_connection()
    usage_df = conn.read(worksheet="usage", ttl=0).dropna(how="all")
    today = datetime.date.today().strftime("%Y-%m-%d")

    new_data = pd.DataFrame([{
        "username": username,
        "date": today,
        "elec_kwh": float(elec),
        "gas_m3": float(gas),
        "co2_kg": float(co2)
    }])

    usage_df = pd.concat([usage_df, new_data], ignore_index=True)
    conn.update(worksheet="usage", data=usage_df)

def get_usage_data(username):
    conn = get_connection()
    usage_df = conn.read(worksheet="usage", ttl=0).dropna(how="all")

    if usage_df.empty:
        return pd.DataFrame(columns=["username", "date", "elec_kwh", "gas_m3", "co2_kg"])

    user_usage = usage_df[usage_df['username'] == username].sort_values(by="date")
    return user_usage


# -----------------------------------------------------------------------------
# 2. 외부 API 연동 함수들 (Gemma & Gemini)
# -----------------------------------------------------------------------------
def call_text_api_with_fallback(prompt, models):
    for model in models:
        url = f"{API_BASE_URL}/{model}:generateContent?key={API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=15)
            if response.status_code == 429:
                continue
                
            response.raise_for_status()
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            log_error(traceback.format_exc(), st.session_state.get('username', 'system'))
            continue

    return "⚠️ 모든 텍스트 AI 모델 호출에 실패했습니다. API 키나 한도를 확인하세요."

def get_gemma_advice(elec, gas, co2):
    prompt = f"사용자가 이번 달에 전기 {elec}kWh, 가스 {gas}m3를 사용하여 총 {co2:.2f}kg의 탄소를 배출했어. 이 사용자에게 에너지 절약을 독려하고 실생활에서 실천할 수 있는 팁을 친절하게 한국어 3문장 이내로 조언해줘."
    gemma_models = ["gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it", "gemma-3-1b-it"]
    return call_text_api_with_fallback(prompt, gemma_models)

def ask_gemma_custom_question(user_message):
    gemma_models = ["gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it", "gemma-3-1b-it"]
    return call_text_api_with_fallback(user_message, gemma_models)

def analyze_image_with_gemini(uploaded_file):
    try:
        image_bytes = uploaded_file.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = uploaded_file.type if uploaded_file.type else "image/jpeg"
    except Exception as e:
        return None, f"이미지 처리 중 오류 발생: {e}"

    prompt = """
    이 이미지를 분석해서 사용자가 '어떤 에너지 절약 행동'을 하고 있는지 파악해줘.
    그리고 그 행동으로 인해 대략 몇 kWh의 전기를 절약했을지 추정해줘. (예: 전등 끄기 1시간 = 0.05kWh 소모 가정)
    답변은 반드시 아래 JSON 형식으로만 해줘. 마크다운 기호 없이 순수 JSON 텍스트만 출력해.

    {
      "action_found": "true 또는 false",
      "description": "어떤 행동인지 한글 설명 (예: 빈 방 불 끄기)",
      "estimated_save_kwh": "추정 절약량 숫자만 (예: 0.1)"
    }
    """

    gemini_models = [
        "gemini-3-flash-preview", 
        "gemini-2.5-flash", 
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-flash-live-preview"
    ]

    with st.spinner("AI가 이미지를 분석 중입니다..."):
        for model in gemini_models:
            url = f"{API_BASE_URL}/{model}:generateContent?key={API_KEY}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": encoded_image
                                }
                            }
                        ]
                    }
                ]
            }

            try:
                response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30)
                if response.status_code == 429:
                    continue
                    
                response.raise_for_status()

                result_data = response.json()
                if "candidates" in result_data and len(result_data["candidates"]) > 0:
                    result_text = result_data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    json_match = re.search(r'\{.*\}', result_text.replace('\n', ''), re.DOTALL)
                    
                    if json_match:
                        result_json = json.loads(json_match.group())
                        return result_json, None
                    else:
                        if "```json" in result_text:
                            result_text = result_text.split("```json")[1].split("```")[0]
                        elif "```" in result_text:
                            result_text = result_text.split("```")[1].split("```")[0]
                        
                        result_json = json.loads(result_text.strip())
                        return result_json, None

            except Exception as e:
                error_detail = traceback.format_exc() 
                current_user = st.session_state.get('username', 'unknown')
                log_error(error_detail, current_user)
                continue 

    return None, "⚠️ 모든 AI 모델 호출에 실패했습니다. 관리자 페이지에서 로그를 확인하세요."


# -----------------------------------------------------------------------------
# 3. Streamlit 앱 UI 구성
# -----------------------------------------------------------------------------
st.set_page_config(page_title="청년 기획 봉사단", page_icon="🌱", layout="wide")

# 모바일 앱 감성 커스텀 CSS 주입 (다크 & 둥글둥글)
st.markdown("""
    <style>
    /* Pretendard 폰트 적용 */
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        background-color: #121212 !important; /* 완벽한 블랙 배경 */
        color: #e0e0e0 !important;
    }

    /* 앱 전체 배경색 덮어쓰기 */
    .stApp {
        background-color: #121212 !important;
    }

    /* 상단 헤더 투명화 */
    header {
        background-color: transparent !important;
    }

    /* 모서리가 매우 둥근 다크 그레이 카드 디자인 (Metric, Form, Expander) */
    div[data-testid="metric-container"], 
    div[data-testid="stForm"], 
    div[data-testid="stExpander"] {
        background-color: #1e1e1e !important;
        border-radius: 24px !important;
        padding: 24px;
        border: 1px solid #2c2c2e !important;
        box-shadow: none !important;
    }

    /* Metric 내부 글씨 색상 */
    div[data-testid="stMetricLabel"] {
        color: #98989d !important;
        font-weight: 500;
        font-size: 14px;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* 탭(Tab)을 사진처럼 동글동글한 알약(Pill) 형태로 변경 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: none !important;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e !important;
        border-radius: 50px !important; /* 알약 모양 */
        padding: 12px 20px !important;
        font-weight: 600 !important;
        color: #98989d !important;
        border: 1px solid #2c2c2e !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important; /* 선택된 탭은 흰색 */
        color: #121212 !important; /* 글씨는 검정색 */
        border: none !important;
    }
    
    /* 버튼 디자인 (알약 형태) */
    .stButton>button {
        background-color: #2c2c2e;
        color: #ffffff;
        border-radius: 50px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #3a3a3c;
        color: #ffffff;
    }

    /* 입력창 디자인 */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1e1e1e !important;
        color: white !important;
        border-radius: 16px !important;
        border: 1px solid #3a3a3c !important;
    }

    /* 텍스트 색상 조절 */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* 사이드바 배경 및 텍스트 */
    [data-testid="stSidebar"] {
        background-color: #1a1a1c !important;
        border-right: 1px solid #2c2c2e !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 교내 에코 셔틀")
st.markdown("나의 에너지 절약 기록 및 AI 분석")

with st.sidebar:
    st.header("사용자 로그인")
    username_input = st.text_input("닉네임을 입력하세요", placeholder="예: 무악학사생")
    if st.button("접속/회원가입"):
        if username_input:
            st.session_state['username'] = username_input
            login_user(username_input)
            st.success(f"환영합니다, {username_input}님!")
        else:
            st.warning("닉네임을 입력해주세요.")

if 'username' in st.session_state:
    user = st.session_state['username']

    if user == 'admin':
        if not st.session_state.get('admin_authenticated', False):
            st.subheader("🔒 관리자 권한 인증")
            st.info("관리자 대시보드에 접근하려면 비밀번호가 필요합니다.")
            admin_pw = st.text_input("관리자 비밀번호를 입력하세요", type="password")

            if st.button("인증하기"):
                if admin_pw == "seoul1234":
                    st.session_state['admin_authenticated'] = True
                    st.success("인증 성공! 대시보드를 불러옵니다...")
                    st.rerun()
                else:
                    st.error("비밀번호가 일치하지 않습니다.")
        else:
            st.subheader("🛠️ 관리자(Admin) 전용 데이터 통합 조회 대시보드")
            if st.button("🔒 관리자 모드 로그아웃"):
                st.session_state['admin_authenticated'] = False
                st.session_state.pop('username', None)
                st.rerun()

            conn = get_connection()
            st.divider()
            st.markdown("### 🚨 시스템 오류 로그 (최근 순)")
            
            try:
                logs_df = conn.read(worksheet="logs", ttl=0).dropna(how="all")
                if not logs_df.empty:
                    if "timestamp" in logs_df.columns:
                        logs_df = logs_df.sort_values(by="timestamp", ascending=False)
                    st.dataframe(logs_df, use_container_width=True)
                else:
                    st.info("기록된 오류가 없습니다. 아주 평화롭네요!")
            except Exception as e:
                st.error(f"로그 데이터를 불러올 수 없습니다. (에러: {e})")

            users_df = conn.read(worksheet="users", ttl=0).dropna(how="all")
            usage_df = conn.read(worksheet="usage", ttl=0).dropna(how="all")

            col_a1, col_a2, col_a3 = st.columns(3)
            col_a1.metric("총 가입자 수", f"{len(users_df)} 명")
            
            if not users_df.empty and 'total_points' in users_df.columns:
                total_points = pd.to_numeric(users_df['total_points'], errors='coerce').sum()
            else:
                total_points = 0
            col_a2.metric("전체 누적 포인트", f"{total_points} P")
            col_a3.metric("총 기록된 데이터 수", f"{len(usage_df)} 건")

    else:
        # 사진의 둥근 버튼 형태를 상단 탭으로 구현
        tab1, tab2, tab3, tab4 = st.tabs(["📊 기록", "🤖 코칭", "📸 인증", "🏆 랭킹"])

        with tab1:
            st.subheader("이번 달 사용량 입력")
            with st.form("usage_form"):
                col1, col2 = st.columns(2)
                elec_input = col1.number_input("전기 사용량 (kWh)", min_value=0.0, value=250.0, step=1.0)
                gas_input = col2.number_input("가스 사용량 (m³)", min_value=0.0, value=20.0, step=1.0)
                submit_btn = st.form_submit_button("저장 및 분석")

            if submit_btn:
                co2_emission = (elec_input * 0.4781) + (gas_input * 2.176)
                save_usage(user, elec_input, gas_input, co2_emission)

                st.success("데이터가 기록되었습니다!")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("전기 사용", f"{elec_input} kWh")
                col_m2.metric("가스 사용", f"{gas_input} m³")
                col_m3.metric("탄소 배출량", f"{co2_emission:.2f} kg CO2e")

            st.divider()
            st.subheader(f"📈 {user}님의 트렌드")
            df = get_usage_data(user)
            if not df.empty and len(df) > 0:
                fig = px.area(df, x='date', y='co2_kg', markers=True)
                # 다크 모드 차트 적용
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white")
                ) 
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("아직 기록된 데이터가 없습니다.")

        with tab2:
            st.subheader("🤖 AI 에너지 코칭")
            with st.expander("📊 내 사용량 기반 자동 코칭 받기", expanded=False):
                df = get_usage_data(user)
                if not df.empty and len(df) > 0:
                    latest_data = df.iloc[-1]
                    if st.button("조언 듣기"):
                        with st.spinner("생성 중..."):
                            advice = get_gemma_advice(latest_data['elec_kwh'], latest_data['gas_m3'],
                                                      latest_data['co2_kg'])
                            st.info(advice)
                else:
                    st.warning("데이터를 먼저 기록해주세요.")

            st.divider()
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("질문을 입력하세요..."):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.chat_messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("고민 중..."):
                        response = ask_gemma_custom_question(prompt)
                        st.markdown(response)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})

        with tab3:
            st.subheader("📸 실천 인증")
            uploaded_file = st.file_uploader("인증 사진 업로드", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
                st.image(uploaded_file, caption="업로드된 사진", use_container_width=True)

                if st.button("AI 인증하기"):
                    result_json, error = analyze_image_with_gemini(uploaded_file)

                    if error:
                        st.error(error)
                    elif result_json:
                        if str(result_json.get("action_found")).lower() == "true":
                            description = result_json.get("description", "행동")
                            raw_kwh = result_json.get("estimated_save_kwh", "0")

                            match = re.search(r'[\d\.]+', str(raw_kwh))
                            saved_kwh = float(match.group(0)) if match else 0.0

                            gained_points = max(10, min(500, int(saved_kwh * 100)))

                            st.success(f"🎉 **'{description}'** 확인 완료!")
                            col_r1, col_r2 = st.columns(2)
                            col_r1.metric("절약량", f"{saved_kwh:.2f} kWh")
                            col_r2.metric("포인트", f"{gained_points} P")

                            update_user_points(user, gained_points)
                        else:
                            st.warning("행동을 인식하지 못했습니다.")

        with tab4:
            st.subheader("🏆 리더보드")
            conn = get_connection()
            users_df = conn.read(worksheet="users", ttl=0).dropna(how="all")

            if not users_df.empty:
                users_df['total_points'] = pd.to_numeric(users_df['total_points'], errors='coerce').fillna(0)
                users_df['login_count'] = pd.to_numeric(users_df['login_count'], errors='coerce').fillna(0)

                leaderboard_df = users_df.sort_values(by=['total_points', 'login_count'],
                                                      ascending=[False, False]).head(10)

                leaderboard_df = leaderboard_df.reset_index(drop=True)
                leaderboard_df.index = leaderboard_df.index + 1

                display_df = leaderboard_df[['username', 'login_count', 'total_points']].rename(
                    columns={'username': '닉네임', 'login_count': '접속 횟수', 'total_points': '누적 포인트'}
                )
                st.table(display_df)

else:
    st.info("👈 닉네임을 입력하고 로그인 해주세요.")
