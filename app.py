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

# -----------------------------------------------------------------------------
# [중요] Google Gemini / Gemma API 설정
# -----------------------------------------------------------------------------
API_KEY = st.secrets["GEMINI_API_KEY"]
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


# -----------------------------------------------------------------------------
# 1. 데이터베이스 설정 (Google Sheets 연동)
# -----------------------------------------------------------------------------
# 주의: 구글 시트에 'users'와 'usage'라는 이름의 워크시트가 미리 생성되어 있어야 하며,
# 각각의 1행에 헤더가 입력되어 있어야 합니다.

def get_connection():
    return st.connection("gsheets", type=GSheetsConnection)


def login_user(username):
    conn = get_connection()
    # ttl=0을 주어 캐시된 데이터가 아닌 실시간 데이터를 불러옴
    users_df = conn.read(worksheet="users", ttl=0).dropna(how="all")

    if username not in users_df['username'].values:
        # 신규 사용자 추가
        new_user = pd.DataFrame([{"username": username, "login_count": 1, "total_points": 0}])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
    else:
        # 기존 사용자 접속 횟수 증가
        users_df.loc[users_df['username'] == username, 'login_count'] += 1

    conn.update(worksheet="users", data=users_df)


def update_user_points(username, points):
    conn = get_connection()
    users_df = conn.read(worksheet="users", ttl=0).dropna(how="all")

    # 포인트 누적
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

def log_error(error_message, current_user="unknown"):
    """발생한 에러를 구글 시트 logs 탭에 기록합니다."""
    conn = get_connection()
    try:
        logs_df = conn.read(worksheet="logs", ttl=0).dropna(how="all")
    except Exception:
        # 처음 생성되어 비어있거나 읽기 실패 시 빈 데이터프레임 생성
        logs_df = pd.DataFrame(columns=["timestamp", "username", "error_message"])
        
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_log = pd.DataFrame([{
        "timestamp": now,
        "username": current_user,
        "error_message": str(error_message)
    }])
    
    logs_df = pd.concat([logs_df, new_log], ignore_index=True)
    conn.update(worksheet="logs", data=logs_df)

# -----------------------------------------------------------------------------
# 2. 외부 API 연동 함수들 (이전 코드와 동일)
# -----------------------------------------------------------------------------
def call_text_api_with_fallback(prompt, models):
    for model in models:
        url = f"{API_BASE_URL}/{model}:generateContent?key={API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status()

            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException:
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
    # 1. 이미지 준비 과정
    try:
        image_bytes = uploaded_file.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        # Streamlit이 제공하는 mime type을 우선 사용하고, 없으면 확장자로 판단
        mime_type = uploaded_file.type if uploaded_file.type else "image/jpeg"
    except Exception as e:
        return None, f"이미지 처리 중 오류 발생: {e}"

    # 2. 프롬프트 설정
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

    # 모델 리스트 (현재 사용 가능한 정확한 ID로 확인 필요)
    gemini_models = [
        "gemini-1.5-flash", 
        "gemini-2.0-flash", 
        "gemini-1.5-flash-8b"
    ]

    # 3. 모델 순차 호출 (Fallback)
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
                response.raise_for_status()

                result_data = response.json()
                if "candidates" in result_data and len(result_data["candidates"]) > 0:
                    result_text = result_data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # 마크다운 코드 블록 제거를 위한 유연한 처리
                    if "```json" in result_text:
                        result_text = result_text.split("```json")[1].split("```")[0]
                    elif "```" in result_text:
                        result_text = result_text.split("```")[1].split("```")[0]
                    
                    result_json = json.loads(result_text.strip())
                    return result_json, None

            except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError):
                # 에러 로그 기록
                error_detail = traceback.format_exc() 
                current_user = st.session_state.get('username', 'unknown')
                log_error(error_detail, current_user)
                # 다음 모델로 시도
                continue 
            except Exception as e:
                # 예상치 못한 치명적 오류
                error_detail = traceback.format_exc()
                current_user = st.session_state.get('username', 'unknown')
                log_error(error_detail, current_user)
                return None, "이미지 처리 중 치명적인 오류가 발생했습니다."

    return None, "⚠️ 모든 AI 모델 호출에 실패했습니다. 관리자 페이지에서 로그를 확인하세요."


# -----------------------------------------------------------------------------
# 3. Streamlit 앱 UI 구성
# -----------------------------------------------------------------------------
st.set_page_config(page_title="청년 기획 봉사단", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stMetric { background-color: #f0f8ff; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 서울 청년 기획 봉사단")
st.markdown("에너지를 지키보세요!")

with st.sidebar:
    st.header("사용자 로그인")
    username_input = st.text_input("닉네임을 입력하세요", placeholder="예: 한강지킴이")
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
            # ... (인증 로직 동일) ...
            pass
        else:
            st.subheader("🛠️ 관리자(Admin) 전용 데이터 통합 조회 대시보드")
            if st.button("🔒 관리자 모드 로그아웃"):
                st.session_state['admin_authenticated'] = False
                st.session_state.pop('username', None)
                st.rerun()

            # [수정] conn을 먼저 선언해야 합니다!
            conn = get_connection()

            st.divider()
            st.markdown("### 🚨 시스템 오류 로그 (최근 순)")
            
            try:
                # [수정] logs 시트를 읽어옵니다.
                logs_df = conn.read(worksheet="logs", ttl=0).dropna(how="all")
                if not logs_df.empty:
                    # timestamp 컬럼이 있는지 확인 후 정렬
                    if "timestamp" in logs_df.columns:
                        logs_df = logs_df.sort_values(by="timestamp", ascending=False)
                    st.dataframe(logs_df, use_container_width=True)
                else:
                    st.info("기록된 오류가 없습니다. 아주 평화롭네요!")
            except Exception as e:
                # 시트가 없거나 읽기 오류 시 출력
                st.error(f"로그 데이터를 불러올 수 없습니다. 구글 시트에 'logs' 탭이 있는지 확인해 주세요. (에러: {e})")

            # 하단 통계 부분 (이미 위에서 conn을 선언했으므로 바로 사용 가능)
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

            st.divider()
            st.markdown("### 👥 전체 사용자 계정 및 포인트 현황")
            st.dataframe(users_df, use_container_width=True)
            st.markdown("### 📊 전체 사용자 에너지 사용 상세 기록")
            st.dataframe(usage_df, use_container_width=True)

    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 에너지 사용 분석", "🤖 AI 코칭", "📸 사진 인증 절약", "🏆 리더보드"])

        with tab1:
            st.subheader("이번 달 생활 에너지 사용량 입력")
            with st.form("usage_form"):
                col1, col2 = st.columns(2)
                elec_input = col1.number_input("전기 사용량 (kWh)", min_value=0.0, value=250.0, step=1.0)
                gas_input = col2.number_input("가스 사용량 (m³)", min_value=0.0, value=20.0, step=1.0)
                submit_btn = st.form_submit_button("기록 및 분석")

            if submit_btn:
                co2_emission = (elec_input * 0.4781) + (gas_input * 2.176)
                save_usage(user, elec_input, gas_input, co2_emission)

                st.success("데이터가 성공적으로 기록되었습니다!")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("전기 사용", f"{elec_input} kWh")
                col_m2.metric("가스 사용", f"{gas_input} m³")
                col_m3.metric("탄소 배출량", f"{co2_emission:.2f} kg CO2e")

            st.divider()
            st.subheader(f"📈 {user}님의 누적 사용 에너지 트렌드")
            df = get_usage_data(user)
            if not df.empty and len(df) > 0:
                fig = px.area(df, x='date', y='co2_kg', markers=True, title="일자별 탄소 배출량 변화")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("아직 기록된 데이터가 없습니다. 사용량을 입력해 주세요.")

        with tab2:
            st.subheader("🤖 AI 에너지 코칭 & Q&A")
            with st.expander("📊 내 사용량 기반 자동 코칭 받기", expanded=False):
                df = get_usage_data(user)
                if not df.empty and len(df) > 0:
                    latest_data = df.iloc[-1]
                    if st.button("AI에게 조언 듣기"):
                        with st.spinner("AI가 조언을 생성 중입니다..."):
                            advice = get_gemma_advice(latest_data['elec_kwh'], latest_data['gas_m3'],
                                                      latest_data['co2_kg'])
                            st.info(advice)
                else:
                    st.warning("먼저 '에너지 사용 분석' 탭에서 데이터를 기록해주세요.")

            st.divider()
            st.markdown("### 💬 무엇이든 물어보세요!")
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("예: 여름철 에어컨 전기세 줄이는 꿀팁 알려줘"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.chat_messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("AI가 답변을 고민 중입니다..."):
                        response = ask_gemma_custom_question(prompt)
                        st.markdown(response)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})

        with tab3:
            st.subheader("📸 실천 인증하고 에코 포인트 받기")
            uploaded_file = st.file_uploader("인증 사진 업로드", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
                st.image(uploaded_file, caption="업로드된 인증 사진", width=400)

                if st.button("AI 인증하기"):
                    result_json, error = analyze_image_with_gemini(uploaded_file)

                    if error:
                        st.error(error)
                    elif result_json:
                        if str(result_json.get("action_found")).lower() == "true":
                            description = result_json.get("description", "에너지 절약 행동")
                            raw_kwh = result_json.get("estimated_save_kwh", "0")

                            import re

                            match = re.search(r'[\d\.]+', str(raw_kwh))
                            saved_kwh = float(match.group(0)) if match else 0.0

                            gained_points = max(10, min(500, int(saved_kwh * 100)))

                            st.success(f"🎉 인증 성공! AI 분석 결과: **'{description}'** 실천이 확인되었습니다.")
                            col_r1, col_r2 = st.columns(2)
                            col_r1.metric("추정 절약량", f"{saved_kwh:.2f} kWh")
                            col_r2.metric("획득 포인트", f"{gained_points} P")

                            update_user_points(user, gained_points)
                            st.balloons()
                        else:
                            st.warning("⚠️ AI가 사진에서 명확한 에너지 절약 행동을 인식하지 못했습니다.")

        with tab4:
            st.subheader("🏆 봉사단 에코 리더보드")
            conn = get_connection()
            users_df = conn.read(worksheet="users", ttl=0).dropna(how="all")

            if not users_df.empty:
                # 데이터 정렬: 누적 포인트 내림차순, 접속 횟수 내림차순
                # 구글 시트에서 문자로 읽어왔을 수 있으므로 숫자로 형변환
                users_df['total_points'] = pd.to_numeric(users_df['total_points'], errors='coerce').fillna(0)
                users_df['login_count'] = pd.to_numeric(users_df['login_count'], errors='coerce').fillna(0)

                leaderboard_df = users_df.sort_values(by=['total_points', 'login_count'],
                                                      ascending=[False, False]).head(10)

                # 인덱스를 순위로 표시
                leaderboard_df = leaderboard_df.reset_index(drop=True)
                leaderboard_df.index = leaderboard_df.index + 1

                # 보여줄 컬럼 한글화 및 선택
                display_df = leaderboard_df[['username', 'login_count', 'total_points']].rename(
                    columns={'username': '닉네임', 'login_count': '접속 횟수', 'total_points': '누적 포인트'}
                )
                st.table(display_df)
            else:
                st.info("아직 가입한 사용자가 없습니다.")

else:
    st.info("👈 왼쪽 사이드바에서 닉네임을 입력하고 로그인 해주세요.")
