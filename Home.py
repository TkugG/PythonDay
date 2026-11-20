from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import os

# ตั้งค่าหน้าเว็บแบบ Wide Screen และซ่อน UI เริ่มต้น
st.set_page_config(
    page_title="MGS3 - TACTICAL SURVIVAL VIEWER",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# สไตล์ Metal Gear Solid 3: Codec Screen & Tactical HUD (ไม่มีอิโมจิ 100%)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    /* กำหนด Theme รวมเป็น MGS CRT Terminal */
    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', 'Courier New', monospace !important;
        background-color: #050A05 !important;
        color: #88C488 !important;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
        max-width: 1300px;
    }

    /* เส้น Scanline แบบจอ CRT สายลับ */
    .crt-overlay {
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.4) 50%);
        background-size: 100% 4px;
        pointer-events: none;
    }

    /* กรอบ HUD ทหาร */
    .mgs-box {
        border: 1px solid #00FF66;
        background: rgba(8, 18, 10, 0.85);
        box-shadow: 0 0 12px rgba(0, 255, 102, 0.2), inset 0 0 10px rgba(0, 255, 102, 0.05);
        padding: 16px;
        margin-bottom: 15px;
    }

    .mgs-title {
        color: #00FF66;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #00FF66;
        padding-bottom: 4px;
        margin-bottom: 12px;
        text-shadow: 0 0 8px rgba(0, 255, 102, 0.6);
    }

    /* แถบหลอด Life & Stamina สไตล์ MGS3 */
    .bar-label {
        font-size: 0.85rem;
        color: #00FF66;
        margin-bottom: 2px;
        letter-spacing: 1px;
    }
    .bar-container {
        background-color: #112211;
        border: 1px solid #00AA44;
        height: 14px;
        width: 100%;
        margin-bottom: 10px;
    }
    .life-fill {
        background: linear-gradient(90deg, #009933, #00FF66);
        height: 100%;
        width: 100%;
    }
    .stamina-fill-high {
        background: linear-gradient(90deg, #DDAA00, #FFCC00);
        height: 100%;
        width: 100%;
        box-shadow: 0 0 8px #FFCC00;
    }
    .stamina-fill-med {
        background: linear-gradient(90deg, #009933, #00FF66);
        height: 100%;
        width: 60%;
    }
    .stamina-fill-low {
        background: linear-gradient(90deg, #990000, #FF2222);
        height: 100%;
        width: 20%;
        box-shadow: 0 0 8px #FF2222;
    }

    /* กล่องสนทนา Codec */
    .codec-dialog {
        background-color: #061007;
        border-left: 3px solid #00FF66;
        padding: 14px;
        color: #C0ECC0;
        font-size: 0.95rem;
        line-height: 1.5;
        letter-spacing: 0.5px;
    }

    /* ลิ้นชักปรับค่าตัวแปรซ้ายล่างแบบ Smooth Floating Drawer */
    div[data-testid="stExpander"] {
        position: fixed !important;
        bottom: 20px !important;
        left: 20px !important;
        width: 440px !important;
        max-width: 90vw !important;
        z-index: 99999 !important;
        background-color: rgba(5, 12, 6, 0.97) !important;
        border: 1px solid #00FF66 !important;
        box-shadow: 0 0 20px rgba(0, 255, 102, 0.35) !important;
        border-radius: 0px !important;
        transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    }
    
    div[data-testid="stExpander"] summary {
        background-color: #0B1C0E !important;
        color: #00FF66 !important;
        font-weight: bold !important;
        letter-spacing: 1.5px !important;
        padding: 10px 14px !important;
        border-bottom: 1px solid #00AA44 !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #FFFFFF !important;
        background-color: #123018 !important;
    }

    /* ปรับแต่ง Sliders ให้เข้ากับสไตล์ Green Military */
    .stSlider > div > div > div > div {
        background-color: #00FF66 !important;
    }
</style>
""", unsafe_allow_html=True)

# โหลดชุดข้อมูลไวน์
@st.cache_data
def load_data():
    df = pd.read_csv("./data/winequality-red.csv", sep=";")
    def categorize(score):
        if score <= 5:
            return "LOW"
        elif score == 6:
            return "MED"
        else:
            return "HIGH"
    df["MGS_RANK"] = df["quality"].apply(categorize)
    return df

# เทรนโมเดล KNN
@st.cache_resource
def train_model(df):
    features = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
    X = df[features]
    y = df["MGS_RANK"]
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model

dt = load_data()
model = train_model(dt)

# Header แถบภารกิจ MGS3
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid #00FF66; padding-bottom: 8px; margin-bottom: 16px;">
    <div>
        <span style="color:#00FF66; font-size:1.3rem; font-weight:bold; letter-spacing:2px;">OPERATION SNAKE EATER // TACTICAL SURVIVAL VIEWER</span><br>
        <span style="color:#4B7A4B; font-size:0.85rem;">MISSION LOCATION: VIRTUOUS MISSION SECTOR 1964 // TARGET: SOVIET MILITARY RATION</span>
    </div>
    <div style="text-align:right;">
        <span style="color:#00FF66; font-size:1.1rem; font-weight:bold;">CODEC: 140.85 MHz</span><br>
        <span style="color:#4B7A4B; font-size:0.85rem;">SECURE COMM CHANNEL</span>
    </div>
</div>
""", unsafe_allow_html=True)

# แท็บพับเก็บค่าตัวแปรลอยอยู่ที่ "ซ้ายล่าง" (Fixed Floating Drawer)
with st.expander("[+] SENSOR TUNER: FIELD RATION PARAMETERS", expanded=False):
    st.markdown("<div style='color:#00FF66; font-size:0.85rem; margin-bottom:8px;'>// ADJUST CHEMICAL SENSORS //</div>", unsafe_allow_html=True)
    
    val_alcohol = st.slider("ALCOHOL PROOF (%)", 8.4, 14.9, 11.5, 0.1)
    val_va = st.slider("VOLATILE ACIDITY (SPOILAGE INDEX)", 0.12, 1.58, 0.40, 0.01)
    val_sulph = st.slider("SULPHATES (PRESERVATION)", 0.33, 2.00, 0.74, 0.01)
    val_citric = st.slider("CITRIC ACID (FRESHNESS)", 0.00, 1.00, 0.38, 0.01)

# คำนวณผลทำนายทางทหารด้วย KNN
features_list = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
user_scenario = pd.DataFrame([[val_alcohol, val_va, val_sulph, val_citric]], columns=features_list)
predicted_rank = model.predict(user_scenario)[0]

# โครงสร้างหน้าจอ Codec: ฝั่งซ้ายรูป Snake / ฝั่งขวาบทวิเคราะห์เสบียงภาคสนาม
col_codec_left, col_codec_right = st.columns([1, 2], gap="medium")

# ฝั่งซ้าย: รูปภาพและข้อมูลสายลับ (Snake)
with col_codec_left:
    st.markdown('<div class="mgs-box">', unsafe_allow_html=True)
    st.markdown('<div class="mgs-title">CALLSIGN: NAKED SNAKE</div>', unsafe_allow_html=True)
    
    if os.path.exists("./img/myprofile.png"):
        st.image("./img/myprofile.png", use_container_width=True)
    
    st.markdown("""
    <div style="font-size:0.8rem; line-height:1.6; margin-top:8px;">
        UNIT: FOX UNIT SPECIAL FORCES<br>
        STATUS: INFILTRATION ACTIVE<br>
        CAMO INDEX: 95% [TIGER STRIPE]
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ฝั่งขวา: การวิเคราะห์เสบียงภาคสนามและแถบสถานะ MGS3
with col_codec_right:
    st.markdown('<div class="mgs-box">', unsafe_allow_html=True)
    st.markdown('<div class="mgs-title">SURVIVAL VIEWER: FIELD SPECIMEN ANALYSIS</div>', unsafe_allow_html=True)
    
    # หลอดพลัง LIFE & STAMINA
    st.markdown('<div class="bar-label">LIFE</div>', unsafe_allow_html=True)
    st.markdown('<div class="bar-container"><div class="life-fill"></div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="bar-label">STAMINA GAIN PREDICTION</div>', unsafe_allow_html=True)
    if predicted_rank == "HIGH":
        st.markdown('<div class="bar-container"><div class="stamina-fill-high"></div></div>', unsafe_allow_html=True)
    elif predicted_rank == "MED":
        st.markdown('<div class="bar-container"><div class="stamina-fill-med"></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="bar-container"><div class="stamina-fill-low"></div></div>', unsafe_allow_html=True)

    # รายละเอียดการประเมินตาม Rank
    if predicted_rank == "HIGH":
        rank_badge = '<span style="color:#FFD700; font-size:1.3rem; font-weight:bold; letter-spacing:2px;">[ S-RANK ] EXCEPTIONAL VINTAGE RATION</span>'
        dialog_text = "PARA-MEDIC: 'Snake, this is an authentic 1964 Crimean private cellar reserve! The chemical balance and alcohol purity are immaculate. Drinking this will fully restore your stamina and steady your aim. Excellent find in the field!'"
        img_url = "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?auto=format&fit=crop&w=700&q=80"
    elif predicted_rank == "MED":
        rank_badge = '<span style="color:#00FF66; font-size:1.3rem; font-weight:bold; letter-spacing:2px;">[ B-RANK ] STANDARD ISSUE MILITARY RATION</span>'
        dialog_text = "PARA-MEDIC: 'It is a standard Soviet field wine, Snake. The acidity is slightly elevated, but completely safe to consume. It should restore moderate stamina while you navigate the swamp.'"
        img_url = "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?auto=format&fit=crop&w=700&q=80"
    else:
        rank_badge = '<span style="color:#FF3333; font-size:1.3rem; font-weight:bold; letter-spacing:2px;">[ D-RANK ] HAZARDOUS / SEVERELY SPOILED</span>'
        dialog_text = "PARA-MEDIC: 'Snake, drop that bottle immediately! The volatile acidity indicates severe bacterial degradation into acetic acid. Consuming this will cause violent stomach cramps and drain your remaining stamina!'"
        img_url = "https://images.unsplash.com/photo-1558001373-7b93ee48ffa0?auto=format&fit=crop&w=700&q=80"

    st.markdown(f"<div style='margin-bottom:12px;'>{rank_badge}</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="codec-dialog">{dialog_text}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ส่วนแสดงภาพตัวอย่างและรายงานเซนเซอร์เคมี
col_specimen, col_sensor = st.columns([1, 2], gap="medium")

with col_specimen:
    st.markdown('<div class="mgs-box">', unsafe_allow_html=True)
    st.markdown('<div class="mgs-title">FIELD SPECIMEN OPTICAL FEED</div>', unsafe_allow_html=True)
    st.image(img_url, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_sensor:
    st.markdown('<div class="mgs-box">', unsafe_allow_html=True)
    st.markdown('<div class="mgs-title">SENSOR TELEMETRY // CURRENT INPUT VS S-RANK BENCHMARK</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("ALCOHOL PROOF", f"{val_alcohol:.1f}%", f"{val_alcohol - 11.52:+.2f}% VS S-RANK")
        st.metric("SULPHATES", f"{val_sulph:.2f}", f"{val_sulph - 0.74:+.2f} VS S-RANK")
    with c2:
        st.metric("VOLATILE ACIDITY", f"{val_va:.2f}", f"{val_va - 0.41:+.2f} VS S-RANK", delta_color="inverse")
        st.metric("CITRIC ACID", f"{val_citric:.2f}", f"{val_citric - 0.38:+.2f} VS S-RANK")
    st.markdown('</div>', unsafe_allow_html=True)
