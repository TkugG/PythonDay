from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import os

# ตั้งค่าหน้าเว็บแบบกว้างและเปิด Sidebar เป็นค่าเริ่มต้น
st.set_page_config(
    page_title="Tactical Wine Quality Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# สไตล์ CSS: ฟอนต์ Prompt อ่านง่าย สบายตา คมชัด สไตล์ Tactical Dark โดยไม่มีอิโมจิ
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, p, div, span, label {
        font-family: 'Prompt', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* พื้นหลังโทน Tactical Slate Dark */
    .stApp {
        background-color: #070C08 !important;
        color: #F1F5F9 !important;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* กล่องข้อมูล Tactical Card คมชัด ไม่เบลอ */
    .tactical-card {
        background-color: #0E1610;
        border: 1px solid #1E3824;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .tactical-header {
        color: #00FF66;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 1px;
        border-bottom: 1px solid #1E3824;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* สไตล์ป้ายผลการประเมินคุณภาพ */
    .badge-s {
        background-color: rgba(0, 255, 102, 0.12);
        color: #00FF66;
        border: 1px solid #00FF66;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 1.15rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    .badge-b {
        background-color: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 1.15rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    .badge-d {
        background-color: rgba(244, 63, 94, 0.12);
        color: #F43F5E;
        border: 1px solid #F43F5E;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 1.15rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }

    /* แถบหลอดประเมินคะแนนรสสัมผัส (Sensory Index) */
    .bar-bg {
        background-color: #152218;
        border: 1px solid #28442E;
        height: 12px;
        border-radius: 6px;
        width: 100%;
        overflow: hidden;
        margin-bottom: 14px;
    }
    .bar-s {
        background: #00FF66;
        height: 100%;
        width: 100%;
    }
    .bar-b {
        background: #38BDF8;
        height: 100%;
        width: 65%;
    }
    .bar-d {
        background: #F43F5E;
        height: 100%;
        width: 25%;
    }

    /* กล่องข้อความวิเคราะห์ผลไวน์ */
    .analysis-text {
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.6;
        background-color: #09100A;
        border-left: 3px solid #00FF66;
        padding: 12px 16px;
        border-radius: 0 4px 4px 0;
    }

    /* ปรับแต่ง Sidebar ให้เป็นแผงควบคุม Tactical */
    section[data-testid="stSidebar"] {
        background-color: #0A120C !important;
        border-right: 1px solid #1E3824 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00FF66 !important;
    }
</style>
""", unsafe_allow_html=True)

# ฟังก์ชันโหลดและจัดกลุ่มข้อมูลไวน์
@st.cache_data
def load_wine_data():
    df = pd.read_csv("./data/winequality-red.csv", sep=";")
    def categorize(score):
        if score <= 5:
            return "D-Rank: คุณภาพต่ำ"
        elif score == 6:
            return "B-Rank: คุณภาพปานกลาง"
        else:
            return "S-Rank: คุณภาพดีเยี่ยม"
    df["เกรดคุณภาพ"] = df["quality"].apply(categorize)
    return df

# เทรนโมเดล KNN
@st.cache_resource
def train_knn_model(df):
    features = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
    X = df[features]
    y = df["เกรดคุณภาพ"]
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model

df_wine = load_wine_data()
knn_model = train_knn_model(df_wine)

# แถบควบคุมตัวแปรสารเคมีใน Sidebar (พับเก็บได้ ไม่บังหน้าจอหลัก)
with st.sidebar:
    st.markdown("### แผงควบคุมตัวแปรไวน์")
    st.caption("ปรับแต่งคุณลักษณะทางเคมีเพื่อจำลองผลลัพธ์แบบทันที")
    
    if os.path.exists("./img/myprofile.png"):
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            st.image("./img/myprofile.png", width=75)
        with col_p2:
            st.markdown("<div style='font-size:0.85rem; color:#94A3B8; margin-top:8px;'>ผู้ประเมินคุณภาพ<br><strong style='color:#F1F5F9; font-size:0.95rem;'>Wine Analyst</strong></div>", unsafe_allow_html=True)
        st.divider()

    val_alcohol = st.slider("ปริมาณแอลกอฮอล์ (Alcohol %)", 8.4, 14.9, 11.5, 0.1)
    val_va = st.slider("กรดระเหย (Volatile Acidity: g/dm³)", 0.12, 1.58, 0.40, 0.01)
    val_sulph = st.slider("ปริมาณซัลเฟต (Sulphates: g/dm³)", 0.33, 2.00, 0.74, 0.01)
    val_citric = st.slider("กรดซิตริก (Citric Acid: g/dm³)", 0.00, 1.00, 0.38, 0.01)

# ประมวลผลการทำนายด้วย KNN แบบ Real-time
features_list = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
input_scenario = pd.DataFrame([[val_alcohol, val_va, val_sulph, val_citric]], columns=features_list)
prediction_result = knn_model.predict(input_scenario)[0]

# หัวข้อหลักบนหน้าจอ
st.markdown("""
<div style="border-bottom: 1px solid #1E3824; padding-bottom: 10px; margin-bottom: 20px;">
    <h2 style="color:#00FF66; margin:0; font-size:1.6rem; font-weight:600; letter-spacing:0.5px;">ระบบวิเคราะห์และจำแนกคุณภาพไวน์แดง (Tactical Wine Analyzer)</h2>
    <div style="color:#94A3B8; font-size:0.9rem; margin-top:4px;">ประเมินโครงสร้างทางเคมีและจำแนกเกรดมาตรฐานด้วยแบบจำลอง K-Nearest Neighbors</div>
</div>
""", unsafe_allow_html=True)

# รูปภาพไวน์ตามระดับคุณภาพ
img_dict = {
    "S-Rank: คุณภาพดีเยี่ยม": "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?auto=format&fit=crop&w=700&q=80",
    "B-Rank: คุณภาพปานกลาง": "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?auto=format&fit=crop&w=700&q=80",
    "D-Rank: คุณภาพต่ำ": "https://images.unsplash.com/photo-1558001373-7b93ee48ffa0?auto=format&fit=crop&w=700&q=80"
}

# จัด Layout หลักเป็น 2 คอลัมน์ที่ชัดเจนและไม่ซ้อนทับกัน
col_view_left, col_view_right = st.columns([1, 1], gap="large")

with col_view_left:
    st.markdown('<div class="tactical-card">', unsafe_allow_html=True)
    st.markdown('<div class="tactical-header">ผลการจำแนกเกรดคุณภาพ (Classification Result)</div>', unsafe_allow_html=True)
    
    # แสดงป้ายระดับคุณภาพและหลอดประเมินคะแนน
    if "S-Rank" in prediction_result:
        st.markdown('<div class="badge-s">S-Rank: ไวน์คุณภาพดีเยี่ยม</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.85rem; color:#94A3B8; margin-bottom:4px;">ดัชนีรสสัมผัสและโครงสร้าง (Sensory Index: 95%)</div>', unsafe_allow_html=True)
        st.markdown('<div class="bar-bg"><div class="bar-s"></div></div>', unsafe_allow_html=True)
        desc_text = "ไวน์มีโครงสร้างกลมกล่อมยอดเยี่ยม แอลกอฮอล์และกรดซิตริกมีความสมดุลสูง กรดระเหยต่ำทำให้ไม่มีกลิ่นเปรี้ยวฉุน ซัลเฟตช่วยรักษากลิ่นหอมและรสชาติได้อย่างสมบูรณ์แบบ"
    elif "B-Rank" in prediction_result:
        st.markdown('<div class="badge-b">B-Rank: ไวน์คุณภาพปานกลาง</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.85rem; color:#94A3B8; margin-bottom:4px;">ดัชนีรสสัมผัสและโครงสร้าง (Sensory Index: 65%)</div>', unsafe_allow_html=True)
        st.markdown('<div class="bar-bg"><div class="bar-b"></div></div>', unsafe_allow_html=True)
        desc_text = "ไวน์มาตรฐาน ดื่มได้ตามเกณฑ์ทั่วไป สัดส่วนกรดและแอลกอฮอล์อยู่ในเกณฑ์ปกติ ไม่มีข้อบกพร่องทางรสชาติที่เด่นชัด เหมาะสำหรับการบริโภคทั่วไป"
    else:
        st.markdown('<div class="badge-d">D-Rank: ไวน์คุณภาพต่ำ (เสี่ยงต่อรสเปรี้ยวเสีย)</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.85rem; color:#94A3B8; margin-bottom:4px;">ดัชนีรสสัมผัสและโครงสร้าง (Sensory Index: 25%)</div>', unsafe_allow_html=True)
        st.markdown('<div class="bar-bg"><div class="bar-d"></div></div>', unsafe_allow_html=True)
        desc_text = "ไวน์มีกรดระเหย (Volatile Acidity) ในปริมาณสูงเกินเกณฑ์ ทำให้เกิดกลิ่นกรดอะซิติกคล้ายน้ำส้มสายชู สารกันเสียหรือแอลกอฮอล์ไม่สมดุล ส่งผลให้คุณภาพรสชาติด้อยลงอย่างเห็นได้ชัด"

    st.image(img_dict[prediction_result], use_container_width=True)
    st.markdown(f'<div class="analysis-text" style="margin-top:14px;"><strong>บทสรุปคุณลักษณะ:</strong> {desc_text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_view_right:
    st.markdown('<div class="tactical-card">', unsafe_allow_html=True)
    st.markdown('<div class="tactical-header">การวิเคราะห์ค่าสารประกอบเทียบเกณฑ์ไวน์คุณภาพสูง</div>', unsafe_allow_html=True)
    st.caption("เปรียบเทียบค่าที่จำลองกับเกณฑ์เฉลี่ยของไวน์ระดับคุณภาพดีเยี่ยม (S-Rank Benchmark)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            label="ปริมาณแอลกอฮอล์ (Alcohol)",
            value=f"{val_alcohol:.1f}%",
            delta=f"{val_alcohol - 11.52:+.2f}% เกณฑ์มาตรฐาน (11.5%)",
            delta_color="normal"
        )
        st.metric(
            label="ปริมาณซัลเฟต (Sulphates)",
            value=f"{val_sulph:.2f} g/dm³",
            delta=f"{val_sulph - 0.74:+.2f} เกณฑ์มาตรฐาน (0.74)",
            delta_color="normal"
        )
    with c2:
        st.metric(
            label="กรดระเหย (Volatile Acidity)",
            value=f"{val_va:.2f} g/dm³",
            delta=f"{val_va - 0.41:+.2f} เกณฑ์มาตรฐาน (0.41)",
            delta_color="inverse"
        )
        st.metric(
            label="กรดซิตริก (Citric Acid)",
            value=f"{val_citric:.2f} g/dm³",
            delta=f"{val_citric - 0.38:+.2f} เกณฑ์มาตรฐาน (0.38)",
            delta_color="normal"
        )

    st.markdown("""
    <div style="margin-top:20px; font-size:0.85rem; color:#94A3B8; line-height:1.6; border-top:1px solid #1E3824; padding-top:12px;">
        <strong style="color:#F1F5F9;">หลักการวิเคราะห์ทางเคมี:</strong><br>
        - <strong>Alcohol:</strong> ให้ความอบอุ่น รสสัมผัส (Body) และความหวานละมุนตามธรรมชาติ<br>
        - <strong>Volatile Acidity:</strong> ต้องมีค่าน้อยที่สุด หากสูงจะทำให้ไวน์เปรี้ยวฉุนคล้ายน้ำส้มสายชู<br>
        - <strong>Sulphates:</strong> สารต้านอนุมูลอิสระ ป้องกันแบคทีเรียและรักษาความสดของไวน์<br>
        - <strong>Citric Acid:</strong> เสริมความสดชื่นและความมีชีวิตชีวาของรสชาติ
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# กล่องข้อมูลตัวอย่างพับเก็บได้ด้านล่างสุด
with st.expander("ดูตารางข้อมูลตัวอย่างและสถิติชุดข้อมูล (Dataset & Statistics)"):
    st.write("ตัวอย่างข้อมูล 10 รายการแรกจากชุดข้อมูลไวน์แดง:")
    cols_display = ["alcohol", "volatile acidity", "sulphates", "citric acid", "quality", "เกรดคุณภาพ"]
    st.dataframe(df_wine[cols_display].head(10), use_container_width=True)
    
    st.write("การกระจายตัวของจำนวนตัวอย่างตามเกรดคุณภาพ:")
    st.bar_chart(df_wine["เกรดคุณภาพ"].value_counts())
