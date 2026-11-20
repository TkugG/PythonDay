from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import os

# การตั้งค่าโครงสร้างหน้าจอแบบกว้างและซ่อน sidebar เริ่มต้น
st.set_page_config(
    page_title="Wine Quality What-If Simulator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# สไตล์ CSS เพื่อความเป็นระเบียบ สะอาดตา และเน้นความคลีนแบบมืออาชีพ
st.markdown("""
<style>
    /* ซ่อน Header และ Footer ส่วนเกินของ Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ระยะขอบและการจัดวาง */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* การ์ดแสดงผล */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* ป้ายระดับคุณภาพ */
    .status-badge-high {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 1.15rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    .status-badge-med {
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 1.15rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    .status-badge-low {
        background-color: #FFF1F2;
        color: #9F1239;
        border: 1px solid #FECDD3;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 1.15rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# โหลดข้อมูลพร้อมแคชเพื่อประสิทธิภาพ
@st.cache_data
def load_data():
    df = pd.read_csv("./data/winequality-red.csv", sep=";")
    def categorize(score):
        if score <= 5:
            return "คุณภาพทั่วไป"
        elif score == 6:
            return "คุณภาพปานกลาง"
        else:
            return "คุณภาพดีเยี่ยม"
    df["ระดับคุณภาพ"] = df["quality"].apply(categorize)
    return df

# เทรนโมเดล KNN ครั้งเดียวพร้อมแคช
@st.cache_resource
def train_model(df):
    features = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
    X = df[features]
    y = df["ระดับคุณภาพ"]
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model

dt = load_data()
model = train_model(dt)

# ค่าเฉลี่ยมาตรฐานของไวน์คุณภาพดีเยี่ยม (Benchmark สำหรับ What-If Analysis)
BENCHMARK = {
    "alcohol": 11.52,
    "volatile acidity": 0.41,
    "sulphates": 0.74,
    "citric acid": 0.38
}

# ส่วนหัวเว็บแอปพลิเคชัน
col_header_text, col_header_profile = st.columns([5, 1])
with col_header_text:
    st.title("ระบบจำลองสถานการณ์คุณภาพไวน์แดง")
    st.caption("What-If Simulation Dashboard วิเคราะห์ผลลัพธ์ทันทีเมื่อปรับเปลี่ยนคุณลักษณะทางเคมี")
with col_header_profile:
    if os.path.exists("./img/myprofile.png"):
        st.image("./img/myprofile.png", width=90)

st.divider()

# แหล่งที่อยู่ภาพตัวอย่างไวน์ภายนอก
img_dict = {
    "คุณภาพทั่วไป": "https://images.unsplash.com/photo-1558001373-7b93ee48ffa0?auto=format&fit=crop&w=700&q=80",
    "คุณภาพปานกลาง": "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?auto=format&fit=crop&w=700&q=80",
    "คุณภาพดีเยี่ยม": "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?auto=format&fit=crop&w=700&q=80"
}

# โครงสร้าง 2 คอลัมน์สำหรับ What-If Dashboard
col_inputs, col_results = st.columns([5, 6], gap="large")

# คอลัมน์ซ้าย: ตัวแปรจำลองสถานการณ์
with col_inputs:
    st.subheader("ตัวแปรจำลองสถานการณ์")
    st.caption("ปรับเลื่อนค่าคุณลักษณะเพื่อดูผลการประเมินแบบทันที")
    
    val_alcohol = st.slider(
        "ปริมาณแอลกอฮอล์ (Alcohol %)",
        min_value=8.4,
        max_value=14.9,
        value=10.4,
        step=0.1
    )
    
    val_va = st.slider(
        "กรดระเหย (Volatile Acidity: g/dm³)",
        min_value=0.12,
        max_value=1.58,
        value=0.52,
        step=0.01
    )
    
    val_sulph = st.slider(
        "ปริมาณซัลเฟต (Sulphates: g/dm³)",
        min_value=0.33,
        max_value=2.00,
        value=0.66,
        step=0.01
    )
    
    val_citric = st.slider(
        "กรดซิตริก (Citric Acid: g/dm³)",
        min_value=0.00,
        max_value=1.00,
        value=0.27,
        step=0.01
    )

# คำนวณผลลัพธ์การทำนายแบบ Real-time
features_list = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
user_scenario = pd.DataFrame(
    [[val_alcohol, val_va, val_sulph, val_citric]],
    columns=features_list
)
predicted_class = model.predict(user_scenario)[0]

# คอลัมน์ขวา: ผลการจำลอง What-If
with col_results:
    st.subheader("ผลการจำลองสถานการณ์")
    
    # ป้ายแสดงสถานะตามเกรด
    if predicted_class == "คุณภาพดีเยี่ยม":
        badge_html = f'<div class="status-badge-high">ระดับคุณภาพ: {predicted_class}</div>'
    elif predicted_class == "คุณภาพปานกลาง":
        badge_html = f'<div class="status-badge-med">ระดับคุณภาพ: {predicted_class}</div>'
    else:
        badge_html = f'<div class="status-badge-low">ระดับคุณภาพ: {predicted_class}</div>'
    st.markdown(badge_html, unsafe_allow_html=True)
    
    # แสดงรูปภาพไวน์ตามผลการทำนาย
    st.image(img_dict[predicted_class], use_container_width=True)
    
    st.write("การเปรียบเทียบค่าจำลองกับเกณฑ์ไวน์คุณภาพดีเยี่ยม:")
    
    # การ์ดตัวชี้วัดเปรียบเทียบกับ Benchmark
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        diff_alc = round(val_alcohol - BENCHMARK["alcohol"], 2)
        st.metric(
            label="Alcohol",
            value=f"{val_alcohol:.1f}%",
            delta=f"{diff_alc:+.2f}% เกณฑ์มาตรฐาน",
            delta_color="normal"
        )
    with col_m2:
        diff_va = round(val_va - BENCHMARK["volatile acidity"], 2)
        st.metric(
            label="Volatile Acidity",
            value=f"{val_va:.2f}",
            delta=f"{diff_va:+.2f} เกณฑ์มาตรฐาน",
            delta_color="inverse"
        )
        
    col_m3, col_m4 = st.columns(2)
    with col_m3:
        diff_sul = round(val_sulph - BENCHMARK["sulphates"], 2)
        st.metric(
            label="Sulphates",
            value=f"{val_sulph:.2f}",
            delta=f"{diff_sul:+.2f} เกณฑ์มาตรฐาน",
            delta_color="normal"
        )
    with col_m4:
        diff_cit = round(val_citric - BENCHMARK["citric acid"], 2)
        st.metric(
            label="Citric Acid",
            value=f"{val_citric:.2f}",
            delta=f"{diff_cit:+.2f} เกณฑ์มาตรฐาน",
            delta_color="normal"
        )

st.write("")
st.divider()

# กล่องพับเก็บได้สำหรับข้อมูลตัวอย่างและสถิติ (ไม่รกสายตา)
with st.expander("ข้อมูลตัวอย่างและสถิติชุดข้อมูล"):
    st.write("ตัวอย่างข้อมูล 10 แถวแรกจาก winequality-red.csv:")
    display_cols = ["alcohol", "volatile acidity", "sulphates", "citric acid", "quality", "ระดับคุณภาพ"]
    st.dataframe(dt[display_cols].head(10), use_container_width=True)
    
    st.write("จำนวนตัวอย่างในแต่ละระดับคุณภาพ:")
    st.bar_chart(dt["ระดับคุณภาพ"].value_counts())
