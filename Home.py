from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import os

# โครงสร้างหน้าเว็บและข้อมูลผู้จัดทำ
col_title, col_pic = st.columns([4, 1])
with col_title:
    st.header("โปรเจคการจำแนกคุณภาพไวน์แดง")
with col_pic:
    if os.path.exists("./img/myprofile.png"):
        st.image("./img/myprofile.png", width=110)

# แสดงประเภทคุณภาพไวน์ 3 ระดับ พร้อมรูปภาพตัวอย่าง
img_low = "https://images.unsplash.com/photo-1558001373-7b93ee48ffa0?auto=format&fit=crop&w=600&q=80"
img_med = "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?auto=format&fit=crop&w=600&q=80"
img_high = "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?auto=format&fit=crop&w=600&q=80"

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("คุณภาพทั่วไป")
    st.image(img_low)

with col2:
    st.subheader("คุณภาพปานกลาง")
    st.image(img_med)

with col3:
    st.subheader("คุณภาพดีเยี่ยม")
    st.image(img_high)

# ส่วนแสดงข้อมูลสถิติ
html_7 = """
<div style="background-color:#722F37;padding:12px;border-radius:10px;color:white;text-align:center;">
<h5 style="margin:0;">สถิติข้อมูลไวน์แดง</h5>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.write("")

# โหลดชุดข้อมูลไวน์แดง
dt = pd.read_csv("./data/winequality-red.csv", sep=";")

# ฟังก์ชันจัดกลุ่มคุณภาพไวน์เป็น 3 ระดับ
def categorize_quality(score):
    if score <= 5:
        return "คุณภาพทั่วไป"
    elif score == 6:
        return "คุณภาพปานกลาง"
    else:
        return "คุณภาพดีเยี่ยม"

dt["ระดับคุณภาพ"] = dt["quality"].apply(categorize_quality)

# แสดงตัวอย่างข้อมูล
selected_columns = ["alcohol", "volatile acidity", "sulphates", "citric acid", "quality", "ระดับคุณภาพ"]
st.write(dt[selected_columns].head(10))

# ส่วนการแสดงผลกราฟข้อมูล
if st.button("แสดงการจินตทัศน์ข้อมูล"):
    quality_counts = dt["ระดับคุณภาพ"].value_counts()
    st.write("จำนวนข้อมูลตามระดับคุณภาพ:")
    st.bar_chart(quality_counts)
else:
    st.write("ไม่แสดงข้อมูล")

# ส่วนทำนายผล
html_8 = """
<div style="background-color:#8B0000;padding:12px;border-radius:10px;color:white;text-align:center;">
<h5 style="margin:0;">ทำนายคุณภาพไวน์</h5>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.write("")

# อินพุตฟีเจอร์เด่น 4 ตัว
alc = st.slider("ปริมาณแอลกอฮอล์ (Alcohol)", min_value=8.0, max_value=15.0, value=10.4, step=0.1)
vol_acid = st.slider("กรดระเหย (Volatile Acidity)", min_value=0.10, max_value=1.60, value=0.52, step=0.01)
sulph = st.slider("ปริมาณซัลเฟต (Sulphates)", min_value=0.30, max_value=2.00, value=0.66, step=0.01)
cit_acid = st.slider("กรดซิตริก (Citric Acid)", min_value=0.00, max_value=1.00, value=0.27, step=0.01)

if st.button("ทำนายผล"):
    feature_cols = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
    X = dt[feature_cols]
    y = dt["ระดับคุณภาพ"]

    # เทรนโมเดล KNN อย่างง่าย
    Knn_model = KNeighborsClassifier(n_neighbors=5)
    Knn_model.fit(X, y)

    x_input = pd.DataFrame([[alc, vol_acid, sulph, cit_acid]], columns=feature_cols)
    out = Knn_model.predict(x_input)
    result = out[0]

    st.subheader(f"ผลการทำนาย: {result}")

    if result == "คุณภาพทั่วไป":
        st.image(img_low)
    elif result == "คุณภาพปานกลาง":
        st.image(img_med)
    else:
        st.image(img_high)
else:
    st.write("ไม่ทำนาย")
