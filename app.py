import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="PPE安全穿戴检测系统",
    page_icon="🪖",
    layout="wide"
)

st.title("🪖 PPE安全穿戴智能检测系统")
st.markdown("检测安全帽、口罩、反光衣佩戴情况")


# 加载模型（使用缓存，只加载一次）
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # 模型文件在项目根目录


model = load_model()

# 侧边栏配置
st.sidebar.header("检测设置")
conf_threshold = st.sidebar.slider("置信度阈值", 0.25, 0.9, 0.5)
source_type = st.sidebar.radio("输入方式", ["📷 图片上传", "🎥 摄像头"])

# 类别名称映射
class_names = {
    0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask",
    4: "NO-Safety Vest", 5: "Person", 6: "Safety Cone",
    7: "Safety Vest", 8: "machinery", 9: "vehicle"
}

# 违规类别
violation_classes = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]

if source_type == "📷 图片上传":
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # 读取图片
        image = Image.open(uploaded_file)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 检测
        results = model(img_cv, conf=conf_threshold)[0]
        annotated = results.plot()

        # 显示结果
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="原图")
        with col2:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="检测结果")

        # 统计违规
        violations = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, "Unknown")
            if cls_name in violation_classes:
                violations.append(cls_name)

        st.subheader("检测结果")
        if violations:
            st.error(f"⚠️ 违规：{', '.join(violations)}")
        else:
            st.success("✅ 所有人员穿戴合规")

        # 显示检测统计
        if results.boxes is not None:
            st.info(f"共检测到 {len(results.boxes)} 个目标")

else:  # 摄像头模式
    st.write("使用摄像头实时检测")
    camera_image = st.camera_input("拍照检测")
    if camera_image:
        image = Image.open(camera_image)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model(img_cv, conf=conf_threshold)[0]
        annotated = results.plot()
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="检测结果")