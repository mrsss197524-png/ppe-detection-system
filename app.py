"""
PPE安全穿戴智能检测系统 - Streamlit Web完整版
保留桌面版全部功能：类别筛选、统计、日志、告警冷却、违规截图
"""
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from collections import defaultdict

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="PPE安全穿戴检测系统",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    /* 主背景 */
    .stApp {
        background-color: #1e1e2e;
    }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background-color: #181825;
    }
    
    /* 卡片样式 */
    .stat-card {
        background-color: #313244;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #45475a;
    }
    .stat-card .value {
        font-size: 36px;
        font-weight: bold;
        color: #89b4fa;
    }
    .stat-card .label {
        font-size: 14px;
        color: #a6adc8;
        margin-top: 8px;
    }
    
    /* 告警样式 */
    .alert-normal {
        background-color: #a6e3a1;
        color: #1e1e2e;
        padding: 16px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .alert-warning {
        background-color: #f38ba8;
        color: #1e1e2e;
        padding: 16px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* 违规详情卡片 */
    .violation-card {
        background-color: #313244;
        border-left: 4px solid #f38ba8;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .violation-card .class-name {
        font-weight: bold;
        color: #f38ba8;
        font-size: 16px;
    }
    .violation-card .confidence {
        color: #a6adc8;
        font-size: 14px;
    }
    
    /* 日志区域 */
    .log-container {
        background-color: #11111b;
        border: 1px solid #45475a;
        border-radius: 8px;
        padding: 12px;
        max-height: 200px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #a6e3a1;
    }
    
    /* 按钮 */
    .stButton > button {
        background-color: #89b4fa !important;
        color: #1e1e2e !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #b4befe !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State 初始化 ====================
def init_session_state():
    """初始化所有会话状态"""
    defaults = {
        'detector': None,
        'total_detections': 0,
        'violation_count': 0,
        'log_messages': [],
        'violation_history': [],
        'alert_cooldown': {},  # {class_name: last_alert_time}
        'alert_interval': 5,  # 冷却时间(秒)
        'current_frame': None,
        'is_detecting': False,
        # 类别筛选
        'enable_hardhat': True,
        'enable_vest': True,
        'enable_mask': True,
        # 告警配置
        'alert_enabled': True,
        'save_violations': True,
        # 模型信息
        'model_loaded': False,
        'model_info': {},
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== 工具函数 ====================
def log(msg):
    """添加日志消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{timestamp}] {msg}")
    if len(st.session_state.log_messages) > 200:
        st.session_state.log_messages = st.session_state.log_messages[-100:]

def get_timestamp():
    """获取可读时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def draw_boxes_custom(image, results, class_names, violation_classes):
    """
    自定义绘制检测框（带中文标签）
    违规类别用红色框，正常类别用绿色框
    """
    if results.boxes is None:
        return image
    
    img = image.copy()
    boxes = results.boxes
    
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = boxes.conf[i].item()
        cls_name = class_names.get(cls_id, f"Class_{cls_id}")
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        
        # 判断是否违规
        is_violation = cls_name in violation_classes
        
        # 颜色设置
        if is_violation:
            color = (243, 139, 168)  # 红色 BGR
        else:
            color = (166, 227, 161)  # 绿色 BGR
        
        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 标签
        label = f"{cls_name} {conf:.2f}"
        
        # 背景框
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        
        # 文字
        cv2.putText(img, label, (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 46), 2)
    
    return img

# ==================== 检测器类 ====================
@st.cache_resource(show_spinner=False)
def load_model(model_path, confidence):
    """加载YOLO模型（缓存）"""
    model = YOLO(model_path)
    model.conf = confidence
    return model

class PPEDetector:
    """PPE检测器 - 完整版"""
    
    def __init__(self, model_path, confidence=0.5):
        self.model = load_model(model_path, confidence)
        self.confidence = confidence
        self.class_names = self.model.names
        
        # 类别映射
        self.class_mapping = self._build_class_mapping()
        self.enabled_class_ids = set(self.class_mapping.values())
        
        # 违规类别定义
        self.violation_classes = {
            'NO-Hardhat': {'color': '#f38ba8', 'icon': '🪖', 'label': '未戴安全帽'},
            'NO-Mask': {'color': '#fab387', 'icon': '😷', 'label': '未戴口罩'},
            'NO-Safety Vest': {'color': '#f9e2af', 'icon': '🦺', 'label': '未穿反光衣'}
        }
        
        # 中文标签
        self.class_labels = {
            'Hardhat': '安全帽 ✅',
            'Mask': '口罩 ✅',
            'Safety Vest': '反光衣 ✅',
            'NO-Hardhat': '未戴安全帽 ❌',
            'NO-Mask': '未戴口罩 ❌',
            'NO-Safety Vest': '未穿反光衣 ❌',
            'Person': '人员',
            'Safety Cone': '安全锥',
            'machinery': '机械',
            'vehicle': '车辆'
        }
        
        # 统计
        self.total_detections = 0
        self.violation_count = 0
    
    def _build_class_mapping(self):
        """构建类别映射"""
        mapping = {}
        for class_id, class_name in self.class_names.items():
            name_lower = class_name.lower()
            if 'hardhat' in name_lower and 'no' not in name_lower:
                mapping['hardhat'] = class_id
            elif 'vest' in name_lower and 'no' not in name_lower and 'safety' in name_lower:
                mapping['vest'] = class_id
            elif 'mask' in name_lower and 'no' not in name_lower:
                mapping['mask'] = class_id
            elif 'no-hardhat' in name_lower:
                mapping['no_hardhat'] = class_id
            elif 'no-mask' in name_lower:
                mapping['no_mask'] = class_id
            elif 'no-safety' in name_lower or 'no-vest' in name_lower:
                mapping['no_vest'] = class_id
            elif 'person' in name_lower:
                mapping['person'] = class_id
            elif 'cone' in name_lower:
                mapping['cone'] = class_id
            elif 'machinery' in name_lower:
                mapping['machinery'] = class_id
            elif 'vehicle' in name_lower:
                mapping['vehicle'] = class_id
            elif 'safety' in name_lower and 'vest' in name_lower:
                mapping['vest'] = class_id
            else:
                mapping[name_lower] = class_id
        
        return mapping
    
    def set_enabled_classes(self, hardhat=True, vest=True, mask=True):
        """设置启用的检测类别"""
        self.enabled_class_ids.clear()
        
        if hardhat:
            if 'hardhat' in self.class_mapping:
                self.enabled_class_ids.add(self.class_mapping['hardhat'])
            if 'no_hardhat' in self.class_mapping:
                self.enabled_class_ids.add(self.class_mapping['no_hardhat'])
        
        if vest:
            if 'vest' in self.class_mapping:
                self.enabled_class_ids.add(self.class_mapping['vest'])
            if 'no_vest' in self.class_mapping:
                self.enabled_class_ids.add(self.class_mapping['no_vest'])
        
        if mask:
            if 'mask' in self.class_mapping:
                self.enabled_class_ids.add(self.class_mapping['mask'])
            if 'no_mask' in self.class_mapping:
                self.enabled_class_ids.add(self.class_mapping['no_mask'])
        
        # 始终包含 person 类别（作为辅助检测）
        if 'person' in self.class_mapping:
            self.enabled_class_ids.add(self.class_mapping['person'])
    
    def set_confidence(self, confidence):
        """设置置信度阈值"""
        self.confidence = confidence
    
    def detect(self, image):
        """执行检测"""
        results = self.model(image, conf=self.confidence)[0]
        
        # 过滤类别
        filtered = self._filter_results(results)
        
        # 绘图
        annotated = self._annotate_image(image, filtered)
        
        # 分析违规
        violations = self._analyze_violations(filtered)
        
        # 更新统计
        if filtered.boxes is not None:
            self.total_detections += len(filtered.boxes)
            self.violation_count += len(violations)
            st.session_state.total_detections = self.total_detections
            st.session_state.violation_count = self.violation_count
        
        return filtered, annotated, violations
    
    def _filter_results(self, results):
        """过滤检测结果"""
        if results.boxes is None:
            return results
        
        keep_idx = []
        for i, cls_id in enumerate(results.boxes.cls):
            if int(cls_id.item()) in self.enabled_class_ids:
                keep_idx.append(i)
        
        if len(keep_idx) == 0:
            results.boxes = None
        else:
            results.boxes = results.boxes[keep_idx]
        
        return results
    
    def _annotate_image(self, image, results):
        """标注图像"""
        return draw_boxes_custom(
            image, results, 
            self.class_names, 
            list(self.violation_classes.keys())
        )
    
    def _analyze_violations(self, results):
        """分析违规"""
        violations = []
        if results.boxes is None:
            return violations
        
        for i in range(len(results.boxes)):
            cls_id = int(results.boxes.cls[i].item())
            conf = results.boxes.conf[i].item()
            cls_name = self.class_names.get(cls_id, f"Unknown_{cls_id}")
            
            if cls_name in self.violation_classes:
                violations.append({
                    'class': cls_name,
                    'confidence': conf,
                    'label': self.class_labels.get(cls_name, cls_name),
                    'icon': self.violation_classes[cls_name]['icon'],
                    'color': self.violation_classes[cls_name]['color'],
                    'time': datetime.now().strftime('%H:%M:%S')
                })
        
        return violations
    
    def get_violation_summary(self, violations):
        """获取违规摘要"""
        if not violations:
            return "✅ 所有人员穿戴合规"
        
        count = defaultdict(int)
        for v in violations:
            count[v['class']] += 1
        
        parts = []
        for cls_name, num in count.items():
            label = self.class_labels.get(cls_name, cls_name)
            icon = self.violation_classes.get(cls_name, {}).get('icon', '')
            parts.append(f"{icon} {label}: {num}人")
        
        return "⚠️ " + " | ".join(parts)
    
    def get_stats(self):
        """获取统计"""
        total = max(self.total_detections, 1)
        return {
            'total': self.total_detections,
            'violations': self.violation_count,
            'rate': self.violation_count / total
        }
    
    def get_class_names_list(self):
        """获取所有类别名称列表"""
        return [f"{cls_id}: {name}" for cls_id, name in self.class_names.items()]
    
    def reset_stats(self):
        """重置统计"""
        self.total_detections = 0
        self.violation_count = 0
        st.session_state.total_detections = 0
        st.session_state.violation_count = 0

# ==================== 告警管理器 ====================
class AlertManager:
    """告警管理器"""
    
    def __init__(self, enabled=True, cooldown=5):
        self.enabled = enabled
        self.cooldown = cooldown
    
    def check_and_alert(self, violations):
        """
        检查是否需要告警
        返回: (是否告警, 告警消息)
        """
        if not self.enabled or not violations:
            return False, ""
        
        current_time = datetime.now()
        trigger_violations = []
        
        for v in violations:
            cls_name = v['class']
            last_time = st.session_state.alert_cooldown.get(cls_name)
            
            if last_time is None:
                # 第一次检测到
                st.session_state.alert_cooldown[cls_name] = current_time
                trigger_violations.append(v)
            elif (current_time - last_time).seconds >= self.cooldown:
                # 超过冷却时间
                st.session_state.alert_cooldown[cls_name] = current_time
                trigger_violations.append(v)
        
        if not trigger_violations:
            return False, ""
        
        # 生成告警消息
        count = defaultdict(int)
        for v in trigger_violations:
            count[v['label']] += 1
        
        parts = []
        for label, num in count.items():
            parts.append(f"{label} x{num}")
        
        message = "⚠️ 违规告警: " + ", ".join(parts)
        
        # 记录到历史
        st.session_state.violation_history.append({
            'time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'violations': trigger_violations,
            'message': message
        })
        
        # 保持历史记录不超过100条
        if len(st.session_state.violation_history) > 100:
            st.session_state.violation_history = st.session_state.violation_history[-100:]
        
        return True, message

# ==================== 主界面 ====================
def main():
    st.title("🪖 PPE安全穿戴智能检测系统")
    st.markdown("*智能检测安全帽、口罩、反光衣穿戴情况*")
    
    # 检查模型文件
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ 模型文件 `{model_path}` 未找到！请将 `best.pt` 放在项目根目录。")
        return
    
    # ==================== 侧边栏 ====================
    with st.sidebar:
        st.header("⚙️ 检测设置")
        
        # 模型信息
        st.subheader("📦 模型信息")
        st.info(f"模型文件: `{model_path}`")
        
        # 置信度
        st.subheader("🎚 置信度阈值")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.25, max_value=0.9, value=0.5, step=0.05,
            help="越高越严格，误检越少但可能漏检"
        )
        
        # 类别筛选
        st.subheader("🔍 检测类别筛选")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            enable_hardhat = st.checkbox("🪖 安全帽", value=st.session_state.enable_hardhat)
            st.session_state.enable_hardhat = enable_hardhat
        with col_f2:
            enable_vest = st.checkbox("🦺 反光衣", value=st.session_state.enable_vest)
            st.session_state.enable_vest = enable_vest
        with col_f3:
            enable_mask = st.checkbox("😷 口罩", value=st.session_state.enable_mask)
            st.session_state.enable_mask = enable_mask
        
        # 告警设置
        st.subheader("🚨 告警设置")
        alert_enabled = st.checkbox("启用告警", value=st.session_state.alert_enabled)
        st.session_state.alert_enabled = alert_enabled
        
        alert_cooldown = st.number_input(
            "告警冷却时间(秒)",
            min_value=1, max_value=30, 
            value=st.session_state.alert_interval,
            help="同一类型违规的告警间隔"
        )
        st.session_state.alert_interval = alert_cooldown
        
        # 输入方式
        st.subheader("📥 输入方式")
        source_type = st.radio(
            "选择输入源",
            ["📷 图片上传", "🎥 摄像头拍照", "📁 批量图片"]
        )
        
        # 重置按钮
        st.markdown("---")
        if st.button("🔄 重置统计"):
            if st.session_state.detector:
                st.session_state.detector.reset_stats()
            st.session_state.alert_cooldown = {}
            st.session_state.violation_history = []
            st.session_state.log_messages = []
            log("统计已重置")
            st.rerun()
        
        # 关于
        st.markdown("---")
        st.caption("PPE安全穿戴智能检测系统 v1.0")
        st.caption("Powered by YOLOv8 + Streamlit")
    
    # ==================== 初始化检测器 ====================
    detector = PPEDetector(model_path, confidence)
    detector.set_enabled_classes(enable_hardhat, enable_vest, enable_mask)
    st.session_state.detector = detector
    
    # 初始化告警管理器
    alert_manager = AlertManager(alert_enabled, alert_cooldown)
    
    # 显示模型信息
    with st.expander("📋 模型详情", expanded=False):
        class_list = detector.get_class_names_list()
        st.write(f"**总类别数:** {len(class_list)}")
        
        cols = st.columns(3)
        for i, cls in enumerate(class_list):
            with cols[i % 3]:
                st.code(cls, language=None)
    
    # ==================== 主体区域 ====================
    
    # 状态概览卡片
    st.subheader("📊 实时统计")
    stats = detector.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="value">{stats['total']}</div>
            <div class="label">总检测数</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="value" style="color: {'#f38ba8' if stats['violations'] > 0 else '#a6e3a1'};">
                {stats['violations']}
            </div>
            <div class="label">违规次数</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        rate_color = '#f38ba8' if stats['rate'] > 0.3 else '#f9e2af' if stats['rate'] > 0 else '#a6e3a1'
        st.markdown(f"""
        <div class="stat-card">
            <div class="value" style="color: {rate_color};">{stats['rate']:.1%}</div>
            <div class="label">违规率</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        enabled_count = sum([enable_hardhat, enable_vest, enable_mask])
        st.markdown(f"""
        <div class="stat-card">
            <div class="value">{enabled_count}</div>
            <div class="label">启用类别</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== 图片上传模式 ====================
    if source_type == "📷 图片上传":
        st.subheader("📷 上传检测图片")
        uploaded_file = st.file_uploader(
            "选择图片", 
            type=["jpg", "jpeg", "png", "bmp"],
            help="支持 JPG、PNG、BMP 格式"
        )
        
        if uploaded_file:
            # 读取图片
            image = Image.open(uploaded_file)
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 检测
            with st.spinner("🔍 正在检测..."):
                results, annotated, violations = detector.detect(img_bgr)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # 显示结果
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption="📷 原图", use_container_width=True)
            with col_img2:
                st.image(annotated_rgb, caption="🔍 检测结果", use_container_width=True)
            
            # 告警检查
            has_alert, alert_msg = alert_manager.check_and_alert(violations)
            
            # 违规详情
            st.markdown("### 📋 检测详情")
            if violations:
                st.error(detector.get_violation_summary(violations))
                
                # 列出每个违规
                for i, v in enumerate(violations):
                    st.markdown(f"""
                    <div class="violation-card">
                        <span class="class-name">{v['icon']} {v['label']}</span><br>
                        <span class="confidence">置信度: {v['confidence']:.2%} | 时间: {v['time']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ 未检测到违规，所有人员穿戴合规")
            
            # 总检测数
            if results.boxes is not None:
                st.info(f"📊 共检测到 **{len(results.boxes)}** 个目标")
            
            log(f"图片检测完成 - {detector.get_violation_summary(violations)}")
    
    # ==================== 摄像头模式 ====================
    elif source_type == "🎥 摄像头拍照":
        st.subheader("🎥 摄像头实时拍照检测")
        st.info("点击下方按钮打开摄像头拍照")
        
        camera_photo = st.camera_input("📸 拍照检测")
        
        if camera_photo:
            image = Image.open(camera_photo)
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            with st.spinner("🔍 正在检测..."):
                results, annotated, violations = detector.detect(img_bgr)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_rgb, caption="🔍 检测结果", use_container_width=True)
            
            has_alert, alert_msg = alert_manager.check_and_alert(violations)
            
            if violations:
                st.error(detector.get_violation_summary(violations))
                for v in violations:
                    st.markdown(f"""
                    <div class="violation-card">
                        <span class="class-name">{v['icon']} {v['label']}</span><br>
                        <span class="confidence">置信度: {v['confidence']:.2%}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ 所有人员穿戴合规")
            
            if results.boxes is not None:
                st.info(f"📊 共检测到 **{len(results.boxes)}** 个目标")
            
            log(f"摄像头检测完成 - {detector.get_violation_summary(violations)}")
    
    # ==================== 批量图片模式 ====================
    elif source_type == "📁 批量图片":
        st.subheader("📁 批量图片检测")
        uploaded_files = st.file_uploader(
            "选择多张图片",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            help="可同时选择多张图片批量检测"
        )
        
        if uploaded_files:
            st.info(f"已选择 **{len(uploaded_files)}** 张图片")
            
            progress_bar = st.progress(0)
            batch_violations = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                results, annotated, violations = detector.detect(img_bgr)
                batch_violations.extend(violations)
                
                # 更新进度
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # 批量结果汇总
            st.markdown("### 📊 批量检测汇总")
            
            total_detected = sum(1 for _ in uploaded_files)
            
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("检测图片数", len(uploaded_files))
            with col_b2:
                st.metric("总违规数", len(batch_violations))
            with col_b3:
                avg_violations = len(batch_violations) / max(len(uploaded_files), 1)
                st.metric("平均违规/图", f"{avg_violations:.1f}")
            
            # 分类统计
            if batch_violations:
                violation_counts = defaultdict(int)
                for v in batch_violations:
                    violation_counts[v['label']] += 1
                
                st.write("**违规分布:**")
                for label, count in violation_counts.items():
                    st.write(f"- {label}: {count}次")
            
            log(f"批量检测完成 - {len(uploaded_files)}张图片, {len(batch_violations)}个违规")
    
    # ==================== 告警状态显示 ====================
    st.markdown("---")
    st.subheader("🚨 告警状态")
    
    has_recent_violation = False
    recent_violations = []
    current_time = datetime.now()
    
    for vh in st.session_state.violation_history[-5:]:
        try:
            vh_time = datetime.strptime(vh['time'], '%Y-%m-%d %H:%M:%S')
            if (current_time - vh_time).seconds < alert_cooldown * 3:
                has_recent_violation = True
                recent_violations.append(vh)
        except:
            pass
    
    if has_recent_violation:
        st.markdown('<div class="alert-warning">⚠️ 检测到违规！请检查人员穿戴情况</div>', 
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-normal">✅ 状态正常，所有人员穿戴合规</div>', 
                    unsafe_allow_html=True)
    
    # ==================== 违规历史 ====================
    with st.expander("📜 违规历史记录", expanded=False):
        if st.session_state.violation_history:
            df = pd.DataFrame([
                {
                    '时间': vh['time'],
                    '违规内容': vh['message'],
                    '违规数': len(vh['violations'])
                }
                for vh in st.session_state.violation_history[-20:]
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无违规记录")
    
    # ==================== 日志区域 ====================
    st.markdown("---")
    st.subheader("📝 运行日志")
    
    log_text = "\n".join(st.session_state.log_messages[-30:]) if st.session_state.log_messages else "暂无日志"
    st.markdown(f"""
    <div class="log-container">
        <pre>{log_text}</pre>
    </div>
    """, unsafe_allow_html=True)

# ==================== 运行 ====================
if __name__ == "__main__":
    main()
