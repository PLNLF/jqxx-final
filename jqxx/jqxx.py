import streamlit as st
import joblib
import numpy as np
import os
import re
import jieba
import pandas as pd
import json
from datetime import datetime

# 初始化设置
st.set_page_config(page_title="假新闻检测系统", page_icon="🔍", layout="wide")

# 启动验证
def check_launch():
    """显示启动信息"""
    st.title("🔍 专业假新闻检测系统")
    st.markdown("使用先进AI技术分析新闻真实性")
    st.divider()
    st.caption(f"系统版本: 3.0 | 更新时间: {datetime.now().strftime('%Y-%m-%d')}")

# 模型加载函数
@st.cache_resource
def load_models():
    """加载预训练模型组件"""
    models = {}
    model_dir = "models"
    
    try:
        models["tfidf"] = joblib.load(os.path.join(model_dir, "tfidf_vectorizer_enhanced.pkl"))
        models["classifier"] = joblib.load(os.path.join(model_dir, "enhanced_fake_news_model.pkl"))
        models["expected_dim"] = 204
        return models
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

# 文本预处理函数
def preprocess_text(text):
    """安全预处理文本"""
    if not text: return ""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    return " ".join(jieba.cut(text.strip()))

# 安全特征工程
def extract_text_features(text):
    """提取安全特征 - 无迭代问题"""
    return {
        # 基本特征
        'length': len(text),
        'sentences': text.count('。') + 1,
        
        # 可信度信号
        'reliable': 1.0 if '新华社' in text or '人民日报' in text or '官方' in text else 0.0,
        
        # 荒谬内容直接识别
        'is_absurd': 1.0 if any(phrase in text for phrase in 
                                ['月球爆炸', '太阳消失', '地球停转', '长生不老']) else 0.0,
        
        # 异常特征
        'urgent': 1.0 if '必看' in text or '紧急' in text or '速看' in text else 0.0,
        'exaggeration': 1.0 if '震惊' in text or '最牛' in text or '100%' in text else 0.0
    }

# 特征生成函数（修复版本）
def generate_features(text, models):
    """生成特征并确保安全类型"""
    processed_text = preprocess_text(text)
    
    # 关键修复：特征向量明确类型转换
    try:
        # TF-IDF特征
        tfidf_vector = models["tfidf"].transform([processed_text]).toarray()
        
        # 高级特征提取
        adv_features = extract_text_features(text)
        adv_vector = [float(adv_features[key]) for key in sorted(adv_features)]
        
        # 合并特征并确保维度
        features = np.concatenate([tfidf_vector[0], adv_vector])
        
        # 严格类型转换确保安全
        return features[:models["expected_dim"]].astype(np.float32).reshape(1, -1)
    
    except Exception as e:
        st.error(f"特征生成错误: {str(e)}")
        return np.zeros((1, models["expected_dim"]), dtype=np.float32)

# 专业荒谬内容识别系统
def absurd_content_detector(text):
    """直接识别荒谬内容 - 无需迭代"""
    absurd_phrases = ['月球爆炸', '太阳消失', '地球停转', '长生不老', '穿越古代']
    for phrase in absurd_phrases:
        if phrase in text:
            return True, f"检测到荒谬内容: {phrase}"
    return False, ""

# 置信度计算
def calculate_confidence(probabilities, text):
    """安全置信度计算"""
    real_prob, fake_prob = probabilities
    
    # 权威来源显著提升真实概率
    if '新华社' in text or '人民日报' in text:
        real_prob = min(real_prob * 1.7, 0.97)
    
    return real_prob, fake_prob

# 用户界面主函数
def main_application():
    # 初始化
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    
    # 侧边栏
    with st.sidebar:
        st.subheader("系统状态")
        models = load_models()
        st.success("✓ 模型加载完成")
        st.info(f"特征维度: {models['expected_dim']}")
        
        st.divider()
        st.subheader("使用指南")
        st.markdown("""
        1. 粘贴完整新闻内容
        2. 避免单句检测
        3. 权威来源提高可信度
        4. 夸张词汇可能导致误判
        """)
        
        st.divider()
        if st.button("报告分析错误", use_container_width=True):
            st.session_state.show_feedback = True

    # 主界面
    st.title("新闻真实性检测")
    news_text = st.text_area("新闻内容:", height=200, 
                            placeholder="粘贴新闻内容...",
                            help="支持中英文内容",
                            key="news_input")
    
    if st.button("检测真实性", type="primary", use_container_width=True):
        if not news_text.strip():
            st.warning("请输入新闻内容")
            return
            
        with st.spinner("深度分析中..."):
            try:
                # 步骤1: 荒谬内容直接识别
                is_absurd, reason = absurd_content_detector(news_text)
                if is_absurd:
                    st.error(f"⚠️ 虚假新闻 - {reason}")
                    st.progress(1.0, "虚假程度: 100%")
                    
                    with st.expander("详细分析报告", expanded=True):
                        st.markdown("### 内容分析")
                        st.warning(f"系统直接识别到关键荒谬短语: **{reason.split(':')[-1]}**")
                        st.markdown("### 建议验证")
                        st.info("此类内容通常缺乏科学依据，建议通过官方渠道核实")
                    return
                
                # 步骤2: 模型分析
                features = generate_features(news_text, models)
                prediction = models["classifier"].predict(features)[0]
                probabilities = models["classifier"].predict_proba(features)[0]
                real_prob, fake_prob = calculate_confidence(probabilities, news_text)
                
                # 归一化处理
                total = real_prob + fake_prob
                real_prob, fake_prob = real_prob/total, fake_prob/total
                
                # 步骤3: 显示结果
                st.subheader("检测结果")
                if prediction == 1:  # 假新闻
                    st.error(f"⚠️ 虚假新闻 (置信度: {fake_prob*100:.1f}%)")
                    st.progress(fake_prob, f"虚假程度: {fake_prob*100:.1f}%")
                else:  # 真新闻
                    st.success(f"✅ 真实新闻 (置信度: {real_prob*100:.1f}%)")
                    st.progress(real_prob, f"可信程度: {real_prob*100:.1f}%")
                
                # 步骤4: 详细报告
                with st.expander("📊 详细分析报告", expanded=True):
                    features = extract_text_features(news_text)
                    st.markdown("### 内容特征分析")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("内容长度", f"{features['length']} 字")
                    col2.metric("科学可信度", "低" if features['is_absurd'] else "高", 
                             delta=None if features['is_absurd'] else "+可靠")
                    
                    # 关键特征指标
                    st.markdown("#### 可信度指标")
                    if features['reliable']:
                        st.success("✅ 包含权威来源或官方声明")
                    else:
                        st.info("ℹ️ 未检测到权威来源引用")
                    
                    if features['urgent'] or features['exaggeration']:
                        st.warning("⚠️ 检测到可能夸大词汇")
                    
                # 用户反馈
                st.session_state.last_result = {
                    "text": news_text, 
                    "prediction": prediction,
                    "confidence": fake_prob if prediction == 1 else real_prob
                }
                
            except Exception as e:
                st.error(f"分析失败: {str(e)}")
                # 提供详细错误信息
                import traceback
                st.code(traceback.format_exc())

    # 用户反馈系统
    if st.session_state.get("show_feedback", False):
        st.divider()
        st.subheader("✍️ 报告分析错误")
        
        actual_label = st.radio("新闻实际类型:", ("真实", "虚假"))
        st.text_area("补充说明 (可选):", key="feedback_comment")
        
        if st.button("提交反馈"):
            # 保存反馈信息
            feedback_dir = "user_feedback"
            os.makedirs(feedback_dir, exist_ok=True)
            
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "text": st.session_state.get("last_result", {}).get("text", "")[:500],
                "reported_label": "real" if actual_label == "真实" else "fake",
                "comment": st.session_state.feedback_comment
            }
            
            try:
                with open(os.path.join(feedback_dir, "feedback.json"), "a") as f:
                    f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
                st.success("感谢您的反馈！系统将据此改进")
                st.session_state.show_feedback = False
            except Exception as e:
                st.error(f"反馈保存失败: {str(e)}")

# 主程序入口
if __name__ == "__main__":
    check_launch()
    main_application()
