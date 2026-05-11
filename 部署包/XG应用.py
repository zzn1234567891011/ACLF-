import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib

st.set_page_config(page_title="ACLF风险预测", layout="centered")
st.title("ACLF发病风险预测")
st.markdown("请输入以下指标，模型将输出风险等级（高风险/低风险）。")

# ==================== 加载已训练好的模型 ====================
@st.cache_resource
def load_model():
    model = joblib.load('部署包/xgboost_model.pkl')
    return model

model = load_model()

# 特征列（顺序必须与训练时一致）
feature_cols = ['TBIL', 'INR', 'Cr', '白细胞计数', '谷丙转氨酶', '高密度脂蛋白']

# ==================== 用户输入（无默认数值，指定小数位） ====================
col1, col2 = st.columns(2)

with col1:
    tb = st.number_input(
        "总胆红素 (TBIL, mg/dL)",
        min_value=0.0,
        step=0.1,
        format="%.1f",          # 保留1位小数
        value=None,             # 无默认数值
        placeholder="例如 5.0"
    )
    inr = st.number_input(
        "国际标准化比值 (INR)",
        min_value=0.0,
        step=0.01,
        format="%.2f",          # 保留2位小数
        value=None,
        placeholder="例如 1.50"
    )
    cr = st.number_input(
        "肌酐 (Cr, mg/dL)",
        min_value=0,
        step=1,
        format="%d",            # 整数，无小数
        value=None,
        placeholder="例如 1"
    )

with col2:
    wbc = st.number_input(
        "白细胞计数 (×10⁹/L)",
        min_value=0.0,
        step=0.01,
        format="%.2f",          # 保留2位小数
        value=None,
        placeholder="例如 7.00"
    )
    alt = st.number_input(
        "谷丙转氨酶 (ALT, U/L)",
        min_value=0,
        step=1,
        format="%d",            # 整数
        value=None,
        placeholder="例如 40"
    )
    hdl = st.number_input(
        "高密度脂蛋白 (HDL, mg/dL)",
        min_value=0.0,
        step=0.01,
        format="%.2f",          # 保留2位小数
        value=None,
        placeholder="例如 50.00"
    )

# ==================== 预测 ====================
threshold = 0.4052
if st.button("预测风险"):
    # 检查是否所有输入都已填写（注意：number_input 未填写时为 None）
    if None in [tb, inr, cr, wbc, alt, hdl]:
        st.error("请填写所有指标")
        st.stop()
    
    # 构建输入 DataFrame（顺序与训练一致）
    input_values = [tb, inr, cr, wbc, alt, hdl]
    input_df = pd.DataFrame([input_values], columns=feature_cols)
    
    prob = model.predict_proba(input_df)[0, 1]
    risk = "高风险" if prob >= threshold else "低风险"
    
    st.subheader("预测结果")
    st.write(f"风险等级：**{risk}**")
