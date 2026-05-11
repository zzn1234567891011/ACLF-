# ==================== 导入库 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（用于绘图）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 加载数据 ====================
df = pd.read_excel('数据简洁版.xlsx')   # 请确认文件路径正确

# 特征列（请根据实际列名修改）
feature_cols = ['TBIL', 'INR', 'Cr', '白细胞计数','谷丙转氨酶','高密度脂蛋白']   # 使用您选择的四个变量
# 如果还有其他变量，请添加到列表中

X = df[feature_cols].copy()
y = df['ACLF'].copy()

# 删除缺失值（若缺失不多；否则可用均值填充）
X = X.dropna()
y = y[X.index]
print(f"数据量: {X.shape[0]}, 阳性比例: {y.mean():.2f}")

# ==================== 2. 设置交叉验证 ====================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==================== 3. 定义 XGBoost 模型（使用最佳参数） ====================
# 类别权重（负样本数/正样本数）
scale_pos = (len(y) - y.sum()) / y.sum()

best_params = {
    'colsample_bytree': 0.6,
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 50,
    'subsample': 1.0
}

xgb_model = xgb.XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos   # 处理类别不平衡
)

# ==================== 4. 交叉验证预测 ====================
print("\n正在进行交叉验证预测...")
y_pred_proba = cross_val_predict(xgb_model, X, y, cv=cv, method='predict_proba')[:, 1]

# ==================== 5. 模型评估 ====================
auc = roc_auc_score(y, y_pred_proba)
print(f"\n交叉验证 AUC = {auc:.4f}")

# 计算最佳阈值（Youden指数）
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
youden = tpr - fpr
best_idx = np.argmax(youden)
best_threshold = thresholds[best_idx]
print(f"\n最佳阈值 (Youden指数): {best_threshold:.4f}")
print(f"此时敏感度 = {tpr[best_idx]:.4f}, 特异度 = {1 - fpr[best_idx]:.4f}")

# 根据最佳阈值得到预测类别
y_pred = (y_pred_proba >= best_threshold).astype(int)

# 分类报告
print("\n分类报告:")
print(classification_report(y, y_pred, target_names=['未患病', '患病']))

# 混淆矩阵
cm = confusion_matrix(y, y_pred)
print("\n混淆矩阵:")
print(cm)

# ==================== 6. 绘制 ROC 曲线 ====================
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC 曲线 (XGBoost)')
plt.legend()
plt.tight_layout()
plt.show()

# ==================== 7. 特征重要性 ====================
# 由于 cross_val_predict 未保存模型，我们重新在全部数据上训练以获取特征重要性
xgb_model.fit(X, y)
importance = xgb_model.feature_importances_
imp_df = pd.DataFrame({'特征': feature_cols, '重要性': importance})
imp_df = imp_df.sort_values('重要性', ascending=False)

import joblib
joblib.dump(xgb_model, 'xgboost_model.pkl')
print("模型已保存为 xgboost_model.pkl")

print("\n特征重要性排序:")
print(imp_df)

plt.figure(figsize=(6, 4))
plt.barh(imp_df['特征'], imp_df['重要性'], color='steelblue')
plt.xlabel('重要性')
plt.title('XGBoost 特征重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 绘制敏感度-特异度随阈值变化的曲线
thresholds, tpr, fpr = roc_curve(y, y_pred_proba)[2], roc_curve(y, y_pred_proba)[1], roc_curve(y, y_pred_proba)[0]
plt.figure(figsize=(8,5))
plt.plot(thresholds, tpr, label='敏感度')
plt.plot(thresholds, 1-fpr, label='特异度')
plt.xlabel('阈值')
plt.ylabel('比率')
plt.title('不同阈值下的敏感度与特异度')
plt.legend()
plt.grid(True)
plt.show()

# 找到敏感度 >= 0.8 的索引（取第一个满足条件的，即最接近 0.8 且不小于 0.8）
target_sens = 0.8
idx = np.where(tpr >= target_sens)[0]
if len(idx) > 0:
    idx = idx[0]
    threshold = thresholds[idx]
    actual_sens = tpr[idx]
    actual_spec = 1 - fpr[idx]
    print(f"敏感度 ≥ {target_sens} 对应的阈值: {threshold:.4f}")
    print(f"实际敏感度: {actual_sens:.4f}")
    print(f"实际特异度: {actual_spec:.4f}")
else:
    # 如果所有 tpr 都小于 0.8，则取最大值
    idx = np.argmax(tpr)
    threshold = thresholds[idx]
    actual_sens = tpr[idx]
    actual_spec = 1 - fpr[idx]
    print(f"无法达到敏感度 {target_sens}，最大敏感度为 {actual_sens:.4f}")
    print(f"对应阈值: {threshold:.4f}, 特异度: {actual_spec:.4f}")