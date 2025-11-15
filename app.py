import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import plotly.graph_objects as go
import plotly.express as px

# 页面配置
st.set_page_config(
    page_title="KNN脓毒症预后预测系统",
    page_icon="🏥",
    layout="wide"
)

# 标题和说明
st.title("🏥 KNN脓毒症患者预后预测系统")
st.markdown("---")

# 侧边栏 - 模型配置
st.sidebar.header("⚙️ 模型配置")
k_value = st.sidebar.slider("K值设置", min_value=1, max_value=20, value=5, 
                             help="KNN算法的邻居数量")
cv_folds = st.sidebar.slider("交叉验证折数", min_value=5, max_value=10, value=10)

# 创建标签页
tab1, tab2, tab3 = st.tabs(["📊 数据上传与模型训练", "🔮 单个预测", "📈 模型性能"])

# ==================== 标签页1: 数据上传与训练 ====================
with tab1:
    st.header("数据上传与模型训练")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1️⃣ 上传训练数据")
        uploaded_file = st.file_uploader(
            "上传CSV或Excel文件（需包含以下变量）",
            type=['csv', 'xlsx', 'xls'],
            help="数据需包含: RR, SB, AG, SBP, Urea, Age, Temp, HR, Status"
        )
        
        # 显示数据格式示例
        with st.expander("📋 查看数据格式要求"):
            example_df = pd.DataFrame({
                'RR': [20, 22, 18],
                'SB': [95, 88, 92],
                'AG': [15, 18, 12],
                'SBP': [110, 95, 120],
                'Urea': [8.5, 12.3, 7.2],
                'Age': [65, 72, 58],
                'Temp': [38.5, 39.2, 37.8],
                'HR': [95, 105, 88],
                'Status': [0, 1, 0]
            })
            st.dataframe(example_df, use_container_width=True)
            st.caption("Status: 0=存活, 1=死亡")
    
    with col2:
        st.subheader("变量说明")
        st.markdown("""
        **预测变量:**
        - Age: 年龄
        - HR: 心率
        - RR: 呼吸频率
        - SBP: 收缩压
        - Temp: 体温
        - SB: 标准碳酸氢盐
        - AG: 阴离子间隙
        - Urea: 尿素
        
        **结局变量:**
        - Status
          (0=存活, 1=死亡)
        """)
    
    # 处理上传的文件
    if uploaded_file is not None:
        try:
            # 读取数据
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ 成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
            
            # 显示数据预览
            st.subheader("2️⃣ 数据预览")
            st.dataframe(df.head(10), use_container_width=True)
            
            # 定义变量
            predictor_vars = ['RR', 'SB', 'AG', 'SBP', 'Urea', 'Age', 'Temp', 'HR']
            outcome_var = 'Status'
            
            # 检查必需列
            missing_cols = [col for col in predictor_vars + [outcome_var] if col not in df.columns]
            if missing_cols:
                st.error(f"❌ 数据缺失以下列: {', '.join(missing_cols)}")
            else:
                # 准备数据
                X = df[predictor_vars].values
                y = df[outcome_var].values
                
                # 数据统计
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("总样本数", len(y))
                col2.metric("存活患者", np.sum(y == 0))
                col3.metric("死亡患者", np.sum(y == 1))
                col4.metric("死亡率", f"{np.mean(y)*100:.1f}%")
                
                # 训练模型按钮
                st.subheader("3️⃣ 训练模型")
                if st.button("🚀 开始训练KNN模型", type="primary"):
                    with st.spinner("模型训练中，请稍候..."):
                        # 标准化
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # 训练KNN模型
                        knn_model = KNeighborsClassifier(n_neighbors=k_value)
                        knn_model.fit(X_scaled, y)
                        
                        # 交叉验证评估
                        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        
                        # 计算各项指标
                        y_pred = knn_model.predict(X_scaled)
                        y_proba = knn_model.predict_proba(X_scaled)[:, 1]
                        
                        # 计算交叉验证得分
                        cv_scores = cross_val_score(knn_model, X_scaled, y, cv=cv, scoring='roc_auc')
                        
                        # 保存到session state
                        st.session_state['model'] = knn_model
                        st.session_state['scaler'] = scaler
                        st.session_state['X'] = X
                        st.session_state['y'] = y
                        st.session_state['X_scaled'] = X_scaled
                        st.session_state['y_pred'] = y_pred
                        st.session_state['y_proba'] = y_proba
                        st.session_state['cv_scores'] = cv_scores
                        st.session_state['predictor_vars'] = predictor_vars
                        
                    st.success("✅ 模型训练完成！请切换到其他标签页查看结果")
                    st.balloons()
                    
        except Exception as e:
            st.error(f"❌ 数据加载失败: {str(e)}")

# ==================== 标签页2: 单个预测 ====================
with tab2:
    st.header("单个患者预测")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ 请先在【数据上传与模型训练】标签页训练模型")
    else:
        st.success("✅ 模型已加载，可以进行预测")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("输入患者数据")
            rr = st.number_input("呼吸频率 (RR)", min_value=0.0, max_value=60.0, value=20.0, step=1.0)
            sb = st.number_input("碱剩余 (SB)", min_value=-30.0, max_value=30.0, value=0.0, step=0.5)
            ag = st.number_input("阴离子间隙 (AG)", min_value=0.0, max_value=40.0, value=12.0, step=1.0)
            sbp = st.number_input("收缩压 (SBP)", min_value=50.0, max_value=250.0, value=120.0, step=5.0)
        
        with col2:
            st.write("")  # 对齐
            st.write("")
            urea = st.number_input("尿素 (Urea)", min_value=0.0, max_value=50.0, value=7.0, step=0.5)
            age = st.number_input("年龄 (Age)", min_value=18, max_value=120, value=65, step=1)
            temp = st.number_input("体温 (Temp)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
            hr = st.number_input("心率 (HR)", min_value=40, max_value=200, value=80, step=5)
        
        if st.button("🔮 开始预测", type="primary"):
            # 准备输入数据
            input_data = np.array([[rr, sb, ag, sbp, urea, age, temp, hr]])
            input_scaled = st.session_state['scaler'].transform(input_data)
            
            # 预测
            prediction = st.session_state['model'].predict(input_scaled)[0]
            proba = st.session_state['model'].predict_proba(input_scaled)[0]
            
            # 显示结果
            st.markdown("---")
            st.subheader("📊 预测结果")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 0:
                    st.success("### ✅ 预测: 存活")
                else:
                    st.error("### ⚠️ 预测: 死亡")
            
            with col2:
                st.metric("存活概率", f"{proba[0]*100:.1f}%")
            
            with col3:
                st.metric("死亡概率", f"{proba[1]*100:.1f}%")
            
            # 概率柱状图
            fig = go.Figure(data=[
                go.Bar(x=['存活', '死亡'], y=[proba[0]*100, proba[1]*100],
                       marker_color=['#28a745', '#dc3545'])
            ])
            fig.update_layout(
                title="预测概率分布",
                yaxis_title="概率 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================== 标签页3: 模型性能 ====================
with tab3:
    st.header("模型性能评估")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ 请先在【数据上传与模型训练】标签页训练模型")
    else:
        # 计算评估指标
        y_true = st.session_state['y']
        y_pred = st.session_state['y_pred']
        y_proba = st.session_state['y_proba']
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        
        # 显示指标
        st.subheader("📊 核心性能指标")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("准确率", f"{accuracy:.3f}")
        col2.metric("敏感度", f"{recall:.3f}")
        col3.metric("特异度", f"{specificity:.3f}")
        col4.metric("AUC", f"{auc:.3f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("精确率", f"{precision:.3f}")
        col2.metric("F1值", f"{f1:.3f}")
        col3.metric(f"{cv_folds}折CV AUC", f"{np.mean(st.session_state['cv_scores']):.3f}")
        
        # ROC曲线
        st.subheader("📈 ROC曲线")
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                     name=f'KNN (AUC={auc:.3f})',
                                     line=dict(color='#0072B2', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     name='随机分类器',
                                     line=dict(color='gray', width=2, dash='dash')))
        fig_roc.update_layout(
            title=f"ROC曲线 ({cv_folds}折交叉验证)",
            xaxis_title="1 - 特异度",
            yaxis_title="敏感度",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # 混淆矩阵
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔢 混淆矩阵")
            cm_df = pd.DataFrame(cm, 
                                index=['实际存活', '实际死亡'],
                                columns=['预测存活', '预测死亡'])
            st.dataframe(cm_df, use_container_width=True)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['预测存活', '预测死亡'],
                y=['实际存活', '实际死亡'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20}
            ))
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.subheader("📋 详细指标表")
            metrics_df = pd.DataFrame({
                '指标': ['准确率', '敏感度', '特异度', '精确率', 'F1值', 'AUC'],
                '数值': [f"{accuracy:.3f}", f"{recall:.3f}", f"{specificity:.3f}",
                        f"{precision:.3f}", f"{f1:.3f}", f"{auc:.3f}"]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # 下载按钮
            csv = metrics_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载性能指标",
                data=csv,
                file_name="KNN_Performance_Metrics.csv",
                mime="text/csv"
            )

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>KNN脓毒症预后预测系统 v1.0 | 大连医科大学</p>
</div>
""", unsafe_allow_html=True)