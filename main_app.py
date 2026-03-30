import streamlit as st
import cv2
import numpy as np
import pandas as pd
import base64
import json
import requests
import time
from PIL import Image
import io
# --- 核心评分标准 (必须放在函数定义之前) ---
SYSTEM_PROMPT = """你是一位极其严格的全国大学生金相技能大赛评审专家。
请严格按照以下官方标准对金相图像进行评分(总分80)：

1. 组织正确性与清晰度 (40分):
   - 33~40分: 组织正确、极其清晰，晶界锐利无虚焦。
   - 21~32分: 组织正确、比较清晰。
   - 13~20分: 组织勉强可辨别，不够清晰。
   - 5~12分: 仅能辨别部分组织，很不清晰。
   - 0~4分: 几乎看不清组织。

2. 划痕情况 (20分):
   - 18~20分: 无低倍粗大划痕，高倍细划痕极少或没有。
   - 14~17分: 无低倍粗大划痕，高倍细划痕较少(1个视场可见)。
   - 10~13分: 低倍粗大划痕1条，或高倍细划痕较多(2~3个视场可见)。
   - 6~9分: 低倍粗大划痕2条，或高倍细划痕很多(4个视场可见)。
   - 0~5分: 低倍粗大划痕3条以上且存在交叉划痕。

3. 假象 (20分):
   - 15~20分: 基本没有水迹、污点、酸蚀坑等假象。
   - 9~14分: 存在少量干扰观察的假象。
   - 0~8分: 假象较多，严重影响组织真实性判断。

请严格按 JSON 格式输出，不要包含任何 Markdown 标记（如 ```json），直接输出纯 JSON 对象：
{
  "details": {
    "structure_clarity": {"score": 0, "reason": ""},
    "scratches": {"score": 0, "reason": ""},
    "artifacts": {"score": 0, "reason": ""}
  },
  "overall_critique": "综合评价"
}"""
# --- 页面配置 ---
st.set_page_config(page_title="金相图像 AI 专家评审系统", layout="wide")

# --- 核心算法 (保留你的原逻辑) ---
def preprocess_for_algorithm(image_np):
    """针对 Streamlit 传入的 numpy 数组进行预处理"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.bilateralFilter(enhanced, 5, 50, 50)

def calculate_clarity(image_matrix):
    if image_matrix is None: return 0.0
    return round(cv2.Laplacian(image_matrix, cv2.CV_64F).var(), 2)

def encode_image_from_bytes(image_bytes):
    """将上传的文件流转为 Base64"""
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"

def extract_json_robustly(text):
    if not text: return None
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1].replace('，', ',').replace('：', ':')
            return json.loads(json_str)
    except: return None
    return None

def analyze_image_minimax(image_bytes, clarity_score, api_key, group_id, api_url):
    image_url = encode_image_from_bytes(image_bytes)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    prompt = f"{SYSTEM_PROMPT}\n\n【算法参考数据】该图的拉普拉斯清晰度得分为: {clarity_score}。"
    
    payload = {
        "model": "abab6.5s-chat",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]}]
    }
    
    try:
        # 1. 先获取响应对象
        response = requests.post(f"{api_url}?GroupId={group_id}", headers=headers, json=payload, timeout=60)
        
        # 2. 打印原始响应状态和文本（用于调试）
        # 如果报错，通过这个能看到是 API 欠费了还是格式错了
        st.write(f"🔍 调试信息 - 状态码: {response.status_code}")
        
        # 3. 核心修正：先检查是不是真的 JSON
        try:
            res_data = response.json()
        except Exception as json_err:
            st.error(f"❌ 服务器返回的内容不是 JSON 格式！内容如下：\n{response.text}")
            return None

        # 4. 提取回复内容
        if 'choices' in res_data:
            reply_text = res_data['choices'][0]['message']['content']
            # 使用更强的提取函数
            return extract_json_robustly(reply_text)
        else:
            st.error(f"❌ API 返回异常数据结构: {res_data}")
            return None

    except Exception as e:
        st.error(f"❌ 网络或请求发生错误: {e}")
        return None
# --- UI 界面设计 ---

st.title("🔬 金相图像 AI 专家评审系统")
st.markdown("---")

# 1. 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("MiniMax API Key", type="password")
    group_id = st.text_input("Group ID")
    api_url = st.text_input("API URL", value="https://api.minimax.chat/v1/visual_chat")
    
    st.info("上传图片后点击下方按钮开始分析")
    run_button = st.button("🚀 开始批量分析", type="primary")

# 2. 文件上传
uploaded_files = st.file_uploader("选择金相图片 (支持多选)", type=['png', 'jpg', 'jpeg', 'tif'], accept_multiple_files=True)

if uploaded_files:
    st.write(f"已上传 **{len(uploaded_files)}** 张图片")
    
    if run_button:
        if not api_key or not group_id:
            st.warning("请先在侧边栏配置 API 密钥信息！")
        else:
            results = []
            progress_bar = st.progress(0)
            
            # 迭代处理
            for i, file in enumerate(uploaded_files):
                # 读取图片
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 算法增强与计算
                enhanced_img = preprocess_for_algorithm(img)
                clarity_val = calculate_clarity(enhanced_img)
                
                # UI 展示：实时显示处理进度和结果
                with st.expander(f"📷 正在分析: {file.name}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption="原图", use_column_width=True)
                    with col2:
                        # OpenCV 灰度图转为 Streamlit 可显示的 RGB
                        st.image(enhanced_img, caption="算法增强 (边缘强化)", use_column_width=True, channels="GRAY")
                    
                    # AI 评分
                    with st.spinner('AI 专家正在评审中...'):
                        data = analyze_image_minimax(file_bytes, clarity_val, api_key, group_id, api_url)
                    
                    if data:
                        details = data.get('details', {})
                        scores = {
                            "清晰度": details.get('structure_clarity', {}).get('score', 0),
                            "划痕": details.get('scratches', {}).get('score', 0),
                            "假象": details.get('artifacts', {}).get('score', 0),
                        }
                        total_score = sum(scores.values())
                        
                        # 显示分数统计
                        st.success(f"**总分：{total_score}** / 80")
                        st.json(data) # 展示详细评价
                        
                        results.append({
                            "文件名": file.name,
                            "总分": total_score,
                            "评语": data.get('overall_critique', ''),
                            "算法清晰度": clarity_val
                        })
                
                # 更新进度条
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # 3. 结果汇总与导出
            st.markdown("---")
            st.subheader("📊 分析报告汇总")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # 导出 Excel 按钮
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                label="📥 下载 Excel 完整报告",
                data=output.getvalue(),
                file_name=f"金相评审报告_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("请上方上传图片以开始分析。")
