import os
import base64
import json
import requests
import pandas as pd
import time
import cv2
import numpy as np
import re

# 尝试导入配置
try:
    from config import API_KEY, GROUP_ID, API_URL
except ImportError:
    print("❌ 错误：未找到 config.py 文件或其中缺少 API_KEY, GROUP_ID, API_URL")
    exit(1)

# --- 优化后的核心评分 Prompt (集成大赛官方标准) ---
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
def preprocess_for_algorithm(image_path):
    """专门为算法准备的增强：转灰度 + CLAHE 强化边缘"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自适应直方图均衡化 (限制对比度)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # 双边滤波去噪并保留边缘
    return cv2.bilateralFilter(enhanced, 5, 50, 50)

def calculate_clarity(image_matrix):
    """计算图像客观清晰度 (拉普拉斯方差法)"""
    if image_matrix is None:
        return 0.0
    return round(cv2.Laplacian(image_matrix, cv2.CV_64F).var(), 2)

def encode_image(image_path):
    """图片转 Base64"""
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

def extract_json_robustly(text):
    """强力 JSON 提取：处理截断、不可见字符及中文标点"""
    if not text: return None
    # 移除不可见字符
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")
    try:
        # 定位最外层的花括号
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            # 修正 AI 可能误用的中文标点
            json_str = json_str.replace('，', ',').replace('：', ':')
            return json.loads(json_str)
    except Exception as e:
        print(f"⚠️ JSON 解析修正失败: {e}")
    return None

def analyze_image_minimax(image_path, clarity_score):
    """调用 MiniMax 视觉模型 (传入彩色原图)"""
    try:
        image_url = encode_image(image_path)
    except Exception as e:
        print(f"❌ 图片编码失败: {e}")
        return None
    
    url = f"{API_URL}?GroupId={GROUP_ID}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    # 结合算法数据
    user_content = f"{SYSTEM_PROMPT}\n\n【算法参考数据】该图的拉普拉斯清晰度得分为: {clarity_score}。请以此为参考进行视觉评审。"
    
    payload = {
        "model": "abab6.5s-chat", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json()
        
        reply_text = res_json['choices'][0]['message']['content']
        return extract_json_robustly(reply_text)
    except Exception as e:
        print(f"❌ API 或网络错误: {e}")
        return None

def main():
    # 路径配置
    input_dir = "./images"
    output_dir = "./output"
    temp_dir = os.path.join(output_dir, "algorithm_enhanced")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)] if os.path.exists(input_dir) else []
    
    if not files:
        print(f"⚠️ 未在 {input_dir} 找到图片。")
        return

    all_results = []
    print(f"🚀 启动金相 AI 评审系统 (处理数: {len(files)})")

    for i, filename in enumerate(files):
        raw_path = os.path.join(input_dir, filename)
        print(f"\n[{i+1}/{len(files)}] 正在分析: {filename}")
        
        # 1. 算法分支：获取增强灰度图并计算清晰度
        algo_img = preprocess_for_algorithm(raw_path)
        clarity_val = calculate_clarity(algo_img)
        
        # 保存增强后的图（供人工核对算法效果）
        if algo_img is not None:
            cv2.imwrite(os.path.join(temp_dir, f"enhanced_{filename}"), algo_img)
        
        # 2. AI 分支：发送彩色原图进行主观评价
        data = analyze_image_minimax(raw_path, clarity_val)
        
        if data:
            details = data.get('details', {})
            s_clarity = details.get('structure_clarity', {}).get('score', 0)
            s_scratches = details.get('scratches', {}).get('score', 0)
            s_artifacts = details.get('artifacts', {}).get('score', 0)
            
            row = {
                "文件名": filename,
                "总分(80)": s_clarity + s_scratches + s_artifacts,
                "清晰度得分(40)": s_clarity,
                "划痕得分(20)": s_scratches,
                "假象得分(20)": s_artifacts,
                "算法清晰度(Laplacian)": clarity_val,
                "专家评语": data.get('overall_critique', "解析成功但无评语"),
                "检测时间": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            all_results.append(row)
            print(f"✅ 完成评分: {row['总分(80)']}")
        else:
            print(f"⚠️ {filename} 处理失败，请检查网络或 AI 返回格式。")

    if all_results:
        df = pd.DataFrame(all_results)
        report_name = f"金相分析报告_{time.strftime('%m%d_%H%M')}.xlsx"
        df.to_excel(os.path.join(output_dir, report_name), index=False)
        print(f"\n✨ 全部处理完成！报告位置: {output_dir}/{report_name}")
    else:
        print("\n💥 未生成任何结果。")

if __name__ == "__main__":
    main()
