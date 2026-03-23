import os
import base64
import json
import requests
import pandas as pd
import time
from config import API_KEY, GROUP_ID, API_URL

# --- 核心评分 Prompt (强化 JSON 要求) ---
SYSTEM_PROMPT = """你是一位资深金相分析专家。请根据上传的金相照片评分（总分80分）。
请严格按照以下 JSON 格式输出，不要包含任何开头、结尾或解释性文字：
{
  "total_score": 80.0,
  "details": {
    "structure_clarity": {"score": 35, "reason": "晶界清晰"},
    "scratches": {"score": 18, "reason": "无明显划痕"},
    "artifacts": {"score": 15, "reason": "轻微污染"}
  },
  "overall_critique": "综合评价"
}"""

def encode_image(image_path):
    """图片转 Base64，并添加 Data URI 前缀"""
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

def analyze_image_minimax(image_path):
    """调用 MiniMax 视觉模型 (兼容版)"""
    print(f"🔍 正在分析: {os.path.basename(image_path)}...")
    image_url = encode_image(image_path)
    
    url = f"{API_URL}?GroupId={GROUP_ID}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 删除了引起 400 错误的 response_format 参数
    payload = {
        "model": "abab6.5s-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json()
        
        # 提取回复内容
        reply_text = res_json['choices'][0]['message']['content']
        
        # 稳健性处理：去掉 Markdown 代码块标签
        clean_json = reply_text.replace('```json', '').replace('```', '').strip()
        
        return json.loads(clean_json)
        
    except Exception as e:
        print(f"❌ 分析出错: {e}")
        if 'response' in locals():
            print(f"💡 服务器详细反馈: {response.text}")
        return None

def main():
    img_dir = "./images"
    output_dir = "./output"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("💡 提示: images 文件夹里没有发现图片。")
        return

    all_results = []
    print(f"🚀 环境就绪，开始处理 {len(files)} 张图片...\n")

    for filename in files:
        data = analyze_image_minimax(os.path.join(img_dir, filename))
        if data:
            row = {
                "文件名": filename,
                "总分(80)": data.get('total_score'),
                "清晰度得分": data.get('details', {}).get('structure_clarity', {}).get('score'),
                "划痕得分": data.get('details', {}).get('scratches', {}).get('score'),
                "综合评价": data.get('overall_critique'),
                "处理时间": time.strftime("%H:%M:%S")
            }
            all_results.append(row)
            time.sleep(1) # 免费版稍微留点空隙

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(os.path.join(output_dir, "金相最终分析报告.xlsx"), index=False)
        print("\n✨ 恭喜！分析任务全部完成！")
        print(f"📊 报告位置: {os.path.abspath(output_dir)}")
    else:
        print("\n❌ 未能成功生成任何分析数据。")

if __name__ == "__main__":
    main()