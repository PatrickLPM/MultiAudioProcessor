import os
import pandas as pd
import re
from tqdm import tqdm

# 去除表情符号的函数
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 符号和标志
        "\U0001F680-\U0001F6FF"  # 运输和地图符号
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # 杂项符号
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# 输入和输出文件夹
input_folder = "transcriptions"
output_folder = "cleaned_transcriptions_accuracy"
os.makedirs(output_folder, exist_ok=True)

# 获取所有CSV文件
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# 批量处理CSV文件
with tqdm(total=len(csv_files), desc="Processing CSV files") as pbar:
    for csv_file in csv_files:
        try:
            input_path = os.path.join(input_folder, csv_file)
            output_path = os.path.join(output_folder, csv_file)

            # 加载CSV文件
            df = pd.read_csv(input_path)

            # 检查是否存在'text'列
            if 'text' not in df.columns:
                print(f"❌ 跳过：{csv_file}，因为没有'text'列")
                pbar.update(1)
                continue

            # 去除表情符号
            df['text'] = df['text'].astype(str).apply(remove_emojis)

            # 保存去除表情后的CSV
            df.to_csv(output_path, index=False)
            print(f"✅ 已清理：{csv_file}")

        except Exception as e:
            print(f"❌ 处理失败：{csv_file}，错误：{e}")

        pbar.update(1)

print("🚀 所有CSV文件处理完毕！")
