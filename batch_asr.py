from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import soundfile as sf
import os
import pandas as pd
from tqdm import tqdm  # 用于进度显示

# 模型路径
model_dir = "SenseVoiceSmall"
vad_model_dir = "fsmn-vad"  # VAD模型路径

# 音频文件夹路径
audio_folder = "extracted_audio"
output_folder = "transcriptions"
os.makedirs(output_folder, exist_ok=True)

# 加载VAD模型
vad_model = AutoModel(
    model=vad_model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:1",
    disable_update=True
)

# 加载SenseVoice模型
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:1",
    disable_update=True
)

# 定义裁剪音频函数
def crop_audio(audio_data, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate / 1000)  # 转换为样本数
    end_sample = int(end_time * sample_rate / 1000)  # 转换为样本数
    return audio_data[start_sample:end_sample]

# 获取所有WAV文件
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]

# 进度条
with tqdm(total=len(audio_files), desc="Processing audio files") as pbar:
    for audio_file in audio_files:
        try:
            audio_path = os.path.join(audio_folder, audio_file)

            # 使用VAD模型处理音频文件
            vad_res = vad_model.generate(
                input=audio_path,
                cache={},
                max_single_segment_time=30000,
            )
            # 从VAD模型的输出中提取每个语音片段的开始和结束时间
            segments = vad_res[0]['value']

            # 加载原始音频数据
            audio_data, sample_rate = sf.read(audio_path)

            results = []
            for idx, segment in enumerate(segments):
                start_time, end_time = segment
                cropped_audio = crop_audio(audio_data, start_time, end_time, sample_rate)

                # 临时文件路径
                temp_audio_file = f"temp_{idx}.wav"
                sf.write(temp_audio_file, cropped_audio, sample_rate)

                # 语音转文字处理
                res = model.generate(
                    input=temp_audio_file,
                    cache={},
                    language="en",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=10000,
                )

                # 删除临时音频文件
                os.remove(temp_audio_file)

                # 处理输出结果
                text = rich_transcription_postprocess(res[0]["text"])
                # results.append({"start": start_time // 1000, "end": end_time // 1000, "text": text})
                results.append({"start": round(start_time / 1000, 3), "end": round(end_time / 1000, 3), "text": text})


            # 将结果保存到CSV文件
            output_file = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}.csv")
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"✅ 完成: {audio_file} -> {output_file}")

        except Exception as e:
            print(f"❌ 处理失败: {audio_file}, 错误: {e}")

        # 更新进度条
        pbar.update(1)

print("🚀 所有音频文件处理完毕！")
