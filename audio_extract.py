from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import soundfile as sf  # 用于读取和裁剪音频文件
import os
import pandas as pd
 
# 模型路径
model_dir = "SenseVoiceSmall"
vad_model_dir = "fsmn-vad"  # VAD模型路径
 
# 音频文件路径
audio_file_path = 'med.wav'
 
# 加载VAD模型
vad_model = AutoModel(
    model=vad_model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:1",
    disable_update=True
)
 
# 使用VAD模型处理音频文件
vad_res = vad_model.generate(
    input=audio_file_path,
    cache={},
    max_single_segment_time=30000,  # 最大单个片段时长
)
 
# 从VAD模型的输出中提取每个语音片段的开始和结束时间
segments = vad_res[0]['value']  # 假设只有一段音频，且其片段信息存储在第一个元素中
 
# 加载原始音频数据
audio_data, sample_rate = sf.read(audio_file_path)
 
 
# 定义一个函数来裁剪音频
def crop_audio(audio_data, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate / 1000)  # 转换为样本数
    end_sample = int(end_time * sample_rate / 1000)  # 转换为样本数
    return audio_data[start_sample:end_sample]
 
 
# 加载SenseVoice模型
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:1",
    disable_update=True
)
 
# 对每个语音片段进行处理
results = []
for segment in segments:
    start_time, end_time = segment  # 获取开始和结束时间
    cropped_audio = crop_audio(audio_data, start_time, end_time, sample_rate)
 
    # 将裁剪后的音频保存为临时文件
    temp_audio_file = "temp_cropped.wav"
    sf.write(temp_audio_file, cropped_audio, sample_rate)
 
    # 语音转文字处理
    res = model.generate(
        input=temp_audio_file,
        cache={},
        language="auto",  # 自动检测语言
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  # 启用 VAD 断句
        merge_length_s=10000,  # 合并长度，单位为毫秒
    )
    # 处理输出结果
    text = rich_transcription_postprocess(res[0]["text"])
    # 添加时间戳
    results.append({"start": start_time // 1000, "end": end_time // 1000, "text": text})  # 转换为秒
 
# 输出结果
for result in results:
    print(f"Start: {result['start']} s, End: {result['end']} s, Text: {result['text']}")