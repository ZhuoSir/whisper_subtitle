import argparse
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm

# Google 翻译
try:
    from deep_translator import GoogleTranslator
except ImportError:
    os.system("pip install deep-translator")
    from deep_translator import GoogleTranslator

# 本地翻译模型（延迟加载）
local_translator = None
local_tokenizer = None


def load_local_translator(source_lang="ja", target_lang="zh"):
    """加载本地翻译模型"""
    global local_translator, local_tokenizer

    if local_translator is not None:
        return local_translator, local_tokenizer

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # 语言对映射
    model_map = {
        ("ja", "zh"): "larryvrh/mt5-translation-ja_zh",
        ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
        ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
        ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
        ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    }

    key = (source_lang, target_lang)
    if key not in model_map:
        print(f"警告: 不支持的语言对 {source_lang} -> {target_lang}，使用 Google 翻译")
        return None, None

    model_name = model_map[key]
    print(f"加载本地翻译模型: {model_name}（首次需下载）...")

    local_tokenizer = AutoTokenizer.from_pretrained(model_name)
    local_translator = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return local_translator, local_tokenizer


def translate_local(texts, source_lang="ja", target_lang="zh"):
    """使用本地模型批量翻译"""
    model, tokenizer = load_local_translator(source_lang, target_lang)

    if model is None:
        return None

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    translated = model.generate(**inputs)
    results = tokenizer.batch_decode(
        translated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return results


def translate_google_batch(texts, target_lang="zh-CN", batch_size=20):
    """使用 Google 批量翻译"""
    translator = GoogleTranslator(source="auto", target=target_lang)
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        combined = "\n||||\n".join(batch)

        try:
            translated = translator.translate(combined)
            if translated:
                parts = translated.split("\n||||\n")
                if len(parts) == len(batch):
                    results.extend([p.strip() for p in parts])
                else:
                    for text in batch:
                        try:
                            result = translator.translate(text)
                            results.append(result if result else text)
                        except:
                            results.append(text)
        except:
            for text in batch:
                try:
                    result = translator.translate(text)
                    results.append(result if result else text)
                except:
                    results.append(text)

    return results


def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_video_duration(input_file):
    import av

    try:
        container = av.open(input_file)
        duration = container.duration
        if duration:
            return duration / 1000000
        return None
    except:
        return None


def preprocess_audio(input_file, output_audio_file):
    """
    预处理音频，让模型听得更清楚：
    1. 提取音频
    2. 降噪 (afftdn)
    3. 音量标准化 (loudnorm)
    4. 采样率 16kHz, 单声道
    """
    print(f"\n🎧 正在预处理音频 (降噪、人声提取、音量标准化)...")
    start_time = time.time()

    # afftdn: 傅里叶降噪
    # loudnorm: 响度标准化 (解决声音忽大忽小)
    audio_filter = "afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-vn",  # 无视频
        "-af",
        audio_filter,
        "-ac",
        "1",  # 单声道
        "-ar",
        "16000",  # 16kHz 采样率
        "-c:a",
        "pcm_s16le",  # 无损 WAV 格式
        output_audio_file,
    ]

    try:
        process = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if process.returncode != 0:
            print(f"音频预处理失败，将直接使用原视频: {process.stderr[-200:]}")
            return input_file
        print(f"音频清洗完成! 耗时: {time.time() - start_time:.1f}秒")
        return output_audio_file
    except Exception as e:
        print(f"执行 FFmpeg 失败，将直接使用原视频: {e}")
        return input_file


def main():
    parser = argparse.ArgumentParser(
        description="Whisper 字幕生成 HD 版（高清晰度、高精度版）"
    )
    parser.add_argument("input", help="视频或音频文件路径")
    parser.add_argument("-o", "--output", help="输出字幕文件路径")
    parser.add_argument(
        "-d",
        "--dir",
        dest="output_dir",
        default=None,
        help="输出字幕目录（默认: 当前目录）",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="large-v3",
        help="Whisper 模型（默认: large-v3，最准确）",
    )
    parser.add_argument(
        "-l", "--language", default=None, help="识别语言: zh/en/ja（默认: 自动检测）"
    )
    parser.add_argument(
        "-t", "--translate", default=None, help="翻译目标语言: zh/zh-CN/en/ja"
    )
    parser.add_argument(
        "--translator",
        default="local",
        choices=["local", "google"],
        help="翻译引擎: local(本地模型,极速)/google（默认: local）",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备（默认: auto）",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "float32", "int8"],
        help="计算类型（默认: float16，精度高）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="批量翻译大小（默认: 32）"
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print_subtitle",
        action="store_true",
        help="在控制台打印字幕",
    )
    parser.add_argument(
        "--no-preprocess", action="store_true", help="禁用音频降噪与预处理"
    )

    args = parser.parse_args()
    input_file = args.input

    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)

    # 确定输出文件路径
    if args.output:
        output_file = args.output
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(args.output_dir, f"{base_name}.srt")
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}.srt"

    print(f"\n{'=' * 60}")
    print(f"  Whisper 字幕生成 HD（高清晰度、高精度版）")
    print(f"{'=' * 60}")
    print(f"文件: {os.path.basename(input_file)}")

    duration = get_video_duration(input_file)
    if duration:
        print(f"视频时长: {int(duration // 60)}分{int(duration % 60)}秒")
    else:
        duration = 0
        print(f"视频时长: 获取中...")

    print(f"Whisper模型: {args.model}")
    print(f"识别精度: beam_size=5 (高精度)")
    print(f"识别语言: {args.language if args.language else '自动检测'}")
    if args.translate:
        print(f"翻译目标: {args.translate} ({args.translator})")
    print(f"{'=' * 60}\n")

    # ========== 0. 音频预处理 ==========
    audio_to_process = input_file
    temp_audio = None
    if not args.no_preprocess:
        temp_audio = os.path.join(
            tempfile.gettempdir(), f"whisper_clean_{int(time.time())}.wav"
        )
        audio_to_process = preprocess_audio(input_file, temp_audio)

    # ========== 1. 加载 Whisper 模型 ==========
    print("\n加载 Whisper 模型中...")
    load_start = time.time()

    try:
        model = WhisperModel(
            args.model, device=args.device, compute_type=args.compute_type
        )
        device_info = "GPU/Metal" if args.device == "auto" else args.device
    except Exception as e:
        print(f"加载失败，回退到 CPU (float32): {e}")
        model = WhisperModel(args.model, device="cpu", compute_type="float32")
        device_info = "CPU"

    print(
        f"Whisper 加载完成! 耗时: {time.time() - load_start:.1f}秒, 设备: {device_info}"
    )

    # ========== 2. 语音识别 ==========
    language = args.language if args.language and args.language != "auto" else None

    print("\n开始高精度语音识别...")
    start_time = time.time()

    # 高精度识别参数
    segments, info = model.transcribe(
        audio_to_process,
        language=language,
        beam_size=5,  # 提升到 5，牺牲一点速度换取高准确率
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # 温度回退，防止胡言乱语
        condition_on_previous_text=False,  # 防止重复幻觉
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=400,  # 在语音前后多保留一点静音，防止吞字
        ),
    )

    print(f"识别语言: {info.language} (概率: {info.language_probability:.2f})")

    # 收集识别结果
    segment_list = []
    if duration > 0:
        with tqdm(total=int(duration), desc="识别进度", unit="秒", ncols=70) as pbar:
            last_time = 0
            for segment in segments:
                segment_list.append(
                    {"start": segment.start, "end": segment.end, "text": segment.text}
                )
                if segment.end > last_time:
                    pbar.update(int(segment.end - last_time))
                    last_time = segment.end
    else:
        for segment in segments:
            segment_list.append(
                {"start": segment.start, "end": segment.end, "text": segment.text}
            )
            text_preview = segment.text[:30] if len(segment.text) > 30 else segment.text
            print(f"已识别: {segment.start:.1f}s - {text_preview}...")

    recognize_time = time.time() - start_time
    speed_ratio = (
        duration / recognize_time if recognize_time > 0 and duration > 0 else 0
    )
    print(f"\n识别完成! 耗时: {recognize_time:.1f}秒, 速度: {speed_ratio:.1f}x 实时")
    print(f"共 {len(segment_list)} 个片段")

    # 清理临时音频文件
    if temp_audio and os.path.exists(temp_audio):
        os.remove(temp_audio)

    # ========== 3. 翻译 ==========
    if args.translate and len(segment_list) > 0:
        print(f"\n开始翻译 ({args.translator})...")
        translate_start = time.time()
        texts = [s["text"] for s in segment_list]

        if args.translator == "local":
            source_lang = (
                info.language
                if info.language
                else (args.language if args.language else "ja")
            )
            target_lang = args.translate.replace("-CN", "").replace("-TW", "").lower()

            batch_size = args.batch_size
            total_batches = (len(texts) + batch_size - 1) // batch_size
            translated_texts = []

            with tqdm(
                total=total_batches, desc="翻译进度", unit="批", ncols=70
            ) as pbar:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    try:
                        results = translate_local(batch, source_lang, target_lang)
                        if results:
                            translated_texts.extend(results)
                        else:
                            google_results = translate_google_batch(
                                batch, args.translate, batch_size
                            )
                            translated_texts.extend(google_results)
                    except Exception as e:
                        print(f"\n本地翻译出错，回退到 Google...")
                        google_results = translate_google_batch(
                            batch, args.translate, batch_size
                        )
                        translated_texts.extend(google_results)
                    pbar.update(1)

            for i, text in enumerate(translated_texts):
                if i < len(segment_list):
                    segment_list[i]["text"] = text
        else:
            total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
            translated_texts = []

            with tqdm(
                total=total_batches, desc="翻译进度", unit="批", ncols=70
            ) as pbar:
                for i in range(0, len(texts), args.batch_size):
                    batch = texts[i : i + args.batch_size]
                    results = translate_google_batch(batch, args.translate, len(batch))
                    translated_texts.extend(results)
                    pbar.update(1)

            for i, text in enumerate(translated_texts):
                if i < len(segment_list):
                    segment_list[i]["text"] = text

        print(f"翻译完成! 耗时: {time.time() - translate_start:.1f}秒")

    # ========== 4. 生成字幕文件 ==========
    print("\n生成字幕文件...")
    srt_content = []
    for i, segment in enumerate(segment_list, 1):
        start = seconds_to_srt_time(segment["start"])
        end = seconds_to_srt_time(segment["end"])
        text = segment["text"].strip()
        srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")

        if args.print_subtitle:
            print(f"[{start} --> {end}] {text}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))

    # ========== 5. 完成 ==========
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  完成!")
    print(f"  字幕文件: {output_file}")
    print(f"  总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
