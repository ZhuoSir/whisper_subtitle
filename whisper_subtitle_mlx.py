import argparse
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

# =================================================================
# MLX Whisper 专属依赖检查与延迟加载
# =================================================================
try:
    import mlx_whisper
except ImportError:
    print("错误: MLX Whisper 未安装。请运行 'pip install mlx-whisper'")
    sys.exit(1)

# =================================================================
# Google 与本地翻译模型的相关函数 (从 Pro 版迁移)
# =================================================================
local_translator = None
local_tokenizer = None


def load_local_translator(source_lang="ja", target_lang="zh"):
    """加载本地翻译模型"""
    global local_translator, local_tokenizer
    if local_translator:
        return local_translator, local_tokenizer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_map = {
        ("ja", "zh"): "larryvrh/mt5-translation-ja_zh",
        ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
        ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
        ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
        ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    }
    key = (source_lang, target_lang)
    if key not in model_map:
        return None, None
    model_name = model_map[key]
    print(f"加载本地翻译模型: {model_name}...")
    local_tokenizer = AutoTokenizer.from_pretrained(model_name)
    local_translator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return local_translator, local_tokenizer


def translate_local(texts, source_lang="ja", target_lang="zh"):
    """本地模型批量翻译"""
    model, tokenizer = load_local_translator(source_lang, target_lang)
    if not model or not tokenizer:
        return None
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(
        translated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def translate_google_batch(texts, target_lang="zh-CN", batch_size=20):
    """Google 批量翻译"""
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target=target_lang)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        combined = "\n||||\n".join(batch)
        try:
            translated = translator.translate(combined)
            parts = translated.split("\n||||\n") if translated else []
            if len(parts) == len(batch):
                results.extend(p.strip() for p in parts)
            else:
                raise ValueError("Batch translation failed")
        except:
            for text in batch:
                try:
                    results.append(translator.translate(text) or text)
                except:
                    results.append(text)
    return results


# =================================================================
# 工具函数
# =================================================================


def seconds_to_srt_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"


def get_video_duration(input_file):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]
    try:
        return float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
    except:
        return 0


def preprocess_audio(input_file, output_audio_file):
    """音频预处理，提取并转换为 16kHz WAV"""
    print(f"🎧 正在预处理音频 (提取并转换为 16kHz WAV)...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output_audio_file,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ 音频预处理完成!")
        return output_audio_file
    except subprocess.CalledProcessError as e:
        print(
            f"⚠️ 音频预处理失败: {e.stderr.decode()[-200:]}\n将直接使用原视频文件识别。"
        )
        return input_file


# =================================================================
# 主函数
# =================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Whisper 字幕生成 MLX 极速版 (Apple Silicon 专属)"
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
        default="mlx-community/whisper-large-v3-mlx",
        help="MLX Whisper 模型",
    )
    parser.add_argument(
        "-l", "--language", default=None, help="识别语言: ja/zh/en (默认: 自动)"
    )
    parser.add_argument(
        "-t", "--translate", default=None, help="翻译目标语言: zh/en/ja"
    )
    parser.add_argument(
        "--translator",
        default="local",
        choices=["local", "google"],
        help="翻译引擎 (默认: local)",
    )
    parser.add_argument("--no-preprocess", action="store_true", help="禁用音频预处理")
    parser.add_argument(
        "-p",
        "--print",
        dest="print_subtitle",
        action="store_true",
        help="在控制台打印字幕",
    )

    args = parser.parse_args()

    # --- 文件路径设置 ---
    output_file = args.output
    if not output_file:
        dir_path = args.output_dir or os.getcwd()
        os.makedirs(dir_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_file = os.path.join(dir_path, f"{base_name}.srt")

    print(
        f"\n{'=' * 60}\n  Whisper 字幕生成 MLX 极速版 (Apple Silicon 专属)\n{'=' * 60}"
    )
    print(f"文件: {os.path.basename(args.input)}")

    # --- 音频预处理 ---
    audio_to_process = args.input
    temp_audio_file = None
    if not args.no_preprocess:
        temp_dir = tempfile.gettempdir()
        temp_audio_file = os.path.join(temp_dir, f"mlx_whisper_{int(time.time())}.wav")
        audio_to_process = preprocess_audio(args.input, temp_audio_file)

    # --- MLX Whisper 识别 ---
    print(f"\n🚀 开始 MLX 识别 (模型: {args.model})...")
    rec_start_time = time.time()
    result = mlx_whisper.transcribe(
        audio=audio_to_process,
        path_or_hf_repo=args.model,
        language=args.language,
        verbose=True,  # MLX 自带 TQDM 进度条
    )
    rec_time = time.time() - rec_start_time
    print(f"\n✅ 识别完成! 耗时: {rec_time:.1f}秒")

    segment_list = result["segments"]

    # 清理临时音频
    if temp_audio_file and os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)

    # --- 翻译 ---
    if args.translate and segment_list:
        print(f"\n🌍 开始翻译 ({args.translator})...")
        trans_start_time = time.time()
        texts_to_translate = [seg["text"] for seg in segment_list]

        if args.translator == "local":
            source_lang = result.get("language", "ja")
            target_lang = args.translate.replace("-CN", "").lower()
            translated_texts = translate_local(
                texts_to_translate, source_lang, target_lang
            )
        else:
            translated_texts = translate_google_batch(
                texts_to_translate, args.translate
            )

        if translated_texts:
            for i, text in enumerate(translated_texts):
                segment_list[i]["text"] = text
        print(f"✅ 翻译完成! 耗时: {time.time() - trans_start_time:.1f}秒")

    # --- 生成 SRT ---
    print("\n📝 生成字幕文件...")
    srt_content = []
    for i, segment in enumerate(segment_list, 1):
        start = seconds_to_srt_time(segment["start"])
        end = seconds_to_srt_time(segment["end"])
        text = segment["text"].strip()
        srt_line = f"{i}\n{start} --> {end}\n{text}\n"
        srt_content.append(srt_line)
        if args.print_subtitle:
            print(f"[{start} --> {end}] {text}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))

    print(f"\n{'=' * 60}\n  🎉 完成!\n  字幕文件: {output_file}\n{'=' * 60}")


if __name__ == "__main__":
    main()
