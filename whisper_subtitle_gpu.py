import argparse
import os
import sys
import time
from faster_whisper import WhisperModel
from tqdm import tqdm

try:
    from deep_translator import GoogleTranslator
except ImportError:
    os.system("pip install deep-translator")
    from deep_translator import GoogleTranslator


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
            return duration / 1000000  # 微秒转秒
        return None
    except:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="使用 Whisper 生成视频字幕（GPU加速版）"
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
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
        ],
        help="模型: tiny < base < small < medium < large < large-v3-turbo(推荐)",
    )
    parser.add_argument(
        "-l", "--language", default=None, help="语言代码: zh/en/ja（默认: 自动检测）"
    )
    parser.add_argument("-t", "--translate", default=None, help="翻译目标语言: zh-CN")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备: auto(自动)/cpu/cuda（默认: auto）",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["float16", "float32", "int8"],
        help="计算类型: int8(快)/float16/float32（默认: int8）",
    )
    parser.add_argument(
        "--beam-size", type=int, default=1, help="束搜索大小: 1(快)-5(准)（默认: 1）"
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

    print(f"\n{'=' * 50}")
    print(f"视频字幕生成工具（GPU加速版）")
    print(f"{'=' * 50}")
    print(f"文件: {os.path.basename(input_file)}")

    duration = get_video_duration(input_file)
    if duration:
        print(f"视频时长: {int(duration // 60)}分{int(duration % 60)}秒")
    else:
        duration = 0
        print(f"视频时长: 获取中...")

    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"计算类型: {args.compute_type}")
    print(f"束搜索: {args.beam_size}")
    print(f"识别语言: {args.language if args.language else '自动检测'}")
    if args.translate:
        print(f"翻译: {args.translate}")
    print(f"{'=' * 50}\n")

    # 加载模型（启用GPU加速）
    print("加载模型中...")
    load_start = time.time()

    try:
        model = WhisperModel(
            args.model, device=args.device, compute_type=args.compute_type
        )
        device_info = "GPU/Metal" if args.device == "auto" else args.device
    except Exception as e:
        print(f"GPU加载失败，回退到CPU: {e}")
        model = WhisperModel(args.model, device="cpu", compute_type="float32")
        device_info = "CPU"

    load_time = time.time() - load_start
    print(f"模型加载完成! 耗时: {load_time:.1f}秒, 设备: {device_info}")

    # 开始识别
    language = args.language if args.language and args.language != "auto" else None

    print("\n开始识别...")
    start_time = time.time()

    segments, info = model.transcribe(
        input_file,
        language=language,
        beam_size=args.beam_size,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    print(f"识别语言: {info.language} (概率: {info.language_probability:.2f})")

    # 收集识别结果并显示进度
    segment_list = []
    if duration > 0:
        with tqdm(total=int(duration), desc="识别进度", unit="秒") as pbar:
            last_time = 0
            for segment in segments:
                segment_list.append(segment)
                if segment.end > last_time:
                    pbar.update(int(segment.end - last_time))
                    last_time = segment.end
    else:
        for segment in segments:
            segment_list.append(segment)
            text_preview = segment.text[:30] if len(segment.text) > 30 else segment.text
            print(f"已识别: {segment.start:.1f}s - {text_preview}...")

    elapsed = time.time() - start_time
    speed_ratio = duration / elapsed if elapsed > 0 and duration > 0 else 0
    print(f"\n识别完成! 耗时: {elapsed:.1f}秒 ({elapsed / 60:.1f}分钟)")
    print(f"处理速度: {speed_ratio:.1f}x 实时")
    print(f"共 {len(segment_list)} 个片段")

    # 翻译（如果需要）
    if args.translate:
        print(f"\n翻译成 {args.translate}...")
        translate_start = time.time()

        translator = GoogleTranslator(source="auto", target=args.translate)
        total = len(segment_list)

        with tqdm(total=total, desc="翻译进度", unit="条") as pbar:
            for segment in segment_list:
                try:
                    translated = translator.translate(segment.text)
                    if translated:
                        segment.text = translated
                except:
                    pass
                pbar.update(1)

        translate_elapsed = time.time() - translate_start
        print(f"翻译完成! 耗时: {translate_elapsed:.1f}秒")

    # 生成字幕文件
    print("\n生成字幕文件...")
    total = len(segment_list)
    with tqdm(total=total, desc="生成字幕", unit="条") as pbar:
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segment_list, 1):
                start = seconds_to_srt_time(segment.start)
                end = seconds_to_srt_time(segment.end)
                text = segment.text.strip()

                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
                pbar.update(1)

    total_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"完成!")
    print(f"字幕保存至: {output_file}")
    print(f"总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
