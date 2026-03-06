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
            return duration / 1000000
        return None
    except:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="使用 Whisper large-v3-turbo 生成视频字幕（极速版）"
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
        default="deepdml/faster-whisper-large-v3-turbo-ct2",
        help="模型名称（默认: large-v3-turbo）",
    )
    parser.add_argument(
        "-l", "--language", default=None, help="语言代码: zh/en/ja（默认: 自动检测）"
    )
    parser.add_argument("-t", "--translate", default=None, help="翻译目标语言: zh-CN")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备（默认: auto）",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["float16", "float32", "int8"],
        help="计算类型（默认: int8）",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print_subtitle",
        action="store_true",
        help="在控制台打印生成的字幕内容",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="批量翻译每批条数（默认: 20）",
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

    print(f"\n{'=' * 55}")
    print(f"  Whisper large-v3-turbo 字幕生成（极速版）")
    print(f"{'=' * 55}")
    print(f"文件: {os.path.basename(input_file)}")

    duration = get_video_duration(input_file)
    if duration:
        print(f"视频时长: {int(duration // 60)}分{int(duration % 60)}秒")
    else:
        duration = 0
        print(f"视频时长: 获取中...")

    print(f"模型: large-v3-turbo (4x faster)")
    print(f"设备: {args.device}")
    print(f"计算类型: {args.compute_type}")
    print(f"识别语言: {args.language if args.language else '自动检测'}")
    if args.translate:
        print(f"翻译: {args.translate}")
    if args.print_subtitle:
        print(f"控制台输出: 开启")
    print(f"{'=' * 55}\n")

    # 加载模型
    print("加载 large-v3-turbo 模型中（首次需下载约 1.5GB）...")
    load_start = time.time()

    try:
        model = WhisperModel(
            args.model, device=args.device, compute_type=args.compute_type
        )
        device_info = "GPU/Metal" if args.device == "auto" else args.device
    except Exception as e:
        print(f"加载失败，回退到 CPU: {e}")
        model = WhisperModel(args.model, device="cpu", compute_type="float32")
        device_info = "CPU"

    load_time = time.time() - load_start
    print(f"模型加载完成! 耗时: {load_time:.1f}秒, 设备: {device_info}")

    # 开始识别
    language = args.language if args.language and args.language != "auto" else None

    print("\n开始识别（turbo 模式）...")
    start_time = time.time()

    segments, info = model.transcribe(
        input_file,
        language=language,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    print(f"识别语言: {info.language} (概率: {info.language_probability:.2f})")

    # 收集识别结果
    segment_list = []
    if duration > 0:
        with tqdm(total=int(duration), desc="识别进度", unit="秒", ncols=70) as pbar:
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
    print(f"\n识别完成!")
    print(f"耗时: {elapsed:.1f}秒 ({elapsed / 60:.1f}分钟)")
    print(f"处理速度: {speed_ratio:.1f}x 实时")
    print(f"共 {len(segment_list)} 个片段")

    # 翻译
    if args.translate:
        print(f"\n批量翻译成 {args.translate}（每批 {args.batch_size} 条）...")
        translate_start = time.time()

        translator = GoogleTranslator(source="auto", target=args.translate)

        # 批量翻译
        total = len(segment_list)
        batch_size = args.batch_size
        batches = (total + batch_size - 1) // batch_size

        with tqdm(total=batches, desc="翻译进度", unit="批", ncols=70) as pbar:
            for i in range(0, total, batch_size):
                batch = segment_list[i : i + batch_size]
                # 合并文本，用特殊分隔符
                combined_text = "\n||||\n".join([s.text for s in batch])

                try:
                    translated = translator.translate(combined_text)
                    if translated:
                        # 拆分翻译结果
                        translated_texts = translated.split("\n||||\n")
                        # 如果分割数量匹配，逐条赋值
                        if len(translated_texts) == len(batch):
                            for j, text in enumerate(translated_texts):
                                batch[j].text = text.strip()
                        else:
                            # 分割不匹配，尝试其他分隔符
                            translated_texts = translated.split("||||")
                            if len(translated_texts) == len(batch):
                                for j, text in enumerate(translated_texts):
                                    batch[j].text = text.strip()
                except Exception as e:
                    # 批量失败，回退到逐条翻译
                    for segment in batch:
                        try:
                            result = translator.translate(segment.text)
                            if result:
                                segment.text = result
                        except:
                            pass
                pbar.update(1)

        translate_elapsed = time.time() - translate_start
        print(f"翻译完成! 耗时: {translate_elapsed:.1f}秒")

    # 生成字幕文件（一次性写入，极速）
    print("\n生成字幕文件...")
    gen_start = time.time()

    # 构建完整字幕内容
    srt_content = []
    for i, segment in enumerate(segment_list, 1):
        start = seconds_to_srt_time(segment.start)
        end = seconds_to_srt_time(segment.end)
        text = segment.text.strip()
        srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")

        # 控制台打印字幕
        if args.print_subtitle:
            print(f"[{start} --> {end}] {text}")

    # 一次性写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))

    gen_elapsed = time.time() - gen_start
    print(f"字幕生成完成! 耗时: {gen_elapsed:.2f}秒")

    total_time = time.time() - start_time
    print(f"\n{'=' * 55}")
    print(f"  完成!")
    print(f"  字幕: {output_file}")
    print(f"  总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
    print(f"  速度: {speed_ratio:.1f}x 实时")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
