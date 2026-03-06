import argparse
import os
import sys
import time
import threading
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


def translate_text(text, target_lang="zh-CN"):
    try:
        translator = GoogleTranslator(source="auto", target=target_lang)
        result = translator.translate(text)
        return result if result else text
    except Exception as e:
        print(f"翻译错误: {e}")
        return text


def get_video_duration(input_file):
    import av

    try:
        container = av.open(input_file)
        return container.duration / av.time_base
    except:
        return None


def generate_srt(transcription, output_path, translate_to=None, start_time=0.0):
    translator = None
    if translate_to:
        translator = GoogleTranslator(source="auto", target=translate_to)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(transcription, 1):
            # 调整时间戳，减去开始时间
            adjusted_start = max(0, segment.start - start_time)
            adjusted_end = max(0, segment.end - start_time)
            start = seconds_to_srt_time(adjusted_start)
            end = seconds_to_srt_time(adjusted_end)
            text = segment.text.strip()

            if translator:
                try:
                    text = translator.translate(text)
                except:
                    pass

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


class ProgressTracker:
    def __init__(self, total_duration):
        self.total_duration = total_duration
        self.current_time = 0
        self.start_time = time.time()
        self.running = True
        self.pbar = None

    def update(self, current_time):
        self.current_time = current_time
        if self.pbar:
            self.pbar.update(1)

    def close(self):
        self.running = False
        if self.pbar:
            self.pbar.close()


def main():
    parser = argparse.ArgumentParser(description="使用 Whisper 生成视频字幕")
    parser.add_argument("input", help="视频或音频文件路径")
    parser.add_argument("-o", "--output", help="输出字幕文件路径（默认与输入文件同名）")
    parser.add_argument(
        "-d",
        "--dir",
        "--output-dir",
        dest="output_dir",
        default=None,
        help="输出字幕目录（默认: 与输入文件同目录）",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper 模型大小: tiny(最快) < base < small(推荐) < medium < large(最准)",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=float,
        default=0.0,
        help="从第几分钟开始生成字幕（默认: 0.0）",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="语言代码，如 zh、en、ja（默认: 自动检测）",
    )
    parser.add_argument(
        "-t",
        "--translate",
        default=None,
        help="翻译目标语言，如 zh-CN（日语转中文用 ja>zh-CN）",
    )
    parser.add_argument(
        "--compute-type",
        default="float32",
        choices=["float16", "float32", "int8"],
        help="计算类型（默认: float32）",
    )

    args = parser.parse_args()
    input_file = args.input

    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)

    if args.output:
        output_file = args.output
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(args.output_dir, f"{base_name}.srt")
    else:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.srt"

    print(f"\n{'=' * 50}")
    print(f"视频字幕生成工具")
    print(f"{'=' * 50}")
    print(f"文件: {os.path.basename(input_file)}")

    duration = get_video_duration(input_file)
    if duration:
        print(f"视频时长: {int(duration // 60)}分{int(duration % 60)}秒")
    else:
        duration = 0
        print(f"视频时长: 获取中...")

    print(f"模型: {args.model}")
    print(f"识别语言: {args.language if args.language else '自动检测'}")
    if args.translate:
        print(f"翻译: {args.translate}")
    print(f"{'=' * 50}\n")

    print("加载模型中...")
    model = WhisperModel(args.model, compute_type=args.compute_type)

    language = args.language if args.language and args.language != "auto" else None

    print("开始识别...")
    start_time = time.time()

    segments, info = model.transcribe(
        input_file, language=language, beam_size=5, vad_filter=True
    )

    print(f"识别语言: {info.language} (概率: {info.language_probability:.2f})")

    segment_list = []
    if duration > 0:
        # 调整进度条总时长，从开始时间到视频结束
        adjusted_duration = max(0, duration - args.start * 60)
        with tqdm(total=int(adjusted_duration), desc="识别进度", unit="秒") as pbar:
            last_time = args.start * 60  # 从开始时间开始
            for segment in segments:
                segment_list.append(segment)
                if segment.end > last_time:
                    pbar.update(int(segment.end - last_time))
                    last_time = segment.end
    else:
        for segment in segments:
            segment_list.append(segment)
            print(f"已识别: {segment.start:.1f}s - {segment.text[:30]}...")

    elapsed = time.time() - start_time
    print(f"\n识别完成! 耗时: {elapsed:.1f}秒 ({elapsed / 60:.1f}分钟)")
    print(f"共 {len(segment_list)} 个片段")

    if args.translate:
        print(f"\n翻译成 {args.translate}...")
        translate_start = time.time()

        translator = GoogleTranslator(source="auto", target=args.translate)
        total = len(segment_list)

        with tqdm(total=total, desc="翻译进度", unit="条") as pbar:
            for segment in segment_list:
                try:
                    segment.text = translator.translate(segment.text)
                except:
                    pass
                pbar.update(1)

        translate_elapsed = time.time() - translate_start
        print(f"翻译完成! 耗时: {translate_elapsed:.1f}秒")

    print("\n生成字幕文件...")
    total = len(segment_list)
    with tqdm(total=total, desc="生成字幕", unit="条") as pbar:
        translator = None
        if args.translate:
            translator = GoogleTranslator(source="auto", target=args.translate)

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
