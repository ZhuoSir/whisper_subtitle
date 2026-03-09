#!/usr/bin/env python3
"""
字幕合并视频工具
将 SRT 字幕硬编码到视频中，生成带字幕的新视频
"""

import argparse
import os
import sys
import subprocess
import time
import re


def get_video_info(video_path):
    """获取视频信息"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-of",
        "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        parts = result.stdout.strip().split(",")
        if len(parts) >= 3:
            return {
                "width": int(parts[0]),
                "height": int(parts[1]),
                "duration": float(parts[2]) if parts[2] else 0,
            }
    except:
        pass
    return None


def get_video_duration(video_path):
    """获取视频时长"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0


def merge_subtitle(
    video_path,
    subtitle_path,
    output_path,
    font_size=24,
    font_color="white",
    outline_color="black",
    outline_width=2,
    position="bottom",
    margin_v=30,
):
    """
    合并字幕到视频

    Args:
        video_path: 输入视频路径
        subtitle_path: 字幕文件路径
        output_path: 输出视频路径
        font_size: 字体大小
        font_color: 字体颜色
        outline_color: 描边颜色
        outline_width: 描边宽度
        position: 字幕位置 (bottom/top)
        margin_v: 垂直边距
    """

    # 字幕位置
    alignment = (
        2 if position == "bottom" else 6
    )  # ASS alignment: 2=底部居中, 6=顶部居中

    # 构建 subtitles 滤镜
    # 使用绝对路径并转义特殊字符
    subtitle_path_escaped = subtitle_path.replace(":", "\\:").replace("'", "\\'")

    # 字幕样式
    style = f"FontSize={font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline={outline_width},MarginV={margin_v},Alignment={alignment}"

    subtitle_filter = f"subtitles='{subtitle_path_escaped}':force_style='{style}'"

    # FFmpeg 命令
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-c:a",
        "copy",
        "-y",  # 覆盖输出文件
        output_path,
    ]

    return cmd


def run_ffmpeg_with_progress(cmd, duration, verbose=False):
    """运行 FFmpeg 并显示进度"""

    if verbose:
        # 详细模式：直接运行并显示 FFmpeg 输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if process.stdout:
            for line in process.stdout:
                print(line, end="")
        return process.wait()

    # 简单模式：只显示进度条
    print(f"\n开始合并...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="字幕合并视频工具 - 将 SRT 字幕硬编码到视频中"
    )
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("-s", "--subtitle", required=True, help="SRT 字幕文件路径")
    parser.add_argument("-o", "--output", help="输出视频文件路径")
    parser.add_argument(
        "-d", "--dir", dest="output_dir", help="输出目录（默认: 与输入视频同目录）"
    )
    parser.add_argument(
        "--font-size", type=int, default=24, help="字幕字体大小（默认: 24）"
    )
    parser.add_argument("--font-color", default="white", help="字幕颜色（默认: white）")
    parser.add_argument(
        "--outline", type=int, default=2, help="字幕描边宽度（默认: 2）"
    )
    parser.add_argument(
        "--position",
        default="bottom",
        choices=["bottom", "top"],
        help="字幕位置（默认: bottom）",
    )
    parser.add_argument("--margin", type=int, default=30, help="字幕边距（默认: 30）")
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=23,
        choices=range(18, 29),
        metavar="18-28",
        help="视频质量 CRF 值，越小质量越高（默认: 23）",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="编码速度预设（默认: medium）",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="显示 FFmpeg 详细输出"
    )

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 视频文件不存在: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.subtitle):
        print(f"错误: 字幕文件不存在: {args.subtitle}")
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_file = args.output
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_file = os.path.join(args.output_dir, f"{base_name}_subtitled.mp4")
    else:
        base_name = os.path.splitext(args.input)[0]
        output_file = f"{base_name}_subtitled.mp4"

    # 获取视频信息
    duration = get_video_duration(args.input)
    video_info = get_video_info(args.input)

    print(f"\n{'=' * 60}")
    print(f"  字幕合并视频工具")
    print(f"{'=' * 60}")
    print(f"输入视频: {os.path.basename(args.input)}")
    print(f"字幕文件: {os.path.basename(args.subtitle)}")
    if video_info:
        print(f"视频分辨率: {video_info['width']}x{video_info['height']}")
    if duration > 0:
        print(f"视频时长: {int(duration // 60)}分{int(duration % 60)}秒")
    print(f"字幕字号: {args.font_size}")
    print(f"字幕位置: {args.position}")
    print(f"编码质量: CRF {args.quality}")
    print(f"编码速度: {args.preset}")
    print(f"输出文件: {output_file}")
    print(f"{'=' * 60}")

    # 构建 FFmpeg 命令 - 简化路径处理
    subtitle_path = os.path.abspath(args.subtitle)
    input_path = os.path.abspath(args.input)

    # 字幕位置
    alignment = 2 if args.position == "bottom" else 6

    # 字幕样式
    style = f"FontSize={args.font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline={args.outline},MarginV={args.margin},Alignment={alignment}"

    # 添加 setpts=PTS-STARTPTS 修复时间戳问题，防止因为时间戳错误导致画面黑屏丢帧
    subtitle_filter = (
        f"setpts=PTS-STARTPTS,subtitles={subtitle_path}:force_style='{style}'"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-fflags",
        "+genpts",  # 重新生成时间戳
        "-i",
        input_path,
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.quality),
        "-fps_mode",
        "auto",  # 防止异常丢帧
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-map",
        "0:v",
        "-map",
        "0:a?",
        output_file,
    ]

    print(f"\n开始合并...")
    start_time = time.time()

    # 运行 FFmpeg
    returncode = run_ffmpeg_with_progress(cmd, duration, args.verbose)

    if returncode == 0:
        elapsed = time.time() - start_time
        output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB

        print(f"\n{'=' * 60}")
        print(f"  完成!")
        print(f"  输出文件: {output_file}")
        print(f"  文件大小: {output_size:.1f} MB")
        print(f"  总耗时: {elapsed:.1f}秒 ({elapsed / 60:.1f}分钟)")
        print(f"{'=' * 60}")
    else:
        print(f"\n错误: FFmpeg 执行失败，返回码: {returncode}")
        print("请检查输入文件和字幕文件是否正确")
        sys.exit(1)


if __name__ == "__main__":
    main()
