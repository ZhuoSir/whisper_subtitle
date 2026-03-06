#!/usr/bin/env python3
"""
Whisper 字幕生成工具 Web UI
支持字幕生成、翻译、合并视频
"""

import os
import sys

# 禁用代理设置（解决 Gradio 启动问题）
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

import gradio as gr
import time
import tempfile
import subprocess
from pathlib import Path

# 导入核心模块
from faster_whisper import WhisperModel
from tqdm import tqdm

try:
    from deep_translator import GoogleTranslator
except ImportError:
    os.system("pip install deep-translator")
    from deep_translator import GoogleTranslator

# 全局变量
whisper_model = None
whisper_model_name = None
local_translator = None
local_tokenizer = None

# ============== 工具函数 ==============


def seconds_to_srt_time(seconds):
    """秒转 SRT 时间格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


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


def load_whisper_model(model_name, device="auto", compute_type="int8"):
    """加载 Whisper 模型"""
    global whisper_model, whisper_model_name

    if whisper_model is not None and whisper_model_name == model_name:
        return whisper_model

    try:
        whisper_model = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
        whisper_model_name = model_name
    except Exception as e:
        whisper_model = WhisperModel(model_name, device="cpu", compute_type="float32")
        whisper_model_name = model_name

    return whisper_model


def load_local_translator(source_lang="ja", target_lang="zh"):
    """加载本地翻译模型"""
    global local_translator, local_tokenizer

    if local_translator is not None:
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
    local_tokenizer = AutoTokenizer.from_pretrained(model_name)
    local_translator = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return local_translator, local_tokenizer


def translate_batch_local(texts, source_lang="ja", target_lang="zh"):
    """本地模型批量翻译"""
    model, tokenizer = load_local_translator(source_lang, target_lang)

    if model is None:
        return None

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    translated = model.generate(**inputs)
    results = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return results


def translate_batch_google(texts, target_lang="zh-CN"):
    """Google 批量翻译"""
    translator = GoogleTranslator(source="auto", target=target_lang)
    results = []

    combined = "\n||||\n".join(texts)
    try:
        translated = translator.translate(combined)
        if translated:
            parts = translated.split("\n||||\n")
            if len(parts) == len(texts):
                return [p.strip() for p in parts]
    except:
        pass

    # 回退到逐条翻译
    for text in texts:
        try:
            result = translator.translate(text)
            results.append(result if result else text)
        except:
            results.append(text)

    return results


# ============== 核心功能 ==============


def generate_subtitle(
    video_file,
    language,
    translate_target,
    translator_engine,
    model_choice,
    batch_size,
    progress=gr.Progress(),
):
    """生成字幕 - 流式输出"""

    if video_file is None:
        yield "❌ 请先上传视频文件", None, ""
        return

    video_path = video_file
    output_lines = []
    srt_content = []

    # 模型映射
    model_map = {
        "large-v3-turbo (推荐)": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "large-v3 (最准确)": "large-v3",
        "medium (平衡)": "medium",
        "small (较快)": "small",
        "base (最快)": "base",
    }
    model_name = model_map.get(
        model_choice, "deepdml/faster-whisper-large-v3-turbo-ct2"
    )

    # 语言映射
    lang_map = {
        "自动检测": None,
        "日语 (ja)": "ja",
        "中文 (zh)": "zh",
        "英语 (en)": "en",
    }
    lang = lang_map.get(language, None)

    # 翻译目标映射
    trans_map = {
        "不翻译": None,
        "中文 (zh)": "zh",
        "中文简体 (zh-CN)": "zh-CN",
        "英语 (en)": "en",
    }
    trans_target = trans_map.get(translate_target, None)

    try:
        # 步骤1: 加载模型
        output_lines.append("📦 加载 Whisper 模型中...")
        yield "\n".join(output_lines), None, ""

        model = load_whisper_model(model_name)
        output_lines.append("✅ 模型加载完成")
        yield "\n".join(output_lines), None, ""

        # 步骤2: 获取视频信息
        duration = get_video_duration(video_path)
        if duration > 0:
            output_lines.append(
                f"📹 视频时长: {int(duration // 60)}分{int(duration % 60)}秒"
            )
        yield "\n".join(output_lines), None, ""

        # 步骤3: 开始识别
        output_lines.append("\n🎤 开始语音识别...")
        yield "\n".join(output_lines), None, ""

        segments, info = model.transcribe(
            video_path,
            language=lang,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        detected_lang = info.language
        output_lines.append(
            f"🌐 识别语言: {detected_lang} (概率: {info.language_probability:.2f})"
        )
        yield "\n".join(output_lines), None, ""

        # 收集识别结果
        segment_list = []
        output_lines.append("\n📝 识别结果:")
        output_lines.append("-" * 50)

        for i, segment in enumerate(segments):
            segment_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )

            start_time = seconds_to_srt_time(segment.start)
            end_time = seconds_to_srt_time(segment.end)

            # 实时显示识别结果
            output_lines.append(f"[{start_time} --> {end_time}]")
            output_lines.append(f"  {segment.text.strip()}")

            # 更新进度
            if duration > 0:
                progress((segment.end / duration) * 0.6, desc="识别中...")

            yield "\n".join(output_lines), None, ""

        output_lines.append("-" * 50)
        output_lines.append(f"✅ 识别完成，共 {len(segment_list)} 条字幕")
        yield "\n".join(output_lines), None, ""

        # 步骤4: 翻译（如果需要）
        if trans_target and len(segment_list) > 0:
            output_lines.append(f"\n🌍 开始翻译 ({translator_engine})...")
            yield "\n".join(output_lines), None, ""

            texts = [s["text"] for s in segment_list]
            batch = int(batch_size)
            translated_texts = []

            for i in range(0, len(texts), batch):
                batch_texts = texts[i : i + batch]

                if translator_engine == "本地模型 (极速)":
                    source = detected_lang if detected_lang else "ja"
                    target = trans_target.replace("-CN", "").replace("-TW", "").lower()
                    results = translate_batch_local(batch_texts, source, target)
                    if results is None:
                        results = translate_batch_google(batch_texts, trans_target)
                else:
                    google_target = "zh-CN" if trans_target == "zh" else trans_target
                    results = translate_batch_google(batch_texts, google_target)

                translated_texts.extend(results)

                # 更新进度
                prog = 0.6 + (i / len(texts)) * 0.3
                progress(prog, desc="翻译中...")

            # 更新字幕文本并显示翻译结果
            output_lines.append("\n📝 翻译结果:")
            output_lines.append("-" * 50)

            for i, text in enumerate(translated_texts):
                if i < len(segment_list):
                    segment_list[i]["text"] = text
                    start_time = seconds_to_srt_time(segment_list[i]["start"])
                    end_time = seconds_to_srt_time(segment_list[i]["end"])
                    output_lines.append(f"[{start_time} --> {end_time}]")
                    output_lines.append(f"  {text}")

            output_lines.append("-" * 50)
            output_lines.append("✅ 翻译完成")
            yield "\n".join(output_lines), None, ""

        # 步骤5: 生成 SRT 文件
        progress(0.95, desc="生成字幕文件...")

        for i, seg in enumerate(segment_list, 1):
            start = seconds_to_srt_time(seg["start"])
            end = seconds_to_srt_time(seg["end"])
            srt_content.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")

        # 保存 SRT 文件
        video_name = Path(video_path).stem
        srt_path = os.path.join(tempfile.gettempdir(), f"{video_name}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

        output_lines.append(f"\n✅ 字幕文件已生成!")
        progress(1.0, desc="完成!")

        yield "\n".join(output_lines), srt_path, "\n".join(srt_content)

    except Exception as e:
        output_lines.append(f"\n❌ 错误: {str(e)}")
        yield "\n".join(output_lines), None, ""


def merge_subtitle_video(
    video_file, subtitle_file, font_size, position, quality, progress=gr.Progress()
):
    """合并字幕到视频"""

    if video_file is None:
        return "❌ 请上传视频文件", None

    if subtitle_file is None:
        return "❌ 请上传字幕文件", None

    video_path = video_file
    subtitle_path = subtitle_file

    # 输出路径
    video_name = Path(video_path).stem
    output_path = os.path.join(tempfile.gettempdir(), f"{video_name}_subtitled.mp4")

    # 字幕位置
    alignment = 2 if position == "底部" else 6
    margin = 30

    # 字幕样式
    style = f"FontSize={int(font_size)},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,MarginV={margin},Alignment={alignment}"

    # 转义字幕路径
    subtitle_path_escaped = subtitle_path.replace("'", "'\\''")
    subtitle_filter = f"subtitles='{subtitle_path_escaped}':force_style='{style}'"

    # FFmpeg 命令
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(int(quality)),
        "-c:a",
        "copy",
        output_path,
    ]

    try:
        # 获取视频时长
        duration = get_video_duration(video_path)

        # 启动 FFmpeg
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

        # 等待完成
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            return (
                f"✅ 视频合并完成!\n输出文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB",
                output_path,
            )
        else:
            return f"❌ 合并失败: {stderr}", None

    except Exception as e:
        return f"❌ 错误: {str(e)}", None


def one_click_process(
    video_file,
    language,
    translate_target,
    translator_engine,
    model_choice,
    batch_size,
    font_size,
    position,
    quality,
    progress=gr.Progress(),
):
    """一键处理：识别 + 翻译 + 合并"""

    if video_file is None:
        yield "❌ 请先上传视频文件", None, None, ""
        return

    output_lines = []
    srt_path = None
    video_output_path = None
    srt_content = ""

    # ====== 步骤1: 生成字幕 ======
    output_lines.append("=" * 50)
    output_lines.append("📌 步骤 1/2: 生成字幕")
    output_lines.append("=" * 50)
    yield "\n".join(output_lines), None, None, ""

    # 复用 generate_subtitle 的逻辑
    for result, srt, content in generate_subtitle(
        video_file,
        language,
        translate_target,
        translator_engine,
        model_choice,
        batch_size,
        progress,
    ):
        output_lines_temp = output_lines.copy()
        output_lines_temp.append(result)
        yield "\n".join(output_lines_temp), srt, None, content
        if srt:
            srt_path = srt
            srt_content = content

    if srt_path is None:
        output_lines.append("❌ 字幕生成失败，停止处理")
        yield "\n".join(output_lines), None, None, ""
        return

    # ====== 步骤2: 合并视频 ======
    output_lines.append("\n" + "=" * 50)
    output_lines.append("📌 步骤 2/2: 合并字幕到视频")
    output_lines.append("=" * 50)
    output_lines.append("🎬 开始合并...")
    yield "\n".join(output_lines), srt_path, None, srt_content

    merge_result, video_output_path = merge_subtitle_video(
        video_file, srt_path, font_size, position, quality, progress
    )

    output_lines.append(merge_result)
    output_lines.append("\n" + "=" * 50)
    output_lines.append("🎉 全部处理完成!")
    output_lines.append("=" * 50)

    yield "\n".join(output_lines), srt_path, video_output_path, srt_content


# ============== Web UI ==============


def create_ui():
    """创建 Gradio 界面"""

    with gr.Blocks(
        title="Whisper 字幕生成工具",
    ) as app:
        gr.Markdown("""
        # 🎬 Whisper 字幕生成工具
        
        支持视频语音识别、多语言翻译、字幕合并到视频
        """)

        with gr.Tabs():
            # ====== Tab 1: 字幕生成 ======
            with gr.TabItem("📝 字幕生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传视频")
                        video_input1 = gr.Video(label="上传视频文件")

                        gr.Markdown("### 参数设置")
                        language1 = gr.Dropdown(
                            choices=["自动检测", "日语 (ja)", "中文 (zh)", "英语 (en)"],
                            value="自动检测",
                            label="识别语言",
                        )
                        translate1 = gr.Dropdown(
                            choices=[
                                "不翻译",
                                "中文 (zh)",
                                "中文简体 (zh-CN)",
                                "英语 (en)",
                            ],
                            value="中文 (zh)",
                            label="翻译目标",
                        )
                        translator1 = gr.Radio(
                            choices=["本地模型 (极速)", "Google 翻译"],
                            value="本地模型 (极速)",
                            label="翻译引擎",
                        )
                        model1 = gr.Dropdown(
                            choices=[
                                "large-v3-turbo (推荐)",
                                "large-v3 (最准确)",
                                "medium (平衡)",
                                "small (较快)",
                                "base (最快)",
                            ],
                            value="large-v3-turbo (推荐)",
                            label="Whisper 模型",
                        )
                        batch1 = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=32,
                            step=1,
                            label="批量翻译大小",
                        )

                        btn_generate = gr.Button("🚀 开始生成字幕", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 实时输出")
                        output1 = gr.Textbox(
                            label="处理日志",
                            lines=20,
                            max_lines=30,
                            elem_classes="output-text",
                        )
                        srt_preview1 = gr.Textbox(
                            label="字幕预览", lines=10, max_lines=15
                        )
                        srt_download1 = gr.File(label="下载字幕文件")

                btn_generate.click(
                    fn=generate_subtitle,
                    inputs=[
                        video_input1,
                        language1,
                        translate1,
                        translator1,
                        model1,
                        batch1,
                    ],
                    outputs=[output1, srt_download1, srt_preview1],
                )

            # ====== Tab 2: 字幕合并 ======
            with gr.TabItem("🎬 字幕合并"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传文件")
                        video_input2 = gr.Video(label="上传视频文件")
                        subtitle_input2 = gr.File(
                            label="上传字幕文件 (SRT)", file_types=[".srt"]
                        )

                        gr.Markdown("### 字幕样式")
                        font_size2 = gr.Slider(
                            minimum=16, maximum=36, value=24, step=1, label="字幕字号"
                        )
                        position2 = gr.Radio(
                            choices=["底部", "顶部"], value="底部", label="字幕位置"
                        )
                        quality2 = gr.Slider(
                            minimum=18,
                            maximum=28,
                            value=23,
                            step=1,
                            label="视频质量 (CRF, 越小质量越高)",
                        )

                        btn_merge = gr.Button("🎬 开始合并", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 输出结果")
                        output2 = gr.Textbox(label="处理日志", lines=10)
                        video_download2 = gr.File(label="下载带字幕视频")

                btn_merge.click(
                    fn=merge_subtitle_video,
                    inputs=[
                        video_input2,
                        subtitle_input2,
                        font_size2,
                        position2,
                        quality2,
                    ],
                    outputs=[output2, video_download2],
                )

            # ====== Tab 3: 一键处理 ======
            with gr.TabItem("⚡ 一键处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传视频")
                        video_input3 = gr.Video(label="上传视频文件")

                        gr.Markdown("### 识别与翻译设置")
                        language3 = gr.Dropdown(
                            choices=["自动检测", "日语 (ja)", "中文 (zh)", "英语 (en)"],
                            value="自动检测",
                            label="识别语言",
                        )
                        translate3 = gr.Dropdown(
                            choices=[
                                "不翻译",
                                "中文 (zh)",
                                "中文简体 (zh-CN)",
                                "英语 (en)",
                            ],
                            value="中文 (zh)",
                            label="翻译目标",
                        )
                        translator3 = gr.Radio(
                            choices=["本地模型 (极速)", "Google 翻译"],
                            value="本地模型 (极速)",
                            label="翻译引擎",
                        )
                        model3 = gr.Dropdown(
                            choices=[
                                "large-v3-turbo (推荐)",
                                "large-v3 (最准确)",
                                "medium (平衡)",
                                "small (较快)",
                                "base (最快)",
                            ],
                            value="large-v3-turbo (推荐)",
                            label="Whisper 模型",
                        )
                        batch3 = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=32,
                            step=1,
                            label="批量翻译大小",
                        )

                        gr.Markdown("### 视频合并设置")
                        font_size3 = gr.Slider(
                            minimum=16, maximum=36, value=24, step=1, label="字幕字号"
                        )
                        position3 = gr.Radio(
                            choices=["底部", "顶部"], value="底部", label="字幕位置"
                        )
                        quality3 = gr.Slider(
                            minimum=18,
                            maximum=28,
                            value=23,
                            step=1,
                            label="视频质量 (CRF)",
                        )

                        btn_oneclick = gr.Button("⚡ 一键处理", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 实时输出")
                        output3 = gr.Textbox(
                            label="处理日志",
                            lines=20,
                            max_lines=30,
                            elem_classes="output-text",
                        )
                        srt_preview3 = gr.Textbox(label="字幕预览", lines=8)

                        gr.Markdown("### 下载文件")
                        with gr.Row():
                            srt_download3 = gr.File(label="下载字幕")
                            video_download3 = gr.File(label="下载视频")

                btn_oneclick.click(
                    fn=one_click_process,
                    inputs=[
                        video_input3,
                        language3,
                        translate3,
                        translator3,
                        model3,
                        batch3,
                        font_size3,
                        position3,
                        quality3,
                    ],
                    outputs=[output3, srt_download3, video_download3, srt_preview3],
                )

        gr.Markdown("""
        ---
        ### 使用说明
        
        1. **字幕生成**: 上传视频，选择语言和翻译选项，点击生成
        2. **字幕合并**: 上传视频和字幕文件，调整样式，点击合并
        3. **一键处理**: 自动完成识别、翻译、合并全流程
        
        ### 提示
        
        - 首次使用会下载 Whisper 模型（约 1.5GB）
        - 本地翻译模型首次也需下载（约 300MB）
        - 处理速度取决于视频长度和电脑性能
        """)

    return app


# ============== 主程序 ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whisper 字幕生成工具 Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认: 0.0.0.0）")
    parser.add_argument("--port", type=int, default=7860, help="端口（默认: 7860）")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")

    args = parser.parse_args()

    print(f"""
{"=" * 60}
  🎬 Whisper 字幕生成工具 Web UI
{"=" * 60}
  本地访问: http://127.0.0.1:{args.port}
  局域网访问: http://你的IP:{args.port}
{"=" * 60}
    """)

    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
    )
