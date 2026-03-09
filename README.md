# Whisper 字幕生成工具集

一套完整的视频字幕生成、翻译、合并工具。支持命令行和 Web UI 两种使用方式。

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装（确保你已安装 FFmpeg）：

```bash
pip install faster-whisper deep-translator tqdm av transformers sentencepiece gradio==4.44.0 huggingface_hub==0.24.0 "httpx[socks]" mlx-whisper
```

### 启动 Web UI（推荐）

```bash
python webui.py --share
```

访问地址：`http://127.0.0.1:7860`

---

## 脚本列表

| 脚本 | 说明 | 特点 |
|------|------|------|
| `webui.py` | **Web UI 界面** | 可视化操作，推荐使用 |
| `whisper_subtitle.py` | 基础版 | 简单易用 |
| `whisper_subtitle_gpu.py` | GPU加速版 | 速度快 |
| `whisper_subtitle_turbo.py` | 极速版 | large-v3-turbo + 批量翻译 |
| `whisper_subtitle_pro.py` | Pro版 | 本地翻译模型，无网络延迟 |
| `whisper_subtitle_hd.py` | HD高清版 | 包含音频清洗与高精度解码 |
| `whisper_subtitle_mlx.py` | MLX极速版 | **Apple Silicon 专属**，速度最快 |
| `merge_subtitle.py` | 字幕合并 | 将字幕硬编码到视频 |

---

## 0. webui.py（Web UI 界面 - 推荐）

可视化操作界面，支持字幕生成、翻译、合并视频，实时显示翻译字幕。

### 启动方式

```bash
# 本地访问
python webui.py

# 生成公网分享链接（推荐，解决代理问题）
python webui.py --share

# 指定端口
python webui.py --port 8080
```

### 访问地址

- 本地: `http://127.0.0.1:7860`
- 公网: 启动时显示的 share 链接

### 功能模块

| Tab | 功能 | 说明 |
|-----|------|------|
| 📝 字幕生成 | 语音识别 + 翻译 | 实时显示识别/翻译结果 |
| 🎬 字幕合并 | 硬编码字幕到视频 | 自定义字号、位置 |
| ⚡ 一键处理 | 识别 + 翻译 + 合并 | 全自动处理 |

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 监听地址 | 127.0.0.1 |
| `--port` | 端口 | 7860 |
| `--share` | 生成公网分享链接 | 关闭 |

### 常见问题

如果遇到代理问题导致无法启动，使用：

```bash
python webui.py --share
```

---

## 1. whisper_subtitle.py（基础版）

### 使用方法

```bash
# 基本用法
python whisper_subtitle.py /path/to/video.mp4

# 指定输出目录和翻译
python whisper_subtitle.py "/path/to/video.mp4" -d /output/dir -t zh-CN

# 使用更大模型
python whisper_subtitle.py "/path/to/video.mp4" -m large
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 视频/音频文件路径 | 必填 |
| `-o, --output` | 输出字幕文件路径 | 与输入文件同名.srt |
| `-d, --dir` | 输出目录 | 与输入文件同目录 |
| `-m, --model` | 模型: tiny/base/small/medium/large | small |
| `-l, --language` | 识别语言: zh/en/ja | 自动检测 |
| `-t, --translate` | 翻译目标语言: zh-CN | 无 |
| `--compute-type` | 计算类型: float16/float32/int8 | float32 |

---

## 2. whisper_subtitle_gpu.py（GPU加速版）

### 使用方法

```bash
# GPU加速 + int8量化
python whisper_subtitle_gpu.py "/path/to/video.mp4" -d /output/dir -t zh-CN

# 指定模型
python whisper_subtitle_gpu.py "/path/to/video.mp4" -m large-v3 -t zh-CN
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 视频/音频文件路径 | 必填 |
| `-o, --output` | 输出字幕文件路径 | 自动生成 |
| `-d, --dir` | 输出目录 | 当前目录 |
| `-m, --model` | 模型大小 | large-v3 |
| `-l, --language` | 识别语言 | 自动检测 |
| `-t, --translate` | 翻译目标语言 | 无 |
| `--device` | 计算设备: auto/cpu/cuda | auto |
| `--compute-type` | 计算类型: int8/float16/float32 | int8 |
| `--beam-size` | 束搜索大小 | 1 |

---

## 3. whisper_subtitle_turbo.py（极速版）

使用 large-v3-turbo 模型，速度比 large-v3 快 4 倍。

### 使用方法

```bash
# 极速版（推荐）
python whisper_subtitle_turbo.py "/path/to/video.mp4" -d /output/dir -t zh-CN -p

# 调整批量翻译大小
python whisper_subtitle_turbo.py "/path/to/video.mp4" -d /output/dir -t zh-CN --batch-size 30 -p
```

### 完整命令示例

```bash
python whisper_subtitle_turbo.py "/path/to/video.mp4" -d /output/dir -t zh-CN --batch-size 30 -p
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 视频/音频文件路径 | 必填 |
| `-o, --output` | 输出字幕文件路径 | 自动生成 |
| `-d, --dir` | 输出目录 | 当前目录 |
| `-m, --model` | 模型名称 | large-v3-turbo |
| `-l, --language` | 识别语言: zh/en/ja | 自动检测 |
| `-t, --translate` | 翻译目标语言: zh-CN | 无 |
| `--device` | 计算设备 | auto |
| `--compute-type` | 计算类型 | int8 |
| `--batch-size` | 批量翻译条数 | 20 |
| `-p, --print` | 打印字幕到控制台 | 关闭 |

---

## 4. whisper_subtitle_pro.py（Pro版 - 本地翻译）

使用本地翻译模型，无网络延迟，适合日语翻译中文。

### 使用方法

```bash
# 日语视频翻译中文（本地模型）
python whisper_subtitle_pro.py "/path/to/video.mp4" -l ja -t zh --translator local -p

# 使用 Google 翻译（备选）
python whisper_subtitle_pro.py "/path/to/video.mp4" -l ja -t zh-CN --translator google -p
```

### 完整命令示例

```bash
python whisper_subtitle_pro.py "/path/to/video.mp4" -d /output/dir -l ja -t zh --translator local --batch-size 32 -p
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 视频/音频文件路径 | 必填 |
| `-o, --output` | 输出字幕文件路径 | 自动生成 |
| `-d, --dir` | 输出目录 | 当前目录 |
| `-m, --model` | Whisper 模型 | large-v3-turbo |
| `-l, --language` | 识别语言: zh/en/ja | 自动检测 |
| `-t, --translate` | 翻译目标: zh/en | 无 |
| `--translator` | 翻译引擎: local/google | local |
| `--batch-size` | 批量翻译大小 | 32 |
| `-p, --print` | 打印字幕到控制台 | 关闭 |

### 支持的翻译语言对

| 源语言 | 目标语言 | 模型 |
|--------|----------|------|
| ja (日语) | zh (中文) | larryvrh/mt5-translation-ja_zh |
| ja (日语) | en (英语) | Helsinki-NLP/opus-mt-ja-en |
| en (英语) | zh (中文) | Helsinki-NLP/opus-mt-en-zh |
| en (英语) | ja (日语) | Helsinki-NLP/opus-mt-en-jap |
| zh (中文) | en (英语) | Helsinki-NLP/opus-mt-zh-en |

---

## 5. whisper_subtitle_hd.py（HD 高清增强版）

专注于**最高准确率**和**声音清晰度**。通过 FFmpeg 对音频进行降噪、音量标准化处理，并使用 Whisper 最高精度参数进行解码。

### 使用方法

```bash
# 默认使用高精度处理
python whisper_subtitle_hd.py "/path/to/video.mp4" -d /output/dir -t zh-CN -p

# 关闭音频增强预处理（仅保留高精度解码）
python whisper_subtitle_hd.py "/path/to/video.mp4" --no-preprocess
```

### 参数说明

新增参数：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--no-preprocess` | 禁用音频降噪与预处理 | 否 |

---

## 6. whisper_subtitle_mlx.py (MLX 极速版 - Apple Silicon 专属)

为 Apple Silicon (M1/M2/M3) 芯片深度优化，利用 MLX 框架实现极致识别速度。

### 使用方法

```bash
python whisper_subtitle_mlx.py "/path/to/video.mp4" -d /output/dir -l ja -t zh -p
```

### 完整命令示例

```bash
python /Users/bryanchen/Documents/work/opencode/whisper_subtitle_mlx.py "/Users/bryanchen/Documents/图片/SMA-692.mp4" -d /Users/bryanchen/Documents/work/opencode -l ja -t zh --translator local -p
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | MLX 模型 | `mlx-community/whisper-large-v3-mlx` |
| `--no-preprocess` | 禁用音频预处理 | 否 |

---

## 7. merge_subtitle.py（字幕合并视频）

将 SRT 字幕硬编码到视频中，生成带字幕的新视频。

### 使用方法

```bash
# 基本用法
python merge_subtitle.py "/path/to/video.mp4" -s "/path/to/subtitle.srt" -d /output/dir

# 自定义字幕样式
python merge_subtitle.py "/path/to/video.mp4" -s "/path/to/subtitle.srt" --font-size 28 --position bottom
```

### 完整命令示例

```bash
python merge_subtitle.py "/path/to/video.mp4" -s "/path/to/video.srt" -d /output/dir --font-size 24 --position bottom
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频路径 | 必填 |
| `-s, --subtitle` | SRT 字幕文件 | 必填 |
| `-o, --output` | 输出视频路径 | 自动生成 |
| `-d, --dir` | 输出目录 | 视频同目录 |
| `--font-size` | 字幕字号 | 24 |
| `--font-color` | 字幕颜色 | white |
| `--outline` | 描边宽度 | 2 |
| `--position` | 位置: bottom/top | bottom |
| `--margin` | 边距 | 30 |
| `-q, --quality` | 质量 CRF (18-28) | 23 |
| `--preset` | 编码速度 | medium |

### 输出文件

输出文件名为：`原文件名_subtitled.mp4`

---

## 模型选择参考

| 模型 | 大小 | 速度 | 准确率 | 推荐场景 |
|------|------|------|--------|----------|
| tiny | ~75MB | 最快 | 低 | 测试 |
| base | ~140MB | 快 | 中 | 快速预览 |
| small | ~500MB | 中 | 较高 | 日常使用 |
| medium | ~1.5GB | 慢 | 高 | 重要内容 |
| large-v3 | ~3GB | 很慢 | 最高 | 最高要求 |
| large-v3-turbo | ~1.5GB | 快 | 高 | **推荐** |

---

## 模型存储位置

下载的模型存储在：

```
~/.cache/huggingface/hub/
```

查看已下载模型：

```bash
ls -lh ~/.cache/huggingface/hub/
```

清理模型缓存：

```bash
rm -rf ~/.cache/huggingface/hub/
```

---

## 完整工作流示例

### 方式一：使用 Web UI（推荐）

```bash
python webui.py --share
```

然后在浏览器中操作。

### 方式二：使用命令行

#### 1. 生成字幕

```bash
python whisper_subtitle_pro.py "/path/to/video.mp4" -d /output/dir -l ja -t zh --translator local -p
```

#### 2. 合并字幕到视频

```bash
python merge_subtitle.py "/path/to/video.mp4" -s "/output/dir/video.srt" -d /output/dir
```

---

## 常见问题

### 1. Web UI 无法启动

使用 `--share` 参数：

```bash
python webui.py --share
```

### 2. 翻译速度慢

使用 `--translator local` 本地翻译模型，或增大 `--batch-size`。

### 3. 识别不准确

使用更大的模型：`-m large-v3` 或 `-m large-v3-turbo`。

### 4. GPU 不支持 float16

Mac M1/M2/M3 用户使用 `--compute-type float32` 或 `int8`。

### 5. 字幕位置不对

调整 `--margin` 和 `--position` 参数。

### 6. 依赖安装问题

确保使用正确的版本：

```bash
pip install gradio==4.44.0 huggingface_hub==0.24.0 "httpx[socks]"
```

---

## 项目结构

```
opencode/
├── README.md                    # 使用文档
├── requirements.txt             # 依赖列表
├── webui.py                     # Web UI 界面
├── whisper_subtitle.py          # 基础版
├── whisper_subtitle_gpu.py      # GPU加速版
├── whisper_subtitle_turbo.py    # 极速版
├── whisper_subtitle_pro.py      # Pro版（本地翻译）
└── merge_subtitle.py            # 字幕合并视频
```

---

## License

MIT License
