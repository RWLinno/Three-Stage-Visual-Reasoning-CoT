#!/bin/bash
# 洗衣机旋钮状态识别系统 - 一键启动脚本
# 支持EAS API和本地部署模式

set -e  # 任何命令失败时退出
set -o pipefail  # 管道中任何命令失败时也退出

# EAS配置
export EAS_BASE_URL="xx"
export EAS_TOKEN="xx"
export EAS_MODEL_NAME="xx"

# 输入配置
export INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:-./examples/washer_knob}"
# 简化的用户问题 - 核心判断规则应在CoT第一阶段完成
export INPUT_QUESTION="${INPUT_QUESTION:-判断洗衣机当前的档位}"

# 输出配置
export OUTPUT_DIR="${OUTPUT_DIR:-./output/washer_knob}"
export OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/results.jsonl}"
export SAVE_INTERMEDIATE_IMAGES="${SAVE_INTERMEDIATE_IMAGES:-true}"  # 保存中间推理图像

# 性能配置
export NUM_PROCESSORS="${NUM_PROCESSORS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export MAX_TOKENS="${MAX_TOKENS:-1024}"
export TIMEOUT="${TIMEOUT:-300}"

# 其他配置
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export DEBUG_MODE="${DEBUG_MODE:-false}"


# ================== 自动配置区域 ==================
# 获取项目根目录
PROJECT_ROOT=$(pwd)

# 创建输出目录
mkdir -p "${OUTPUT_DIR:-./output}"
mkdir -p "${OUTPUT_DIR}/intermediate_images" 2>/dev/null || true
mkdir -p "${OUTPUT_DIR}/logs" 2>/dev/null || true

# 日志文件
LOG_FILE="${OUTPUT_DIR}/logs/run_$(date +%Y%m%d_%H%M%S).log"
touch "${LOG_FILE}"

# 验证EAS_TOKEN
if [ -z "$EAS_TOKEN" ]; then
    echo "警告: EAS_TOKEN 未设置，尝试使用本地模型模式"
    export USE_LOCAL_MODEL=true
else
    export USE_LOCAL_MODEL=false
    echo "使用EAS API模式"
fi

# 检查输入目录
if [ ! -d "${INPUT_IMAGE_DIR}" ]; then
    echo "错误: 输入图像目录不存在: ${INPUT_IMAGE_DIR}"
    exit 1
fi

# 验证图像文件
IMAGE_COUNT=$(find "${INPUT_IMAGE_DIR}" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.PNG" \) | wc -l)
if [ "${IMAGE_COUNT}" -eq 0 ]; then
    echo "错误: 目录中没有找到图像文件 (支持: .jpg, .jpeg, .png, .PNG)"
    echo "请检查目录: ${INPUT_IMAGE_DIR}"
    exit 1
fi

echo "找到 ${IMAGE_COUNT} 个图像文件，开始处理..."

# ================== 执行区域 ==================
# 设置Python环境
# 优先使用虚拟环境中的Python
if [ -n "${VIRTUAL_ENV}" ]; then
    # 在虚拟环境中，优先使用 python（虚拟环境中通常 python 指向正确的解释器）
    if command -v python &> /dev/null; then
        PYTHON_EXEC="python"
    elif command -v python3 &> /dev/null; then
        PYTHON_EXEC="python3"
    else
        echo "错误: 虚拟环境中未找到 Python"
        exit 1
    fi
    echo "检测到虚拟环境: ${VIRTUAL_ENV}"
elif command -v python &> /dev/null; then
    PYTHON_EXEC="python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="python3"
else
    echo "错误: Python 未安装或不在PATH中"
    exit 1
fi

# 检查Python版本
if ! command -v "${PYTHON_EXEC}" &> /dev/null; then
    echo "错误: Python 未安装或不在PATH中"
    exit 1
fi

# 检查依赖
if [ ! -f "requirements.txt" ]; then
    echo "警告: requirements.txt 不存在，跳过依赖检查"
else
    echo "检查Python依赖..."
    ${PYTHON_EXEC} -c "import pkgutil; import sys; sys.exit(0 if pkgutil.find_loader('requests') else 1)" 2>/dev/null || {
        echo "安装依赖..."
        # 使用当前Python解释器对应的pip
        ${PYTHON_EXEC} -m pip install -r requirements.txt --quiet
    }
fi

# 设置并发环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 记录配置（简化输出）
echo "配置: 图像=${IMAGE_COUNT}, 处理器=${NUM_PROCESSORS}, 超时=${TIMEOUT}s" | tee -a "${LOG_FILE}"

# 准备调试和Python路径参数
DEBUG_ARG=""
# 使用兼容的方式检查DEBUG_MODE（支持大小写不敏感）
DEBUG_MODE_LOWER=$(echo "${DEBUG_MODE}" | tr '[:upper:]' '[:lower:]')
if [ "${DEBUG_MODE_LOWER}" = "true" ]; then
    DEBUG_ARG="--debug"
fi

# 设置Python路径，确保能找到src模块
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 执行主程序
echo "开始处理图像..."
START_TIME=$(date +%s)

# 执行Python脚本并捕获退出码
PYTHON_EXIT_CODE=0
${PYTHON_EXEC} scripts/washer_knob_analyzer.py \
    --image_dir "${INPUT_IMAGE_DIR}" \
    --question "${INPUT_QUESTION}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_jsonl "${OUTPUT_JSONL}" \
    --num_processors "${NUM_PROCESSORS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_tokens "${MAX_TOKENS}" \
    --timeout "${TIMEOUT}" \
    --log_level "${LOG_LEVEL}" \
    --save_intermediate_images "${SAVE_INTERMEDIATE_IMAGES}" \
    --eas_url "${EAS_BASE_URL}" \
    --eas_token "${EAS_TOKEN}" \
    --model_name "${EAS_MODEL_NAME}" \
    ${DEBUG_ARG} \
    2>&1 | tee -a "${LOG_FILE}" || PYTHON_EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 检查Python脚本是否失败
if [ ${PYTHON_EXIT_CODE} -ne 0 ]; then
    echo "错误: Python脚本执行失败，退出码: ${PYTHON_EXIT_CODE}" | tee -a "${LOG_FILE}"
    echo "这可能是由于API调用错误导致的，请检查日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    exit ${PYTHON_EXIT_CODE}
fi

# 检查结果文件是否生成
if [ ! -f "${OUTPUT_JSONL}" ] || [ ! -s "${OUTPUT_JSONL}" ]; then
    echo "错误: 结果文件未生成或为空: ${OUTPUT_JSONL}" | tee -a "${LOG_FILE}"
    echo "这可能是由于API调用错误导致的，请检查日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi

# 检查日志中是否有真正的API调用错误（排除VLM返回内容中的Error字符串）
# 只检测HTTP状态码错误和连接错误，不检测VLM返回内容中的"Error"字符串
if grep -qiE "(API请求失败|API request failed|connection.*error|requests\.exceptions|HTTPError|状态码: (401|403|500|502|503|504))" "${LOG_FILE}" 2>/dev/null; then
    # 进一步检查：如果结果文件存在且有成功记录，说明只是部分失败，不退出
    if [ -f "${OUTPUT_JSONL}" ] && [ -s "${OUTPUT_JSONL}" ]; then
        SUCCESS_COUNT=$(grep -c '"success":\s*true' "${OUTPUT_JSONL}" 2>/dev/null || echo 0)
        if [ "${SUCCESS_COUNT}" -gt 0 ]; then
            echo "警告: 检测到部分API调用错误，但已有成功结果，继续执行" | tee -a "${LOG_FILE}"
        else
            echo "错误: 检测到API调用错误且无成功结果，脚本将退出" | tee -a "${LOG_FILE}"
            echo "请检查日志文件以获取详细信息: ${LOG_FILE}" | tee -a "${LOG_FILE}"
            exit 1
        fi
    else
        echo "错误: 检测到API调用错误且结果文件为空，脚本将退出" | tee -a "${LOG_FILE}"
        echo "请检查日志文件以获取详细信息: ${LOG_FILE}" | tee -a "${LOG_FILE}"
        exit 1
    fi
fi

# 生成HTML报告
HTML_REPORT="${OUTPUT_DIR}/report.html"
if [ -f "scripts/generate_html_report.py" ]; then
    echo "生成HTML报告..."
${PYTHON_EXEC} scripts/generate_html_report.py \
    --results_jsonl "${OUTPUT_JSONL}" \
    --output_html "${HTML_REPORT}" \
    --image_dir "${INPUT_IMAGE_DIR}" \
    --intermediate_dir "${OUTPUT_DIR}/intermediate_images" \
        2>&1 | tee -a "${LOG_FILE}" || true
    echo "HTML报告: ${HTML_REPORT}"
else
    echo "警告: generate_html_report.py 不存在，跳过HTML报告生成"
fi

# 统计结果
if [ -f "${OUTPUT_JSONL}" ]; then
    RESULT_COUNT=$(wc -l < "${OUTPUT_JSONL}" 2>/dev/null || echo 0)
    echo "处理完成: ${RESULT_COUNT}个结果, 耗时${DURATION}秒" | tee -a "${LOG_FILE}"
    echo "结果: ${OUTPUT_JSONL}" | tee -a "${LOG_FILE}"
    [ -f "${HTML_REPORT}" ] && echo "报告: ${HTML_REPORT}" | tee -a "${LOG_FILE}"
else
    echo "警告: 结果文件未生成: ${OUTPUT_JSONL}" | tee -a "${LOG_FILE}"
fi