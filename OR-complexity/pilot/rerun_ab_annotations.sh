#!/usr/bin/env bash
# =============================================================================
# 一键重跑：用更新后的 semantic_annotator_v1.txt 重新标注 gpt-5.6-sol 的 25 个
# 题目 (OR-complexity/pilot/ai_annotations/gpt-5.6-sol)。
#
# 特性：
#   * 幂等 / 断点续跑：脚本本身有文件缓存，已完成的题目不会重复调 API；
#   * 网络不可用时不消耗 API 配额，提示后直接退出，网络恢复后重跑本脚本即可；
#   * 每次运行前自动做前置校验（venv / .env / 输入 staging / 备份 / dry-run）。
#
# 用法：
#   bash OR-complexity/pilot/rerun_ab_annotations.sh
# =============================================================================
set -uo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
SCRIPT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../OR-complexity/pilot
OR_DIR="$(cd "$SCRIPT_SRC/.." && pwd)"                               # .../OR-complexity
WORKSPACE_ROOT="$(cd "$OR_DIR/.." && pwd)"                           # .../RuihaoZhu_Cornell

PY="$WORKSPACE_ROOT/.venv/bin/python"
RUNNER="$OR_DIR/pilot/run_ab_annotation.py"
ENV_FILE="$WORKSPACE_ROOT/.env"
INPUTS=/tmp/ab_inputs_25
OUT="$OR_DIR/pilot/ai_annotations"
MODEL="gpt-5.6-sol"
EXPECTED_ANCHORS=25

ANCHORS=(
  IndustryOR_22 IndustryOR_50
  IndustryOR_Easy_077_machine_assignment IndustryOR_Easy_092_bus_rental
  IndustryOR_Hard_026_shift_scheduling IndustryOR_Hard_036_vrp
  LEAN_Mixture6_truck_scheduling LEAN_TP8_transportation
  MIPLIB_NL_flugpl_fleet MIPLIB_NL_graph20_20_cuts
  NL4OPT_000_oil_spill_transport NL4OPT_013_radiation_beams
  OPTEngine_BinPacking OPTEngine_Inventory_5 OPTEngine_Inventory_5_1
  OPTEngine_Inventory_5_2_aug OPTEngine_Inventory_5_3_aug
  OPTEngine_JobShop OPTEngine_Knapsack_10 OPTEngine_Knapsack_10_1
  OPTEngine_Knapsack_10_1_aug OPTEngine_Knapsack_10_2
  OPTEngine_Knapsack_10_3_aug OPTEngine_TSP_4 OPTEngine_TSP_4_6_aug
)

say()  { printf '\n[rerun] %s\n' "$*"; }
die()  { printf '\n[rerun] ERROR: %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1. 前置检查
# -----------------------------------------------------------------------------
say "==> [1/6] 前置检查"
[ -x "$PY" ] || die "找不到 venv python: $PY (请确认 workspace 根目录 .venv 存在)"
[ -f "$RUNNER" ] || die "找不到标注脚本: $RUNNER"
[ -f "$ENV_FILE" ] || die "找不到 .env: $ENV_FILE"
grep -qE '^OPENROUTER_API_KEY=' "$ENV_FILE" \
  && say "     OPENROUTER_API_KEY 存在于 $ENV_FILE" \
  || die ".env 缺少 OPENROUTER_API_KEY"

# 校验新 prompt 已生效
PROMPT="$OR_DIR/operational_complexity/prompts/semantic_annotator_v1.txt"
grep -q "Semantic Obligation Family" "$PROMPT" \
  && say "     新 prompt 已生效 (semantic_annotator_v1.txt, SOH-1.1)" \
  || die "prompt 似乎不是 SOH-1.1 版本 (缺少 'Semantic Obligation Family' 关键词)"

# -----------------------------------------------------------------------------
# 2. staging 25 个输入（幂等：已存在则跳过）
# -----------------------------------------------------------------------------
say "==> [2/6] 准备 25 个输入文件 -> $INPUTS"
mkdir -p "$INPUTS/OPTEngine-augmented"
staged=0
for id in "${ANCHORS[@]}"; do
  src="$(find "$OR_DIR/pilot/inputs" -name "$id.json" 2>/dev/null | head -1)"
  [ -n "$src" ] || die "找不到输入: $id.json"
  if echo "$src" | grep -q "OPTEngine-augmented"; then
    dst="$INPUTS/OPTEngine-augmented/$id.json"
  else
    dst="$INPUTS/$id.json"
  fi
  if [ ! -f "$dst" ]; then cp "$src" "$dst"; fi
  staged=$((staged + 1))
done
say "     staging 完成: $staged 个文件"

# -----------------------------------------------------------------------------
# 3. 备份旧标注（幂等：已有备份则跳过）
# -----------------------------------------------------------------------------
say "==> [3/6] 备份旧标注"
OUT_DIR="$OUT/$MODEL"
BACKUP="$OUT/gpt-5.6-sol_v1_old"
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  if [ -d "$BACKUP" ] && [ -n "$(ls -A "$BACKUP" 2>/dev/null)" ]; then
    say "     检测到输出目录已有内容且备份已存在 -> 视为断点续跑，不覆盖备份"
  else
    mv "$OUT_DIR" "$BACKUP"
    mkdir -p "$OUT_DIR"
    say "     旧标注已备份到 $BACKUP"
  fi
else
  mkdir -p "$OUT_DIR"
  say "     输出目录为空（或备份已就绪），无需再次备份"
fi

# -----------------------------------------------------------------------------
# 4. 网络连通性检测（不可用则退出，不消耗配额）
# -----------------------------------------------------------------------------
say "==> [4/6] 检测网络连通性 (openrouter.ai)"
# 显式绕过本机代理 (--noproxy '*')，避免 403 ProxyError；000 表示完全连不上
code="$(curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' --max-time 20 \
        https://openrouter.ai/api/v1/models 2>/dev/null)"
if [ "$code" = "000" ] || [ -z "$code" ]; then
  die "网络不可用 (curl 无法连接 openrouter.ai, code=$code)。\
请恢复外网/代理后重新运行本脚本；API 配额未消耗。"
fi
say "     网络可用 (HTTP $code)"

# -----------------------------------------------------------------------------
# 5. dry-run 验证选中数量
# -----------------------------------------------------------------------------
say "==> [5/6] dry-run 验证 anchor 选择"
DRY_OUT="$(mktemp -d /tmp/ab_dry_XXXX)"
"$PY" "$RUNNER" --dry-run --models "$MODEL" --inputs "$INPUTS" --out "$DRY_OUT" >/dev/null 2>&1 \
  || die "dry-run 失败"
n_sel="$(ls "$DRY_OUT/_dry_run_prompts/" 2>/dev/null | sed 's/_SYSTEM.txt//;s/_USER.txt//' | sort -u | wc -l | tr -d ' ')"
rm -rf "$DRY_OUT"
if [ "$n_sel" != "$EXPECTED_ANCHORS" ]; then
  die "dry-run 选中 $n_sel 个 anchor（期望 $EXPECTED_ANCHORS），中止"
fi
say "     dry-run 选中 $n_sel 个 anchor，符合预期"

# -----------------------------------------------------------------------------
# 6. 正式运行
# -----------------------------------------------------------------------------
say "==> [6/6] 正式标注 (model=$MODEL, anchors=$n_sel)"
cd "$OR_DIR"
"$PY" "$RUNNER" --models "$MODEL" --inputs "$INPUTS" --out "$OUT"
rc=$?
[ $rc -ne 0 ] && die "标注运行失败 (exit=$rc)，网络恢复后可直接重跑本脚本续跑"

# -----------------------------------------------------------------------------
# 校验输出
# -----------------------------------------------------------------------------
say "==> 校验输出"
n_out="$(ls "$OUT/$MODEL/"*.txt 2>/dev/null | wc -l | tr -d ' ')"
say "     输出文件数: $n_out / $EXPECTED_ANCHORS"
ok_count="$(grep -c '"status": "ok"' "$OUT/annotation_status.json" 2>/dev/null || echo 0)"
fail_count="$(grep -c '"status": "ok"' "$OUT/annotation_status.json" 2>/dev/null \
  && echo " (see annotation_status.json)" || true)"
say "     annotation_status.json: $ok_count 条 ok"
[ -f "$OUT/pilot_metric_table.csv" ] \
  && say "     metric 表已生成: $OUT/pilot_metric_table.csv ($(wc -l < "$OUT/pilot_metric_table.csv" | tr -d ' ') 行)" \
  || say "     警告: 未生成 metric 表（可能所有标注均未通过校验）"

say "完成。新标注位于 $OUT/$MODEL/；旧版在 $BACKUP/（如需回滚可换回）"
