#!/usr/bin/env bash
set -uo pipefail

log_file="${LOG_FILE:-out/forge.log}"
checkpoint="${MODEL:-out/forge-98m.bin}"
tokens_per_update="${TOKENS_PER_UPDATE:-32768}"
hourly_rate="${HOURLY_RATE:-0.44977777777777783}"
display_timezone="${DISPLAY_TZ:-PST8PDT}"

if [[ -t 1 ]]; then
    bold=$'\033[1m'
    dim=$'\033[2m'
    green=$'\033[32m'
    yellow=$'\033[33m'
    red=$'\033[31m'
    cyan=$'\033[36m'
    reset=$'\033[0m'
else
    bold="" dim="" green="" yellow="" red="" cyan="" reset=""
fi

format_duration() {
    local seconds=${1:-0}
    local days=$((seconds / 86400))
    local hours=$(((seconds % 86400) / 3600))
    local minutes=$(((seconds % 3600) / 60))
    if ((days > 0)); then
        printf '%dd %02dh %02dm' "$days" "$hours" "$minutes"
    elif ((hours > 0)); then
        printf '%dh %02dm' "$hours" "$minutes"
    else
        printf '%dm' "$minutes"
    fi
}

elapsed_to_seconds() {
    local value=${1:-0:00:00}
    local hours minutes seconds
    IFS=: read -r hours minutes seconds <<< "$value"
    printf '%d' "$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))"
}

bar() {
    local current=$1 total=$2 width=${3:-42}
    local filled=$((current * width / total))
    local empty=$((width - filled))
    printf '%s' "$green"
    printf '%*s' "$filled" '' | tr ' ' '#'
    printf '%s' "$dim"
    printf '%*s' "$empty" '' | tr ' ' '-'
    printf '%s' "$reset"
}

if [[ ! -f "$log_file" ]]; then
    printf '%serror:%s training log not found: %s\n' "$red" "$reset" "$log_file" >&2
    exit 1
fi

last_step_line=$(awk '/^step +[0-9]+\/[0-9]+/ { line=$0 } END { print line }' "$log_file")
if [[ -z "$last_step_line" ]]; then
    printf '%sForge training is initializing.%s No optimizer update has been logged yet.\n' "$yellow" "$reset"
    tail -12 "$log_file"
    exit 0
fi

step=$(awk '{ split($2, n, "/"); print n[1] }' <<< "$last_step_line")
total=$(awk '{ split($2, n, "/"); print n[2] }' <<< "$last_step_line")
lr=$(awk '{ for (i=1; i<=NF; i++) if ($i=="lr") print $(i+1) }' <<< "$last_step_line")
loss=$(awk '{ for (i=1; i<=NF; i++) if ($i=="loss") print $(i+1) }' <<< "$last_step_line")
tps_text=$(awk '{ for (i=1; i<=NF; i++) if ($i=="tok/s") print $(i-1) }' <<< "$last_step_line")
tps=${tps_text//,/}
elapsed=$(sed -nE 's/.*\(([0-9]+:[0-9]{2}:[0-9]{2})\)$/\1/p' <<< "$last_step_line")

last_val_line=$(awk '/^step +[0-9]+\/[0-9]+/ && / val / { line=$0 } END { print line }' "$log_file")
val_step="n/a"
val_loss="n/a"
if [[ -n "$last_val_line" ]]; then
    val_step=$(awk '{ split($2, n, "/"); print n[1] }' <<< "$last_val_line")
    val_loss=$(awk '{ for (i=1; i<=NF; i++) if ($i=="val") print $(i+1) }' <<< "$last_val_line")
fi

processed_tokens=$((step * tokens_per_update))
total_tokens=$((total * tokens_per_update))
remaining_updates=$((total - step))
percent=$(awk -v s="$step" -v t="$total" 'BEGIN { printf "%.2f", 100*s/t }')
processed_human=$(awk -v n="$processed_tokens" 'BEGIN { if (n>=1e9) printf "%.3fB",n/1e9; else printf "%.1fM",n/1e6 }')
total_human=$(awk -v n="$total_tokens" 'BEGIN { if (n>=1e9) printf "%.3fB",n/1e9; else printf "%.1fM",n/1e6 }')

eta_seconds=0
finish_time="calculating"
remaining_cost="n/a"
projected_cost="n/a"
if [[ "$tps" =~ ^[0-9]+$ ]] && ((tps > 0)); then
    eta_seconds=$((remaining_updates * tokens_per_update / tps))
    finish_time=$(TZ="$display_timezone" date -d "+${eta_seconds} seconds" '+%a %b %d at %I:%M %p %Z' 2>/dev/null || echo unavailable)
    remaining_cost=$(awk -v s="$eta_seconds" -v r="$hourly_rate" 'BEGIN { printf "$%.2f",s/3600*r }')
    projected_seconds=$((total * tokens_per_update / tps))
    projected_cost=$(awk -v s="$projected_seconds" -v r="$hourly_rate" 'BEGIN { printf "$%.2f",s/3600*r }')
fi

pid=$(pgrep -x LLM.Cli | head -1 || true)
if [[ -n "$pid" ]]; then
    state="${green}RUNNING${reset}"
    process_age=$(ps -p "$pid" -o etime= | xargs)
else
    state="${red}STOPPED${reset}"
    process_age="n/a"
fi

supervisor_pid=""
if [[ -f out/forge-supervisor.pid ]]; then
    read -r supervisor_pid < out/forge-supervisor.pid
    if ! kill -0 "$supervisor_pid" 2>/dev/null; then
        supervisor_pid=""
    fi
fi
supervisor_state="inactive"
if [[ -n "$supervisor_pid" ]]; then
    supervisor_state="${green}ACTIVE${reset} (PID $supervisor_pid)"
fi

model_info=$(grep -m1 '^model:' "$log_file" || true)
training_info=$(grep -m1 '^training:' "$log_file" || true)

gpu_info=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
gpu_name="unavailable" gpu_util="?" gpu_used="?" gpu_total="?" gpu_power="?" gpu_temp="?"
if [[ -n "$gpu_info" ]]; then
    IFS=, read -r gpu_name gpu_util gpu_used gpu_total gpu_power gpu_temp <<< "$gpu_info"
    gpu_name=$(xargs <<< "$gpu_name")
    gpu_util=$(xargs <<< "$gpu_util")
    gpu_used=$(xargs <<< "$gpu_used")
    gpu_total=$(xargs <<< "$gpu_total")
    gpu_power=$(xargs <<< "$gpu_power")
    gpu_temp=$(xargs <<< "$gpu_temp")
fi

checkpoint_step=$(awk '/^checkpoint: saved .*\(step [0-9]+\)/ { line=$0 } END { if (match(line, /step [0-9]+/)) print substr(line, RSTART+5, RLENGTH-5) }' "$log_file")
checkpoint_detail="not saved yet"
if [[ -f "$checkpoint" ]]; then
    checkpoint_size=$(du -h "$checkpoint" | awk '{print $1}')
    checkpoint_time=$(date -r "$checkpoint" '+%Y-%m-%d %H:%M:%S')
    checkpoint_detail="step ${checkpoint_step:-unknown}, ${checkpoint_size}, written ${checkpoint_time}"
fi

error_lines=$(grep -Ei 'non-finite|unhandled exception|^error:|training failed' "$log_file" | tail -3 || true)
if [[ -z "$error_lines" ]]; then
    health="${green}No training errors detected${reset}"
elif [[ -n "$pid" ]]; then
    health="${yellow}Recovered; trainer is running and historical errors are retained below${reset}"
else
    health="${red}Errors found in log${reset}"
fi

printf '%s\n' "${bold}${cyan}================================================================${reset}"
printf '%s\n' "${bold}                       FORGE TRAINING${reset}"
printf '%s\n' "${dim}$(TZ="$display_timezone" date '+%Y-%m-%d %H:%M:%S %Z')${reset}"
printf '%s\n\n' "${bold}${cyan}================================================================${reset}"

printf '%-19s %b\n' 'State' "$state"
printf '%-19s %s\n' 'Process' "PID ${pid:-n/a}, age $process_age"
printf '%-19s %b\n' 'Recovery supervisor' "$supervisor_state"
printf '%-19s [%s] %s%%\n' 'Progress' "$(bar "$step" "$total")" "$percent"
printf '%-19s %s / %s updates\n' 'Optimizer updates' "$step" "$total"
printf '%-19s %s / %s tokens\n' 'Tokens processed' "$processed_human" "$total_human"
printf '%-19s %s\n' 'Training loss' "$loss"
printf '%-19s %s (step %s)\n' 'Validation loss' "$val_loss" "$val_step"
printf '%-19s %s\n' 'Learning rate' "$lr"
printf '%-19s %s tok/s\n' 'Throughput' "$tps_text"
printf '%-19s %s\n' 'Log elapsed' "${elapsed:-n/a}"
printf '%-19s %s\n' 'ETA remaining' "$(format_duration "$eta_seconds")"
printf '%-19s %s\n' 'Estimated finish' "$finish_time"
printf '%-19s %s at $%s/hr\n' 'Remaining cost' "$remaining_cost" "$hourly_rate"
printf '%-19s %s\n' 'Projected run cost' "$projected_cost"

printf '\n%sGPU%s\n' "$bold" "$reset"
printf '  %-17s %s\n' 'Device' "$gpu_name"
printf '  %-17s %s%%\n' 'Utilization' "$gpu_util"
printf '  %-17s %s / %s MiB\n' 'VRAM' "$gpu_used" "$gpu_total"
printf '  %-17s %s W, %s C\n' 'Power / temp' "$gpu_power" "$gpu_temp"

printf '\n%sCheckpoint%s\n' "$bold" "$reset"
printf '  %s\n' "$checkpoint_detail"

printf '\n%sConfiguration%s\n' "$bold" "$reset"
printf '  %s\n' "$model_info"
printf '  %s\n' "$training_info"

printf '\n%sRecent validation%s\n' "$bold" "$reset"
awk '/^step +[0-9]+\/[0-9]+/ && / val / { lines[++n]=$0 } END { start=n-4; if (start<1) start=1; for (i=start;i<=n;i++) print "  " lines[i] }' "$log_file"

printf '\n%sHealth%s\n' "$bold" "$reset"
printf '  %b\n' "$health"
if [[ -n "$error_lines" ]]; then
    sed 's/^/  /' <<< "$error_lines"
fi
printf '%s\n' "${bold}${cyan}================================================================${reset}"
