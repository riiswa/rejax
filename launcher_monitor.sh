#!/bin/bash

# Simplified script to launch and monitor all exploration experiments
# Usage: ./launch_monitor.sh [--submit-only] [--monitor-only]

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Job script files
JOB_SCRIPTS=(
    "exploration_rnk.sh"
    "exploration_rnd.sh"
    "exploration_vime.sh"
    "exploration_hash.sh"
    "exploration_none.sh"
)

# Bonus types for reference
BONUS_TYPES=("rnk" "rnd" "vime" "hash" "none")

# Common results file
RESULTS_FILE="experiment_results.log"

# Function to print colored messages
print_msg() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to get total number of environments
get_total_envs() {
    source envs.sh
    echo ${#ENVS[@]}
}

# Function to submit all jobs
submit_jobs() {
    print_msg $BLUE "=== Submitting Exploration Experiments ==="
    
    # Clear previous results
    > "$RESULTS_FILE"
    
    local total_envs=$(get_total_envs)
    print_msg $YELLOW "Total environments per bonus type: $total_envs"
    print_msg $YELLOW "Total experiments across all bonus types: $((total_envs * 5))"
    echo
    
    local submitted_count=0
    
    for script in "${JOB_SCRIPTS[@]}"; do
        if [[ ! -f "$script" ]]; then
            print_msg $RED "Error: Script $script not found!"
            continue
        fi
        
        print_msg $BLUE "Submitting $script..."
        
        if sbatch "$script" > /dev/null; then
            print_msg $GREEN "✓ Submitted $script"
            ((submitted_count++))
        else
            print_msg $RED "✗ Failed to submit $script"
        fi
    done
    
    echo
    print_msg $GREEN "Successfully submitted $submitted_count out of ${#JOB_SCRIPTS[@]} jobs"
    
    if [[ $submitted_count -gt 0 ]]; then
        print_msg $YELLOW "Results will be logged to: $RESULTS_FILE"
        print_msg $YELLOW "Use 'squeue -u \$USER' to check SLURM job status"
        print_msg $YELLOW "Use './launch_monitor.sh --monitor-only' to monitor progress"
    fi
}

# Function to parse results and get stats
get_stats() {
    local bonus_type=$1
    local total_envs=$2
    
    if [[ ! -f "$RESULTS_FILE" ]]; then
        echo "0 0 $total_envs"
        return
    fi
    
    local success_count=$(grep "|$bonus_type|.*|SUCCESS|" "$RESULTS_FILE" 2>/dev/null | wc -l)
    local failed_count=$(grep "|$bonus_type|.*|FAILED|" "$RESULTS_FILE" 2>/dev/null | wc -l)
    local remaining=$((total_envs - success_count - failed_count))
    
    echo "$success_count $failed_count $remaining"
}

# Function to monitor experiments
monitor_experiments() {
    print_msg $BLUE "=== Monitoring Exploration Experiments ==="
    
    local total_envs=$(get_total_envs)
    local total_experiments=$((total_envs * 5))
    
    while true; do
        clear
        print_msg $BLUE "=== Experiment Progress Monitor ($(date)) ==="
        echo
        
        if [[ ! -f "$RESULTS_FILE" ]]; then
            print_msg $YELLOW "No results file found yet. Waiting for experiments to start..."
            sleep 10
            continue
        fi
        
        printf "%-10s %-9s %-6s %-9s %-8s\n" "BonusType" "Success" "Failed" "Remaining" "Progress"
        printf "%-10s %-9s %-6s %-9s %-8s\n" "---------" "-------" "------" "---------" "--------"
        
        local total_success=0
        local total_failed=0
        local total_remaining=0
        local all_completed=true
        
        for bonus_type in "${BONUS_TYPES[@]}"; do
            read success failed remaining <<< $(get_stats "$bonus_type" "$total_envs")
            
            total_success=$((total_success + success))
            total_failed=$((total_failed + failed))
            total_remaining=$((total_remaining + remaining))
            
            if [[ $remaining -gt 0 ]]; then
                all_completed=false
            fi
            
            local progress="$((success + failed))/$total_envs"
            
            # Color code based on status
            local color=$NC
            if [[ $remaining -eq 0 ]]; then
                if [[ $failed -eq 0 ]]; then
                    color=$GREEN  # All success
                else
                    color=$YELLOW # Some failed
                fi
            fi
            
            printf "${color}%-10s %-9s %-6s %-9s %-8s${NC}\n" \
                "$bonus_type" "$success" "$failed" "$remaining" "$progress"
        done
        
        echo
        printf "%-10s %-9s %-6s %-9s %-8s\n" "TOTAL" "$total_success" "$total_failed" "$total_remaining" "$((total_success + total_failed))/$total_experiments"
        
        # Calculate overall progress
        local progress_percent=$(( (total_success + total_failed) * 100 / total_experiments ))
        echo
        print_msg $YELLOW "Overall progress: $progress_percent% ($((total_success + total_failed))/$total_experiments experiments)"
        
        if [[ $total_failed -gt 0 ]]; then
            print_msg $RED "⚠️  $total_failed experiments have failed"
            echo
            print_msg $RED "Failed Experiments:"
            grep "|.*|.*|FAILED|" "$RESULTS_FILE" 2>/dev/null | while IFS='|' read -r timestamp bonus_type env status message; do
                printf "  ${RED}• %s - %s${NC}\n" "$bonus_type" "$env"
            done
        fi
        
        if [[ $all_completed == true ]]; then
            echo
            if [[ $total_failed -eq 0 ]]; then
                print_msg $GREEN "🎉 All experiments completed successfully!"
            else
                print_msg $YELLOW "✅ All experiments finished. $total_failed failed, $total_success succeeded."
            fi
            print_msg $YELLOW "Detailed results in: $RESULTS_FILE"
            break
        else
            echo
            print_msg $YELLOW "⏳ $total_remaining experiments remaining. Refreshing in 30 seconds..."
            print_msg $YELLOW "Press Ctrl+C to stop monitoring (experiments will continue running)"
            sleep 30
        fi
    done
}

# Function to show recent activity
show_recent() {
    if [[ -f "$RESULTS_FILE" ]]; then
        echo
        print_msg $BLUE "=== Recent Activity (Last 10 entries) ==="
        tail -n 10 "$RESULTS_FILE" | while IFS='|' read -r timestamp bonus_type env status message; do
            local color=$GREEN
            [[ "$status" == "FAILED" ]] && color=$RED
            printf "${color}[%s] %s - %s: %s${NC}\n" "$timestamp" "$bonus_type" "$env" "$status"
        done
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --submit-only     Submit all jobs and exit"
    echo "  --monitor-only    Monitor experiments (read results file)"
    echo "  --recent          Show recent experiment activity"
    echo "  --help            Show this help message"
    echo
    echo "Default behavior: Submit jobs and then monitor them"
}

# Parse command line arguments
case "${1:-}" in
    --submit-only)
        submit_jobs
        ;;
    --monitor-only)
        monitor_experiments
        ;;
    --recent)
        show_recent
        ;;
    --help)
        show_usage
        ;;
    "")
        # Default: submit and monitor
        submit_jobs
        echo
        print_msg $YELLOW "Starting monitoring in 5 seconds..."
        sleep 5
        monitor_experiments
        ;;
    *)
        print_msg $RED "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac