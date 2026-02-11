#!/bin/bash
# Retraining Scheduler Management Script
# This script provides easy commands to manage the model retraining scheduler

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python"
SCHEDULER_SCRIPT="$SCRIPT_DIR/retraining_scheduler.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if Python is available
check_python() {
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python command '$PYTHON_CMD' not found"
        exit 1
    fi
}

# Check if scheduler script exists
check_scheduler_script() {
    if [ ! -f "$SCHEDULER_SCRIPT" ]; then
        print_error "Scheduler script not found: $SCHEDULER_SCRIPT"
        exit 1
    fi
}

# Start the scheduler
start_scheduler() {
    print_info "Starting Model Retraining Scheduler..."
    check_python
    check_scheduler_script

    $PYTHON_CMD "$SCHEDULER_SCRIPT" --start
}

# Stop the scheduler
stop_scheduler() {
    print_info "Stopping Model Retraining Scheduler..."
    check_python
    check_scheduler_script

    $PYTHON_CMD "$SCHEDULER_SCRIPT" --stop
}

# Check scheduler status
status_scheduler() {
    print_info "Checking scheduler status..."
    check_python
    check_scheduler_script

    $PYTHON_CMD "$SCHEDULER_SCRIPT" --status
}

# Manual retraining
retrain_model() {
    local model_name=$1
    if [ -z "$model_name" ]; then
        print_error "Model name is required for retraining"
        echo "Usage: $0 retrain <model_name> [reason]"
        exit 1
    fi

    local reason=${2:-"Manual retraining"}
    print_info "Retraining model: $model_name (Reason: $reason)"
    check_python
    check_scheduler_script

    $PYTHON_CMD "$SCHEDULER_SCRIPT" --retrain "$model_name"
}

# Retrain all models
retrain_all() {
    print_info "Retraining all models..."
    check_python
    check_scheduler_script

    $PYTHON_CMD "$SCHEDULER_SCRIPT" --retrain-all
}

# Check performance degradation
check_performance() {
    print_info "Checking for performance degradation..."
    check_python
    check_scheduler_script

    $PYTHON_CMD "$SCHEDULER_SCRIPT" --check-performance
}

# Show help
show_help() {
    echo "Model Retraining Scheduler Management Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start                 Start the retraining scheduler"
    echo "  stop                  Stop the retraining scheduler"
    echo "  status                Show scheduler status"
    echo "  retrain <model>       Manually retrain a specific model"
    echo "  retrain-all           Retrain all available models"
    echo "  check-performance     Check for performance degradation"
    echo "  help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 retrain random_forest"
    echo "  $0 retrain-all"
    echo "  $0 check-performance"
    echo "  $0 status"
    echo ""
    echo "Available models: random_forest, xgboost, lightgbm, ensemble, ols_static"
}

# Main script logic
case "${1:-help}" in
    start)
        start_scheduler
        ;;
    stop)
        stop_scheduler
        ;;
    status)
        status_scheduler
        ;;
    retrain)
        retrain_model "$2" "$3"
        ;;
    retrain-all)
        retrain_all
        ;;
    check-performance)
        check_performance
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac</content>
<parameter name="filePath">c:\Users\RAKSHANDA\Downloads\reserach\Bipllab Sir\Stock Market Analysis\manage_retraining.sh