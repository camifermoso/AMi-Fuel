#!/bin/bash
# Script to monitor training progress

echo "======================================"
echo "   AMi-Fuel Model Training Progress"
echo "======================================"
echo ""

# Check if process is running
if ps aux | grep -v grep | grep "train_two_stage_model.py" > /dev/null; then
    echo "✅ Status: Training in progress"
    PID=$(ps aux | grep -v grep | grep "train_two_stage_model.py" | awk '{print $2}')
    echo "   Process ID: $PID"
else
    echo "❌ Status: Not running"
fi

echo ""
echo "📝 Training Log (last 30 lines):"
echo "--------------------------------------"
tail -30 training_log.txt 2>/dev/null || echo "Log file not created yet"

echo ""
echo "======================================"
echo ""
echo "To see live updates: tail -f training_log.txt"
echo "To check again: bash check_training.sh"
echo ""
