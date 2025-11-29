#!/bin/bash
# Script para monitorear el progreso de la descarga de datos

echo "======================================"
echo "   AMi-Fuel Data Fetch Progress"
echo "======================================"
echo ""

# Verificar si el proceso está corriendo
if ps aux | grep -v grep | grep "fetch_expanded_training_data.py" > /dev/null; then
    echo "✅ Status: Running"
    PID=$(ps aux | grep -v grep | grep "fetch_expanded_training_data.py" | awk '{print $2}')
    echo "   Process ID: $PID"
else
    echo "❌ Status: Not running"
fi

echo ""
echo "📊 Progress:"

# Contar sesiones completadas
COMPLETED=$(grep "✓ ([0-9]" fetch_log.txt 2>/dev/null | wc -l | tr -d ' ')
echo "   Completed sessions: $COMPLETED/65"

# Calcular porcentaje
PERCENT=$((COMPLETED * 100 / 65))
echo "   Progress: $PERCENT%"

# Mostrar vueltas totales hasta ahora
TOTAL_LAPS=$(grep "✓ ([0-9]" fetch_log.txt 2>/dev/null | grep -o "[0-9]* laps" | awk '{sum+=$1} END {print sum}')
if [ ! -z "$TOTAL_LAPS" ]; then
    echo "   Total laps fetched: $TOTAL_LAPS"
fi

echo ""
echo "📝 Last 5 completed sessions:"
grep "✓ ([0-9]" fetch_log.txt 2>/dev/null | tail -5

echo ""
echo "🔄 Currently processing:"
tail -3 fetch_log.txt 2>/dev/null | grep "Fetching" || echo "   (checking...)"

echo ""
echo "======================================"
echo ""
echo "To see live updates: tail -f fetch_log.txt"
echo "To check again: bash check_progress.sh"
echo ""
