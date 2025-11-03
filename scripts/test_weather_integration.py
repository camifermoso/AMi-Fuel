"""
Test script to verify weather data fetching works correctly
"""
import sys
sys.path.insert(0, 'scripts')

from fetch_fastf1_highfuel import fetch_session

print("="*80)
print("TESTING WEATHER DATA INTEGRATION")
print("="*80)
print()

# Test fetching one session with weather
print("Fetching 2023 Bahrain GP with weather data...")
df = fetch_session(2023, "bahrain", "R")

print(f"\n✓ Fetched {len(df)} laps")
print(f"\n📊 Columns in dataset:")
print(df.columns.tolist())

print(f"\n🌡️ Weather data sample:")
weather_cols = ['air_temp', 'track_temp', 'humidity', 'pressure', 'rainfall', 'wind_speed', 'wind_direction']
available_weather = [col for col in weather_cols if col in df.columns]

if available_weather:
    print(df[available_weather].head(10))
    print(f"\n📈 Weather statistics:")
    print(df[available_weather].describe())
    
    print(f"\n✅ Weather integration successful!")
    print(f"   • Air temp range: {df['air_temp'].min():.1f}°C - {df['air_temp'].max():.1f}°C")
    print(f"   • Track temp range: {df['track_temp'].min():.1f}°C - {df['track_temp'].max():.1f}°C")
    print(f"   • Humidity range: {df['humidity'].min():.1f}% - {df['humidity'].max():.1f}%")
    print(f"   • Rainfall laps: {df['rainfall'].sum()} / {len(df)}")
else:
    print("❌ No weather data found in dataset!")
    
print()
print("="*80)
print()
print("Next steps:")
print("  1. Tonight: Run fetch_expanded_training_data.py to get 7 years of data")
print("  2. The script will automatically include weather for all sessions")
print("  3. Retrain models with weather features included")
print("  4. Expected improvement: 96% → 97-98% accuracy")
print()
