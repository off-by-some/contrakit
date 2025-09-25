#!/usr/bin/env python3
"""
Golden Example: How Contexts Work in Contrakit

This example demonstrates the fundamental concept of observational contexts,
which are the building blocks for measuring contradiction. Contexts represent
specific subsets of observables that are measured together, allowing us to
model different experimental or observational setups.

Key Concepts Demonstrated:
- Creating observable spaces
- Defining measurement contexts
- Context operations (union, intersection)
- How contexts relate to outcomes and behaviors
- The role of contexts in contradiction measurement

Contexts are essential because they define which variables are jointly observed,
allowing us to analyze whether different measurement perspectives are consistent.
"""

from contrakit.space import Space
from contrakit.context import Context
from contrakit.behavior.behavior import Behavior
from examples.utils import print_header

# Define our weather monitoring system constants
TEMPERATURE = "Temperature"
PRESSURE = "Pressure"
HUMIDITY = "Humidity"
WIND = "Wind"

# Temperature values
HOT = "Hot"
WARM = "Warm"
COLD = "Cold"

# Pressure values
HIGH = "High"
LOW = "Low"

# Humidity values
WET = "Wet"
DRY = "Dry"

# Wind values
STRONG = "Strong"
LIGHT = "Light"

# Context definitions
TEMP_HUMIDITY_CONTEXT = (TEMPERATURE, HUMIDITY)
PRESSURE_WIND_CONTEXT = (PRESSURE, WIND)
TEMP_PRESSURE_CONTEXT = (TEMPERATURE, PRESSURE)

print_header("Golden Example: Observational Contexts")

# 1. Define the Observable Space
print("\n1. Creating the Observable Space")
print("-" * 35)

# Imagine we're studying weather patterns with multiple sensors
weather_space = Space.create(
    **{TEMPERATURE: [HOT, WARM, COLD],
       PRESSURE: [HIGH, LOW],
       HUMIDITY: [WET, DRY],
       WIND: [STRONG, LIGHT]}
)

print(f"Created space with {len(weather_space.names)} observables:")
for name in weather_space.names:
    values = weather_space.alphabets[name]
    print(f"  {name}: {list(values)}")

# 2. Define Measurement Contexts
print("\n2. Defining Measurement Contexts")
print("-" * 35)

# Different measurement setups can only observe certain combinations
temp_humidity = Context(weather_space, TEMP_HUMIDITY_CONTEXT)
pressure_wind = Context(weather_space, PRESSURE_WIND_CONTEXT)
temp_pressure = Context(weather_space, TEMP_PRESSURE_CONTEXT)

print("Created contexts:")
print(f"  temp_humidity: {temp_humidity.observables}")
print(f"  pressure_wind: {pressure_wind.observables}")
print(f"  temp_pressure: {temp_pressure.observables}")

# 3. Context Operations
print("\n3. Context Operations")
print("-" * 22)

# Union: Combine contexts to measure more variables together
combined_context = temp_humidity | [PRESSURE, WIND]
print(f"Union (temp_humidity | ['{PRESSURE}', '{WIND}']): {combined_context.observables}")

# Intersection: Find common observables
overlap = temp_humidity & temp_pressure
print(f"Intersection (temp_humidity & temp_pressure): {overlap.observables}")

# 4. Context Outcomes
print("\n4. Possible Outcomes for Each Context")
print("-" * 38)

print(f"temp_humidity can observe {len(temp_humidity.outcomes())} combinations:")
for i, outcome in enumerate(temp_humidity.outcomes()[:5]):  # Show first 5
    temp, hum = outcome
    print(f"  {i+1}. {TEMPERATURE}={temp}, {HUMIDITY}={hum}")
if len(temp_humidity.outcomes()) > 5:
    print(f"  ... and {len(temp_humidity.outcomes()) - 5} more")

# 5. Context Restriction
print("\n5. Context Restriction")
print("-" * 22)

# A complete assignment to all variables
complete_reading = (HOT, HIGH, WET, STRONG)  # Temp, Pressure, Humidity, Wind

print(f"Complete reading: {dict(zip(weather_space.names, complete_reading))}")

# Different contexts see different parts
temp_hum_part = temp_humidity.restrict_assignment(complete_reading)
pressure_wind_part = pressure_wind.restrict_assignment(complete_reading)

print(f"temp_humidity context sees: {dict(zip(temp_humidity.observables, temp_hum_part))}")
print(f"pressure_wind context sees: {dict(zip(pressure_wind.observables, pressure_wind_part))}")

# 6. Contexts in Behaviors
print("\n6. Contexts Define What Behaviors Measure")
print("-" * 42)

# Create a simple behavior using our contexts
# Note: from_contexts expects tuples of observable names as keys, not Context objects
contexts_data = {
    TEMP_HUMIDITY_CONTEXT: {
        (HOT, WET): 0.4,    # 40% chance of hot and wet
        (HOT, DRY): 0.2,    # 20% chance of hot and dry
        (WARM, WET): 0.3,   # 30% chance of warm and wet
        (WARM, DRY): 0.05,  # 5% chance of warm and dry
        (COLD, WET): 0.04,  # 4% chance of cold and wet
        (COLD, DRY): 0.01,  # 1% chance of cold and dry
    },
    PRESSURE_WIND_CONTEXT: {
        (HIGH, STRONG): 0.25,  # 25% chance of high pressure and strong wind
        (HIGH, LIGHT): 0.35,   # 35% chance of high pressure and light wind
        (LOW, STRONG): 0.15,   # 15% chance of low pressure and strong wind
        (LOW, LIGHT): 0.25,    # 25% chance of low pressure and light wind
    }
}

behavior = Behavior.from_contexts(weather_space, contexts_data)

print("Created behavior measuring weather patterns:")
print(f"  Contradiction cost: {behavior.contradiction_bits:.4f} bits")
print(f"  Best classical overlap: {behavior.alpha_star:.4f}")
print("  Measures weather correlations across temperature+humidity and pressure+wind contexts")
