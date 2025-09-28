# Multiple Observers, One Reality: Navigating Perspective
# How different ways of seeing create apparent tensions—and how mathematics
# measures the true cost of reconciling what seems irreconcilable.

from contrakit.observatory import Observatory
from contrakit.behavior.behavior import Behavior

print("The Sky at Dusk")
print("=" * 50)
print()
print("Picture two astronomers at the same mountaintop observatory, both trying")
print("to answer the same question: Is it day or night? They watch the same sky,")
print("breathe the same air, yet each develops their own strategy for reading the signs.")
print()
print("Dr. Harris trusts the sun's direct light—measuring sky brightness to decide.")
print("Dr. Ward relies on the building's automatic sensors, watching when lights flick on.")
print()
print("Both are careful observers, each with sound reasoning. Yet their answers")
print("sometimes clash. Not because one is wrong, but because they're watching")
print("different instruments in the same unfolding scene.")
print()
print("The question becomes: How much work does it take to reconcile these views?")
print("What information cost hides in their disagreements?")
print()

# First, we need a shared language for measurement
obs = Observatory.create(symbols=["High", "Low"])
High, Low = obs.alphabet

# Define what we'll be observing
DayTime, SkyLuminance, AutoLights = obs.define_many([
    {"name": "DayTime", "symbols": ["Day", "Night"]},
    "SkyLuminance",  # Brightness readings
    "AutoLights"     # Lighting system state
])

print("I. The World as It Is")
print("-" * 30)
print()
print("Before we examine our observers, we need to understand the landscape they're navigating.")
print("Here's how sky brightness and building lights actually correlate in this observatory:")
print()

# The objective patterns in the world
obs.perspectives[(SkyLuminance, AutoLights)] = {
    High & High: 0.08,    # Bright sky with lights on (twilight transitions, sensor glitches)
    High & Low:  0.72,    # Bright sky, lights off (clear daytime)
    Low & High:  0.15,    # Dark sky, lights on (normal nighttime)
    Low & Low:   0.05,    # Dark sky, lights off (power failures, deep night)
}

physical_reality = obs.perspectives.to_behavior()

print("What we observe in the world:")
print("• Bright skies with lights off: 72% of the time (normal daylight)")
print("• Dark skies with lights on: 15% of the time (standard nighttime)")
print("• The remaining 13% shows the world's complexity—twilight, malfunctions, outages")
print()
print("This baseline captures the actual rhythms of the observatory,")
print("the patterns that exist whether anyone is watching or not.")
print("It's our ground truth, the steady pulse beneath all observation.")
print()

print("II. Harris's Path: Following the Sun")
print("-" * 41)
print()
print("Dr. Harris takes the direct approach. 'The sun paints the sky,' she says.")
print("'Brightness tells the true story. When the sky glows, it's day.'")
print()

with obs.lens("Harris") as harris:
    # Harris maps sky brightness directly to time of day
    harris.perspectives[(SkyLuminance, DayTime)] = {
        (High, "Day"): 0.85,    # Bright sky means day (strong pattern)
        (High, "Night"): 0.03,  # Bright sky rarely means night (rare mistake)
        (Low, "Day"): 0.02,     # Dark sky almost never means day (very rare)
        (Low, "Night"): 0.10,   # Dark sky usually means night (solid but not perfect)
    }
    harris_method = harris.to_behavior()

print("Harris's strategy:")
print("• Bright sky signals day (85% reliability)")
print("• Dark sky suggests night (10% uncertainty)")
print("• Overall error rate stays low, around 3%")
print()
print("She's following the most obvious clue, trusting the sun's direct signature.")
print("Simple, intuitive—and effective most of the time.")
print()

print("III. Ward's Path: Trusting the Machine")
print("-" * 43)
print()
print("Dr. Ward takes a different route. 'The building knows,' he says.")
print("'Its sensors are engineered for this. When lights come on, darkness has fallen.'")
print()

with obs.lens("Ward") as ward:
    # Ward interprets the lighting system as the authoritative signal
    ward.perspectives[(AutoLights, DayTime)] = {
        (High, "Day"): 0.04,    # Lights on rarely means day (sensor error)
        (High, "Night"): 0.88,  # Lights on strongly indicates night (system design)
        (Low, "Day"): 0.06,     # Lights off suggests day (reasonable but not certain)
        (Low, "Night"): 0.02,   # Lights off almost never means night (power issues)
    }
    ward_method = ward.to_behavior()

print("Ward's approach:")
print("• Lights on means night (88% certainty)")
print("• Lights off suggests day (6% uncertainty)")
print("• He builds in tolerance for mechanical quirks and outages")
print()
print("He's delegating to technology, letting the building's automation")
print("do the heavy lifting. Smart engineering, with room for hiccups.")
print()

print("IV. Harmony Between Paths")
print("-" * 40)
print()
print("Now we have two honest observers, each following their own reasoning.")
print("The question becomes: How often would they actually agree?")
print()
print("Information theory gives us a way to measure this precisely—the")
print("probability that two methods would converge on the same answer.")
print()

agreement_coefficient = (harris_method | ward_method).agreement.result

print(f"Agreement coefficient: {agreement_coefficient:.4f}")
print(f"They align {agreement_coefficient*100:.1f}% of the time")
print()
print("This measures something concrete: given the same observational data,")
print("what's the chance both observers would call it the same way?")
print(f"In this case, their paths converge {agreement_coefficient*100:.1f}% of the time.")
print()
print("Not perfect harmony, but impressive coordination between different approaches.")
print("The disagreement isn't chaos—it's a structured gap we can analyze.")
print()

print("V. Mapping the Terrain: Lens Operations")
print("-" * 52)
print()
print("We've seen two paths through the same landscape. Now let's use")
print("mathematical tools to chart their relationships more deeply.")
print("These 'lens operations' let us combine, compare, and contrast perspectives.")
print()

# Set up the lenses for detailed analysis
harris_lens = obs.lens("Harris")
ward_lens = obs.lens("Ward")

# Configure each observer's viewpoint
with harris_lens:
    harris_lens.perspectives[(SkyLuminance, DayTime)] = {
        (High, "Day"): 0.85,    # Bright sky → day (high confidence)
        (High, "Night"): 0.03,  # Bright sky → night (rare error)
        (Low, "Day"): 0.02,     # Dark sky → day (rare error)
        (Low, "Night"): 0.10,   # Dark sky → night (moderate confidence)
    }

with ward_lens:
    ward_lens.perspectives[(AutoLights, DayTime)] = {
        (High, "Day"): 0.04,    # Lights on → day (system error)
        (High, "Night"): 0.88,  # Lights on → night (high confidence)
        (Low, "Day"): 0.06,     # Lights off → day (moderate confidence)
        (Low, "Night"): 0.02,   # Lights off → night (power failure)
    }

# Union: The complete picture when we consider both views
combined_story = (harris_lens | ward_lens).to_behavior()

internal_agreement = combined_story.contradiction_bits
print(f"Combined view contradiction cost: {internal_agreement:.3f} bits")

# Intersection: Where both observers would agree
consensus = (harris_lens.intersection(ward_lens)).to_behavior()
print(f"Consensus regions: {len(consensus.distributions)} shared contexts")

# Difference: What Harris sees that Ward might miss
harris_unique = (harris_lens.difference(ward_lens)).to_behavior()
print(f"Harris's unique perspectives: {len(harris_unique.distributions)} contexts")

# Symmetric difference: The core conflicts between approaches
disagreements = (harris_lens.symmetric_difference(ward_lens)).to_behavior()
print(f"Fundamental disagreements: {len(disagreements.distributions)} conflicting contexts")
print()
print("Each operation highlights a different facet:")
print("• Union shows the total tension when both views coexist")
print("• Intersection finds the stable ground where they overlap")
print("• Difference reveals what each observer contributes uniquely")
print("• Symmetric difference isolates the irreconcilable gaps")
print()
print("We're not just noting disagreement—we're measuring its structure,")
print("understanding where harmony lives and where tension persists.")
print()

if internal_agreement > 0.001:
    print("VI. Sources of Tension")
    print("-" * 35)
    print()
    print("When contradictions appear, we can ask: Which specific perspectives")
    print("drive the disagreement? The framework lets us trace the tension back")
    print("to its origins, measuring each viewpoint's contribution.")
    print()

    # Analyze which perspectives create the most friction
    composition = harris_lens | ward_lens

    for perspective, contribution in composition.perspective_contributions.items():
        if contribution > 0.01:  # Focus on meaningful contributors
            print(f"• {perspective}: drives {contribution:.1%} of the contradiction")
    print()
    print("This reveals the disagreement's anatomy—not just that tension exists,")
    print("but where it concentrates and why.")
    print()

print("VII. Against the World")
print("-" * 42)
print()
print("Finally, let's measure both observers against the objective reality")
print("we established earlier. How well does each method capture the world's")
print("actual patterns?")
print()

# Test each approach against ground truth
harris_accuracy = (physical_reality | harris_method).agreement.result
ward_accuracy = (physical_reality | ward_method).agreement.result

print(f"Harris's accuracy: {harris_accuracy*100:.1f}%")
print(f"Ward's accuracy: {ward_accuracy*100:.1f}%")
print()
print("These figures show how faithfully each method reflects the underlying")
print("correlations in the world—not just internal consistency, but alignment")
print("with what actually happens.")
print()
print("Both do reasonably well, but neither is perfect. Reality has its own")
print("complexity that no single observation strategy can fully capture.")
print()

print("VIII. The Operations in Summary")
print("-" * 32)
print()
print("These lens operations give us a complete map of multi-observer dynamics:")
print(f"• Shared ground: {len(consensus.distributions)} contexts where both agree")
print(f"• Individual insights: {len(harris_unique.distributions) + len((ward_lens.difference(harris_lens)).to_behavior().distributions)} unique contributions")
print(f"• Irreconcilable differences: {len(disagreements.distributions)} conflicting contexts")
print()
print("Each number tells a story about how perspectives complement and compete.")
print()

print("IX. What We've Learned")
print("-" * 20)
print()
print("This exploration reveals deeper patterns in how we observe:")
print()
print(f"1. Different paths can lead to substantial agreement ({agreement_coefficient*100:.1f}%)")
print("   even when they follow entirely different clues.")
print()
print("2. The work required to reconcile viewpoints has a concrete cost")
print(f"   ({internal_agreement:.3f} bits in this case)—not vague discomfort, but measurable effort.")
print()
print("3. Mathematical operations let us dissect multi-observer systems:")
print("   • Union captures total tension when views coexist")
print("   • Intersection finds reliable common ground")
print("   • Difference highlights what each observer adds")
print("   • Symmetric difference isolates fundamental conflicts")
print()
print("4. We can trace disagreement to its sources, understanding not just")
print("   that tension exists, but which perspectives drive it most.")
print()
print(f"5. Both approaches capture reality reasonably well (Harris: {harris_accuracy*100:.1f}%, Ward: {ward_accuracy*100:.1f}%),")
print("   but each has blind spots that the other compensates for.")
print()
print("The framework transforms fuzzy intuitions about 'disagreement' into")
print("precise measurements. Just as Shannon gave us bits to measure information,")
print("here we get tools to quantify the coherence of multiple viewpoints.")
print()
print("The result? We can now ask: not whether observers agree, but how much")
print("their disagreements cost—and whether that cost is worth paying.")