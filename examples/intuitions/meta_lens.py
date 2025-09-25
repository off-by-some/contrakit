# Same Scene, Different Zoom Levels
# How the same mathematical lens that measures reviewer disagreement also measures
# supervisor disagreement about reviewers—and director disagreement about supervisors.

# A meta-lens is essentially a recursively composable measure of disagreement. In CS 
# terms, it's akin to having a "monoidal information distance" that works at any 
# abstraction level. You don't need to invent a new metric for "workers vs 
# supervisors vs managers"—the same math applies, automatically.

# New level, same lens.

from contrakit.observatory import Observatory

def watching_the_watchers():
    print("Same Scene, Different Zoom Levels")
    print("=" * 50)
    print()
    print("Think of every person as holding a lens. A lens doesn't create facts;")
    print("it weights them. Three reviewers read the same résumé but weight it")
    print("differently. Their supervisor weighs the reviewers differently. The")
    print("director weighs everyone differently.")
    print()
    print("Same scene, same tools, just different zoom levels.")
    print()
    print("What we'll show is simple but powerful: the very same math that")
    print("describes reviewers looking at a candidate also describes a supervisor")
    print("looking at the reviewers, and a director looking at everyone.")
    print()
    print("New level, same lens.")
    print()

    # Create our measurement space
    obs = Observatory.create(symbols=["Hire", "No_Hire"])
    hire, no_hire = obs.alphabet
    candidate = obs.concept("Candidate")

    print("I. First Lens: The Reviewers")
    print("-" * 32)
    print()
    print("Alice, Bob, and Charlie read the same résumé but weight it differently.")
    print("That's all a 'distribution' is here: a tidy way to say how strongly")
    print("a lens leans toward outcomes.")
    print()

    # Level 1: Individual reviewer perspectives
    alice = obs.lens("Alice", symbols=["Reliable", "Unreliable"])
    bob = obs.lens("Bob", symbols=["Consistent", "Inconsistent"])
    charlie = obs.lens("Charlie", symbols=["Accurate", "Biased"])

    with alice:
        alice.perspectives[candidate] = {hire: 0.8, no_hire: 0.2}
    with bob:
        bob.perspectives[candidate] = {hire: 0.3, no_hire: 0.7}
    with charlie:
        charlie.perspectives[candidate] = {hire: 0.6, no_hire: 0.4}

    print("Alice: 80% Hire, 20% No Hire")
    print("  (She rewards unconventional backgrounds)")
    print()
    print("Bob: 30% Hire, 70% No Hire") 
    print("  (He penalizes technical gaps)")
    print()
    print("Charlie: 60% Hire, 40% No Hire")
    print("  (He tries to balance)")
    print()
    print("There's no chaos here. The differences are information about")
    print("priorities and risk tolerance—not noise. We can already ask:")
    print("How far apart are these lenses? That 'how far' is what our")
    print("contradiction bits number summarizes.")
    print()

    print("II. Second Lens: The Supervisor (A Lens on Lenses)")
    print("-" * 53)
    print()
    print("The supervisor doesn't rescore the candidate directly. She scores")
    print("the reviewers: how reliable Alice is, how consistent Bob is, how")
    print("accurate Charlie tends to be. She assigns distributions to the")
    print("reviewers' traits, and even to correlations between them.")
    print()
    print("Key point: she uses the same format—probabilities over labeled")
    print("outcomes. That's the whole trick. The math doesn't change.")
    print("Only the labels change.")
    print()

    supervisor = obs.lens("Supervisor", symbols=["Trustworthy", "Questionable"])
    reliable, unreliable = alice.alphabet
    consistent, inconsistent = bob.alphabet
    accurate, biased = charlie.alphabet

    with supervisor:
        supervisor.perspectives[alice] = {reliable: 0.9, unreliable: 0.1}
        supervisor.perspectives[bob] = {consistent: 0.7, inconsistent: 0.3}
        supervisor.perspectives[charlie] = {accurate: 0.6, biased: 0.4}
        # She also notices patterns in how reviewers work together
        supervisor.perspectives[alice & bob] = {
            (reliable, consistent): 0.6,
            (reliable, inconsistent): 0.3,
            (unreliable, consistent): 0.1,
            (unreliable, inconsistent): 0.0
        }

    print("The supervisor's lens weights:")
    print("• Alice: 90% reliable, 10% unreliable")
    print("• Bob: 70% consistent, 30% inconsistent") 
    print("• Charlie: 60% accurate, 40% biased")
    print("• Alice & Bob together: 60% both reliable and consistent")
    print()
    print("Same mathematical structure as before. Whether the object under")
    print("the lens is a candidate or a reviewer doesn't matter. The format")
    print("stays identical: probabilities over labeled outcomes.")
    print()

    print("III. Third Lens: The Director (Zooming Out Again)")
    print("-" * 49)
    print()
    print("The director puts a lens on the supervisor and on the reviewers.")
    print("Sometimes she agrees with the supervisor, sometimes not. Again,")
    print("nothing new mathematically has appeared. We're just stacking lenses.")
    print()

    director = obs.lens("Director", symbols=["Competent", "Incompetent"])
    trustworthy, questionable = supervisor.alphabet

    with director:
        director.perspectives[supervisor] = {trustworthy: 0.8, questionable: 0.2}
        director.perspectives[alice] = {reliable: 0.95, unreliable: 0.05}
        director.perspectives[bob] = {consistent: 0.5, inconsistent: 0.5}

    print("The director's lens weights:")
    print("• Supervisor: 80% trustworthy, 20% questionable")
    print("• Alice: 95% reliable, 5% unreliable")
    print("• Bob: 50% consistent, 50% inconsistent")
    print()
    print("This stack—reviewers → supervisor → director—is what we mean by")
    print("a hierarchy of observation. At each level, we attach the same")
    print("kind of distributions to the thing we're watching.")
    print()

    print("IV. What the Numbers Mean (Contradiction Bits)")
    print("-" * 47)
    print()
    print("When we print a value like 0.051 bits for 'reviewer disagreement,'")
    print("read it as: how many crisp yes/no answers you'd need, on average,")
    print("to reconcile these lenses into a single story.")
    print()
    print("0.000 bits means no tension: the pieces fit perfectly.")
    print("Bigger numbers mean more tension: more clarifying answers required.")
    print()

    # Calculate contradiction at each level
    reviewers_behavior = (alice | bob | charlie).to_behavior()
    supervisor_behavior = supervisor.to_behavior()
    director_behavior = director.to_behavior()
    cross_level_behavior = (supervisor | director).to_behavior()

    print(f"Reviewers about candidate: {reviewers_behavior.contradiction_bits:.3f} bits")
    print("  → Mild disagreement: less than one clarifying question needed")
    print(f"Supervisor about reviewers: {supervisor_behavior.contradiction_bits:.3f} bits")
    print("  → Perfect internal consistency: no tension to resolve")
    print(f"Director about supervisor/reviewers: {director_behavior.contradiction_bits:.3f} bits")
    print("  → Also perfectly consistent within her own perspective")
    print(f"Cross-level (supervisor vs. director): {cross_level_behavior.contradiction_bits:.3f} bits")
    print("  → Tiny cross-level tension: they mostly agree about people")
    print()
    print("Same code path; different nouns. Because it's in bits, the scale")
    print("is compact and composable. Doubling tension doesn't feel linear")
    print("to humans, but bits behave well under combination—handy when")
    print("you stack levels.")
    print()


def when_hierarchies_clash():
    print("V. Cross-Level Tension (Watchers Disagreeing About Watchers)")
    print("-" * 62)
    print()
    print("We can compare, for example, the supervisor's view of Alice with")
    print("the director's view of Alice. If they diverge, the cross-level")
    print("contradiction goes up. If they align, it drops toward zero.")
    print()
    print("The useful thing is that it's the same calculation as before—")
    print("just applied to differently labeled objects.")
    print()

    obs = Observatory.create(symbols=["Hire", "No_Hire"])
    hire, no_hire = obs.alphabet
    candidate = obs.concept("Candidate")

    # Set up the basic reviewer perspectives
    alice = obs.lens("Alice", symbols=["Reliable", "Unreliable"])
    bob = obs.lens("Bob", symbols=["Consistent", "Inconsistent"])

    with alice:
        alice.perspectives[candidate] = {hire: 0.8, no_hire: 0.2}
    with bob:
        bob.perspectives[candidate] = {hire: 0.3, no_hire: 0.7}

    # The supervisor thinks highly of Alice
    supervisor = obs.lens("Supervisor", symbols=["Trustworthy", "Questionable"])
    reliable, unreliable = alice.alphabet
    consistent, inconsistent = bob.alphabet

    with supervisor:
        supervisor.perspectives[alice] = {reliable: 0.95, unreliable: 0.05}
        supervisor.perspectives[bob] = {consistent: 0.6, inconsistent: 0.4}

    # The director has the opposite view of Alice
    director = obs.lens("Director", symbols=["Competent", "Incompetent"])
    trustworthy, questionable = supervisor.alphabet

    with director:
        director.perspectives[alice] = {reliable: 0.05, unreliable: 0.95}  # Complete reversal
        director.perspectives[supervisor] = {trustworthy: 0.5, questionable: 0.5}

    print("Opposite views:")
    print("• Supervisor: Alice is 95% reliable")
    print("• Director: Alice is 95% unreliable")
    print()

    # Measure the resulting tensions
    supervisor_behavior = supervisor.to_behavior()
    director_behavior = director.to_behavior()
    cross_behavior = (supervisor | director).to_behavior()

    print("The contradiction measurements:")
    print(f"• Supervisor internal: {supervisor_behavior.contradiction_bits:.3f} bits")
    print("  → Still perfectly consistent within her own worldview")
    print(f"• Director internal: {director_behavior.contradiction_bits:.3f} bits")
    print("  → Also internally consistent, despite disagreeing")
    print(f"• Cross-level: {cross_behavior.contradiction_bits:.3f} bits")
    print("  → Substantial organizational tension: about 1/4 of a clarifying question")
    print()
    print("That 0.239 bits value is not arbitrary; it's the price—in clean,")
    print("additive information units—of carrying a sharp internal split.")
    print()


def tracking_tension_gradients():
    print("VI. Opposite Views and Gradients (What '0.239 bits' Is Telling You)")
    print("-" * 68)
    print()
    print("If we slide the director's belief from agree → neutral → disagree,")
    print("the contradiction moves smoothly. That smooth curve is the tension")
    print("gradient. It's how we quantify 'kind of disagree' versus 'at odds.'")
    print()

    obs = Observatory.create(symbols=["Hire", "No_Hire"])
    candidate = obs.concept("Candidate")

    # Set up Alice and the supervisor's fixed view
    alice = obs.lens("Alice", symbols=["Reliable", "Unreliable"])
    reliable, unreliable = alice.alphabet

    supervisor = obs.lens("Supervisor", symbols=["Trustworthy", "Questionable"])
    with supervisor:
        supervisor.perspectives[alice] = {reliable: 0.95, unreliable: 0.05}

    director = obs.lens("Director", symbols=["Competent", "Incompetent"])

    print("Supervisor fixed: Alice is 95% reliable")
    print("Director sliding from agreement to disagreement:")
    print()

    # Sweep through different levels of director agreement/disagreement
    for p_reliable in [1.0, 0.75, 0.5, 0.25, 0.05, 0.0]:
        with director:
            director.perspectives[alice] = {reliable: p_reliable, unreliable: 1 - p_reliable}

        cross_behavior = (supervisor | director).to_behavior()
        print(f"  Director sees Alice as {p_reliable:.0%} reliable → {cross_behavior.contradiction_bits:.3f} bits")

    print()
    print("The smooth curve from near 0.0 to 0.354 bits maps the landscape")
    print("of disagreement. This isn't just 'they disagree'—it's a precise")
    print("measurement of how much information work it takes to reconcile")
    print("their positions, or the cost of leaving them unreconciled.")
    print()
    print("Why the max isn't 1 bit:")
    print()
    print("We say 'disagreement,' and folks expect a hard ceiling of 1 bit—the")
    print("cost of a clean yes/no. But here the ceiling floats. It's capped by")
    print("how uncertain the first lens is.")
    print()
    print("If the Supervisor is 95% sure Alice is reliable, there isn't a full")
    print("bit of wobble to argue over. The Director can push back, sure, but")
    print("they're only tugging on that thin 5% slack. Less slack, less possible")
    print("contradiction.")
    print()
    print("Think in coin flips: if the Supervisor were 50/50, that's a fair")
    print("coin—max 1 bit of surprise. At 95/5, the coin is loaded. Even perfect")
    print("opposition can't wring out a full flip's worth of information. The")
    print("most you can get is the coin's own uncertainty: about 0.354 bits")
    print("(the entropy of a 95/5 split).")
    print()
    print("One-line intuition: Contradiction can't exceed the uncertainty that")
    print("exists. Near-certainty leaves too little room to disagree.")
    print()


def why_it_works():
    print("VII. Why It Works")
    print("-" * 17)
    print()
    print("Step back and notice what just happened. We measured contradiction")
    print("between reviewers using information theory. Then we measured")
    print("contradiction between their supervisors using the same information")
    print("theory. The framework didn't need modification. It just worked.")
    print()
    print("This isn't coincidence. When observers become observed, the")
    print("mathematical structure stays the same. Whether we're looking at:")
    print()
    print("• Three people evaluating a candidate")
    print("• Two executives evaluating those three people")
    print("• A board evaluating those executives")
    print()
    print("The pattern holds. The mathematics only cares about the structure:")
    print("observers, observables, and the distributions that connect them.")
    print("The organizational chart is irrelevant.")
    print()
    print("This means any system where perspectives can clash can be")
    print("measured using the same tools, regardless of how many levels")
    print("deep the observation goes.")
    print()
    print("We can watch the watchers with the same math we use to watch")
    print("the world. New level, same lens.")


if __name__ == "__main__":
    watching_the_watchers()
    when_hierarchies_clash()
    tracking_tension_gradients()
    why_it_works()
