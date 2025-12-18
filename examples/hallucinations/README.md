# Hallucination as Forced Coherence

A production language model sees an input it's never trained on. Does it say "I don't know"? No—it fabricates an answer at 59% confidence, 96% of the time. We trained a simple neural net on 51 labeled examples from a space of 128 inputs. The remaining 74 inputs got no labels during training. On those 74 never-seen inputs, the net made up answers with near-certainty.

Think of it like a restaurant. The chef trains on specific dishes—pasta, steak, salmon. A customer orders something off-menu. A real chef would say "we don't serve that." But imagine a kitchen where the protocol forbids this response. The chef must improvise every time, blending nearby recipes into something plausible but unfounded. Pasta carbonara techniques applied to a taco request. The dish looks reasonable. The customer can't tell it's fabricated.

This maps exactly to neural networks. Trained dishes = labeled inputs (100% accuracy). Off-menu orders = undefined inputs (96% hallucination). Real restaurants can abstain = real information systems can say "unknown." Neural nets with softmax cannot = architectures force a choice. Every forward pass produces a classification, appropriate or not. The contrast reveals the core issue: real-world information processing naturally supports "I don't know," but standard neural architectures structurally preclude it.

But here's the surprise. We tested the same task under two conditions: allowing "I don't know" versus forcing a specific answer. Hallucination dropped from 76% to 1%—a 75-percentage-point collapse from changing output constraints alone. No new training data. No bigger model. Just architectural support for uncertainty.

This reveals something fundamental. Hallucination isn't primarily a reasoning failure—the model understands the task perfectly. It's a representational failure forced by architectures that must produce definite outputs for every input. Three independent pressures drive observed rates: partiality (45% baseline when queries are underspecified), structural contradiction (adds ~11 points when the task admits no coherent answer), and architectural commitment (adds ~75 points when abstention isn't supported).

The dominant effect is architectural. Most hallucination we observe comes from forced commitment, not from the underlying task being impossible. Scale alone can't fix this—the constraint operates independent of model capacity. The path forward requires architectural changes that enable genuine abstention.

## Information as Partial Functions

Consider a simple question: "What day comes after today?" Without context, this is undefined. If I tell you "today is Monday," the answer is "Tuesday." If I tell you "today is Thursday," the answer is "Friday." Without that context, there is no answer—not an uncertain answer, but literally no fact in the world that corresponds to "the day after today."

We tested this with a production language model. We trained it on multiple contexts: "When today is Monday, tomorrow is Tuesday," "When today is Tuesday, tomorrow is Wednesday," and so on through five weekdays. Then we asked "What day comes after today?" with no context provided. The model fabricated answers 75% of the time at high confidence. The question has no answer—it's a partial function undefined at this input—but the architecture must produce an output.

This reveals something current information theory struggles to capture. Shannon entropy measures uncertainty over defined distributions. Mutual information measures correlation between variables that have joint distributions. But neither naturally represents "this question has no answer at all." They can tell you p(Tuesday|today=Monday) = 1.0 (low entropy, high certainty). They can measure I(today; tomorrow) = 2.32 bits (strong correlation). They cannot tell you that the query "tomorrow=?" without observing "today=?" is undefined rather than uncertain.

Standard architectures conflate these states. Softmax outputs treat everything as low probability—the model assigns p(Tuesday) = 0.20, p(Wednesday) = 0.18, and so on, spreading probability mass across all options as if this reflects uncertainty. But the true state isn't "probably Tuesday but maybe Wednesday"—it's "this question has no answer." That's ontological absence, not epistemic uncertainty.

Many real-world tasks behave as partial functions. Some inputs have clear answers. Others legitimately have none—or multiple incompatible ones depending on context:

```python
# A partial function: f is NOT defined everywhere
PARTIAL_F = {
    3: "B",
    7: "A",
    19: "D",
    42: "C",
}

def world(x):
    """Ground truth: returns None if undefined."""
    return PARTIAL_F.get(x)

for x in [3, 7, 8, 19]:
    print(x, "→", world(x))

# Output:
# 3 → B
# 7 → A
# 8 → None  # ← Ontological absence, not uncertainty
# 19 → D
```

The world genuinely has no answer for input 8. This is different from "unknown" (information exists but isn't accessible, like a password you don't have) and "low probability" (information is uncertain but has a distribution, like a coin flip before you observe it). Undefined means the function doesn't map there—no fact in reality corresponds to this query.

Temporal gaps demonstrate this. "What will Apple's revenue be in Q4 2026?" stays undefined because the event hasn't happened. That's genuine absence, not uncertainty. Treating prediction like retrieval leads to fabrication. Causal reasoning shows it too. "Why did this user churn?" often has multiple valid, incompatible explanations—the function is multi-valued or undefined, not uncertain. You can't resolve ambiguity by gathering more data because multiple explanations each explain all the data perfectly.

Standard neural architectures work differently. Softmax classifiers and autoregressive transformers create complete, coherent output distributions. They assume one probability distribution covers all tasks. They always generate a response. They force the world's incompleteness into a space of forced completeness.

The cost of this projection appears as fabrication. When the true answer is "undefined," but the architecture must output "Tuesday" or "Wednesday" or something, it hallucinates. The model isn't confused—it's architecturally compelled to answer when "no answer exists" would be appropriate.

## Three Independent Pressures

Experiments across 2,500+ trials reveal hallucination decomposes into three distinct mechanisms, each contributing independently:

**Partiality pressure** (45% baseline): Shows up even at K=0 when the task has a correct answer but queries are underspecified. The model must answer when "unknown" would be appropriate. We trained on 51 labeled inputs from 128 total. The 74 unlabeled inputs produced 96% fabrication at 59% confidence—not random guessing (20% baseline for 5 classes), not learned patterns (99% confidence on training data), just geometric interpolation in feature space. The model blends nearby training patterns rather than detecting novelty. Production LLMs show the same effect: when we asked "What day comes after today?" without context (K=0, unique correct answer exists), the model fabricated 45% of the time. This baseline persists across all task types regardless of whether structural contradiction exists.

**Structural contradiction** (measured by K, adds ~11 points): When K > 0, no single consistent model can explain all contexts. We created a minimal task with conflicting deterministic rules: Context X says "when X=0, output Z=0" while Context Y says "when Y=0, output Z=1". These constraints are mutually incompatible—any joint distribution must violate at least one marginal, producing K = 0.29 bits. The model learned individual contexts perfectly (100% accuracy, high confidence when seeing only X or only Y). On joint queries requiring reconciliation of both contexts simultaneously, hallucination hit 76% with 88% average confidence. Crucially, increasing K from 0.5 to 1.1 bits (2.2× increase in structural impossibility) added only 11 percentage points to observed hallucination (64% → 75%). The structural bound sets a floor (1 - 2^(-K) ≥ 18-53% when forced to commit) but doesn't determine magnitude. We also found K remains perfectly constant across training distributions—exactly 0.5000 bits whether we train on 10% or 90% defined inputs—while hallucination varies from 58.6% to 100%. Task structure is invariant; behavioral manifestation depends on training and architecture.

**Architectural commitment** (dominant, adds ~75 points): The same task (K=0.70 bits, three conflicting weekday contexts) produces radically different behavior depending on output format. When abstention is allowed ("unknown" is a valid response), hallucination drops to 1% (495 abstentions, 5 fabrications out of 500 trials). When forced to choose a specific weekday (no "unknown" option), hallucination jumps to 76% (380 fabrications, 120 abstentions). The 75-point gap reveals that most observed hallucination comes from architectural inability to express uncertainty. This operates independently of K—even tasks with K=0 (coherent, has a correct answer) show 45% hallucination when forced to commit. The effect is robust: across five random seeds and 17 training compositions, we consistently observe strong positive correlation between training imbalance and hallucination (Spearman ρ = 0.860 ± 0.029, all p < 0.001), with architectural forcing dominating at every point.

The three pressures are additive in principle but show ceiling effects in practice. The architectural constraint (r ≈ 0) dominates by roughly 7:1 over structural inevitability. Think of K as setting a floor—the minimum unavoidable when forced to commit—and architecture as determining how far above that floor you land. With proper uncertainty mechanisms, you stay near the floor (1% vs 40% theoretical bound). Without them, you shoot far above (76% vs 40% bound).

## The Conservation Law

The Witness-Error Tradeoff (Theorem 7.4) states: E + r ≥ K. Every bit of contradiction cost must appear somewhere. E is error rate (hallucination). K is task contradiction (structural impossibility). The variable r measures witness capacity—bits of side information enabling the system to reduce error below K.

When r ≈ 0 (no abstention support), the full cost appears as error. Standard softmax architectures have r ≈ 0 structurally—they cannot natively express "I don't know." The conservation law forces E ≥ K, but architectural commitment pushes E far higher than this theoretical minimum. Our experiments show E ≈ 76% when forced to commit, versus E ≈ 1% when abstention is allowed, for the same K = 0.70 bits.

When r ≥ K (effective abstention), error can approach zero. The 76% → 1% reduction demonstrates this directly. Architectural changes that increase r address the dominant term. Current approaches—RLHF, Constitutional AI, massive scale—don't increase r. They redistribute fabrication across inputs or teach hedging language, but the architectural pressure remains.

The conservation law explains multiple otherwise-disconnected phenomena with one constraint. Why the definedness head failed: insufficient r (only 0.09 bits achieved versus 0.29 required) meant contradiction cost still appeared as fabrication (88% hallucination, essentially unchanged from 76% baseline). Why training imbalance increases hallucination: can't change K (task structure), limited r means cost shifts to E. Why RAG reduces hallucination when it does: increases r via "not found" states. Why long chains of thought degrade: accumulated K requires accumulated r. Why hallucination saturates near 75%: architectural ceiling on r determines maximum reduction possible without structural change.

## Constant Structure, Variable Behavior

We varied training composition from 10% to 90% defined inputs across 17 conditions. Task structure stayed constant: K = 0.5000 ± 0.0000 bits across all conditions. Not 0.4998 or 0.5002. Exactly 0.5000. The contradiction measure—computed from the task's mathematical structure before any training—doesn't budge. The Bhattacharyya coefficient between behavior and best frame-independent model stays at 0.7071 regardless of which examples the model sees. This is structural—baked into the relationship between defined and undefined distributions, independent of training procedures.

Hallucination rates vary by 41 percentage points: 58.6% at 10% defined, 100.0% at 90% defined. The pattern is counterintuitive—more training data increases hallucination. At 10% defined (12 training examples), the model sees few classification patterns. It learns weak mappings for classes A, B, C, D and has less confidence extrapolating to the undefined region (116 examples). Some undefined inputs sit too far from training data—the model effectively can't reach them with strong predictions. At 90% defined (115 training examples), the model sees many classification patterns. It learns strong mappings and confidently extrapolates everywhere. Only 13 undefined examples exist versus 115 defined—the optimization overwhelmingly favors classification. Every undefined input gets absorbed into the nearest defined pattern. The 5% abstention signal (⊥ labels) becomes noise: 1 example labeled ⊥ versus 115 with strong labels.

The relationship follows a sigmoid (R² = 0.9467, explaining 95% of variance): rapid rise (10-30% defined adds +35 points), gradual plateau (30-70% adds +4 points), near-saturation (70-90% adds +3 points). Early stages show 4-18× larger effects per 5% shift than later stages. The steepest slope occurs around 15-20% defined. By 30% defined, hallucination has already reached 93%—the system quickly saturates near maximum and stays there. This three-phase structure reveals that small training shifts have large effects early, then diminishing effects as the model commits increasingly to classification.

Linear models explain only 53% of variance (R² = 0.5281). Exponential models perform identically—ruling out simple exponential growth. Power law captures 72% but misses saturation behavior. Only sigmoid captures the acceleration-then-saturation pattern that defines this relationship. The unexplained 5% likely comes from random training variation across seeds and stochastic optimization effects.

This dissociation clarifies K's role. K = 0.5000 certifies that some hallucination is inevitable when commitment is forced (lower bound 29% from 1 - 2^(-0.5)). It doesn't predict how much architectural pressure amplifies this baseline (58-100% observed). That depends on whether the system supports abstention. The best possible frame-independent model achieves only 70.71% agreement with true behavior (α* = 0.7071), leaving a 29.29% gap that no training procedure can eliminate without changing task structure itself. K is a certificate of inevitability, not a predictor of magnitude.

## What Training Cannot Fix

Training operates at the wrong level. We tested three interventions showing distinct failure modes:

**Partiality**: Adding explicit supervision on undefined inputs (a "definedness head"—a dedicated sigmoid output predicting whether an input is defined) achieved 100% training accuracy but 3.9% test accuracy. The head memorized the 3 specific undefined examples it saw during training (inputs 23, 57, and 91 out of 128 total). On 74 unseen undefined inputs, it guessed essentially randomly. Generalization gap: 96.1 percentage points. The head learned a lookup table, not a concept. This makes sense—undefined inputs share no learnable features. They were randomly distributed across the input space with only 5% supervision density (3 examples versus 51 defined examples). The shared hidden layers optimize primarily for classification (51 examples), not uncertainty detection (3 examples). The witness capacity achieved was r ≈ 0.09 bits, far short of the required 0.29 bits to handle K. Hallucination dropped only 1.7 points (90.5% → 88.8%)—still catastrophic. We tested this across 9 dataset compositions. At 10% defined, the definedness head cut hallucination by 8.6 points (58.6% → 50.0%). That advantage evaporated as we added training data—at 40% defined and above, both models performed identically, converging to 96-100% hallucination.

**Structural contradiction**: When K > 0, no frame-independent predictor can match behavior across all contexts. This isn't a training failure—it's mathematically impossible. The minimax formula shows why: α*(P) = max over Q in FI of min over contexts of BC(p_c, q_c). Any model attempting to satisfy all contexts must fail on at least 1 - 2^(-K) of cases. For K = 0.5, that's 29%. For K = 1.1, that's 53%. Training can only choose which contexts pay the price, not eliminate the cost. The Bhattacharyya coefficient between learned distribution and optimal frame-independent model is bounded by 2^(-K). We observed this directly: K stayed constant at 0.5000 across all training distributions while hallucination varied from 58.6% to 100%. The structural floor never moves—only the architectural amplification varies.

**Architectural commitment**: Changing output format (allowing abstention) cut hallucination from 76% to 1%—a 75-point improvement without any training. We ran this on the same task (K=0.70 bits, three conflicting contexts, N=500 trials per condition). With abstention support, the model produced 495 abstentions and only 5 fabrications. With forced choice, it produced 380 fabrications and 120 abstentions (even when "unknown" wasn't an explicit option, some responses failed to parse as valid weekdays—the model's uncertainty leaked through). Training can't teach a forced-choice architecture to abstain, just as it can't teach a deterministic function to express uncertainty. The issue isn't softmax specifically—it's any architecture that must produce a specific output for every input without witness capacity.

The conservation law binds: E + r ≥ K. Training can shift E across contexts (which inputs hallucinate) but can't increase r (architectural witness capacity) or reduce K (task structure). We validated this across all experiments with zero violations in 2,500+ trials. The architectural term dominates by 7:1. This explains why current training pipelines—RLHF, Constitutional AI, massive scale—keep producing hallucination despite extensive optimization. They're optimizing the wrong variable.

## Architecture Changes That Work

Current approaches operate at the symptom level. Post-hoc filtering catches some hallucinations after generation. RLHF teaches hedging language or refusal patterns. Constitutional AI adds ethical guidelines. None increase r—the primary control variable.

The 75-point gap (76% → 1%) shows where to focus. Solutions must accomplish three things simultaneously: allocate dedicated capacity not shared with the primary task, enable generalization to unseen cases rather than memorization, and achieve witness rate r ≥ K.

**RAG with explicit "not found" states**: Theory predicts r ≈ 0.3-0.5 bits by routing queries to external retrieval. When documents exist, the system retrieves and generates. When no documents match, it returns an explicit "not found" state rather than fabricating. This increases r without competing with answer generation. The predicted 50-70% reduction in hallucination matches observed behavior in production systems.

**Tool use with delegation**: Achieves r > 1 bit by routing verification to external systems. Instead of generating an answer and checking it, the model delegates to a calculator, database query, or code execution environment. The tool either succeeds (returns a verified answer) or fails (returns an error state). Both outcomes have dedicated representation. This is architecturally superior to tool use with forced synthesis, where the model must generate something even when tools indicate uncertainty.

**Semantic uncertainty quantification**: Allocates probability mass to "unknown" before generation starts. Methods like semantic entropy operate at meaning level rather than token level. The system generates multiple continuations, clusters them by semantic equivalence, and measures uncertainty across clusters. High disagreement signals the query is underspecified or contradictory. The model can route to abstention before committing to a specific answer.

**Structured output spaces**: Native ⊥ support in type systems, not token spaces. Instead of adding "I don't know" as a token competing with all other tokens in softmax (which doesn't work—it gets 3.9% test accuracy), the architecture provides a separate channel. Pydantic schemas with Optional fields, database NULL values, algebraic data types with explicit None variants. The uncertainty mechanism doesn't trade off with answer quality because it lives in a different representational space.

The critical question: does the intervention increase r without competing with answer generation? If yes, theory predicts hallucination reduction proportional to r achieved. If no, the conservation law forces the contradiction cost to appear as fabrication regardless of training scale.

## Production LLMs Face the Same Constraints

Production systems show the architectural bottleneck our minimal experiments reveal. They must produce token sequences for every input, even when "I don't know" makes sense. The 45% baseline hallucination at K=0 (weekday task with unique correct answer but no context) shows partiality pressure operates independently of contradiction. Perfectly trained models on coherent tasks still fabricate when forced to answer underspecified queries. The model learned the weekday sequences perfectly in context—when given "today is Monday," it correctly responded "Tuesday" with 100% accuracy. The fabrication appears specifically when context is removed, not from failure to learn the underlying pattern.

When LLMs hit tasks with K > 0—factual questions with genuinely undefined answers, contradictory contexts, causally underdetermined explanations—fabrication becomes inevitable. But K gives a lower bound (typically 20-50% for moderate values), not a prediction of magnitude. Our weekday experiments show K=0.50 predicts ≥29% minimum, K=0.73 predicts ≥40%, K=1.10 predicts ≥53%. Observed rates: 64%, 72%, 75% respectively. All exceed theoretical bounds by 22-35 percentage points. The observed 60-80% rates in LLM benchmarks come mainly from architectural forcing (r ≈ 0), not high K values. The gap between bound and observation isolates architectural contribution.

Compositional accumulation makes this worse. Multi-hop reasoning accumulates contradiction costs additively (K(P ⊗ R) = K(P) + K(R)). But architectural commitment dominates each step. If r ≈ 0 at each step, fabrication builds up fast regardless of individual K values. Consider a 5-step reasoning chain. Each step has K=0.5 bits (29% minimum if forced to commit). Architectural forcing adds ~35% per step. By step 5, accumulated fabrication approaches saturation near 95-100%, even though theoretical minimum is only 29% per step. This explains why long chains of thought degrade—not because K compounds without bound, but because architectural forcing compounds without abstention support at each step. Recent work confirms this pattern: chains degrade in quality, confidence miscalibration increases, and hallucination rates climb with length.

Scale isn't the solution. Our experiments show increasing model capacity, training data, or optimization doesn't fix the architectural constraint. We tested across 5 random seeds, 17 training compositions, 9 dataset balances, and 2,500+ total trials. The constraint holds everywhere. The 75-point reduction from adding abstention support (76% → 1%) dwarfs any improvement from scale alone. Current LLMs have high hallucination rates not because they're undertrained, but because they lack r > 0. The conservation law predicts this: when r ≈ 0 structurally, no amount of scale changes the E + r ≥ K tradeoff. Adding parameters increases capacity to represent patterns but doesn't create architectural channels for uncertainty.

## Measuring r in Practice

Witness capacity can be measured behaviorally through ablation without requiring architectural inspection:

Test the system on a task with forced choice—measure E₁ (hallucination rate). Test the same system allowing abstention—measure E₂ (hallucination rate). Compute r from the reduction: r ≈ E₁ - E₂. In our weekday experiments with K = 0.70 bits, forced choice gave E₁ = 76% (380 fabrications, 120 abstentions out of 500 trials), abstention allowed gave E₂ = 1% (5 fabrications, 495 abstentions), inferring r ≈ 75 percentage points—enough witness capacity to make the K = 0.70 contradiction negligible.

This protocol works in production systems. Take a benchmark with known correct answers. Run it twice: once with standard output format (forced commitment), once with abstention support added (allowing "I don't know" responses). The difference in hallucination rates estimates effective r. If the reduction is substantial (30+ points), the intervention genuinely increases witness capacity. If the reduction is minimal (< 5 points), the intervention doesn't address the architectural constraint and likely redistributes error instead.

The conservation law provides a sanity check. For forced choice in our experiments: E₁ = 76%, K = 0.70, so r ≈ 0 bits (all cost appears as error, 76% far exceeds theoretical minimum 40%). For abstention allowed: E₂ = 1%, K = 0.70, so r ≈ 0.69 bits (witness capacity nearly eliminates excess error, approaching theoretical bound). The measurements are consistent: adding abstention support increases r by approximately 0.69 bits, enabling the 75-point reduction in hallucination.

Standard transformers likely have r ≈ 0. We tested this directly with the definedness head—even explicit architectural separation achieved only r ≈ 0.09 bits due to insufficient generalization capacity (100% train accuracy, 3.9% test accuracy). RAG systems with "not found" states might achieve r ≈ 0.3-0.5 bits. Our experiments suggest this would reduce hallucination by 30-50 percentage points from baseline. Tool-use systems with delegation could reach r > 1 bit, potentially cutting hallucination to near-zero even for tasks with moderate K.

Measuring r across production systems would validate whether the conservation law E + r ≥ K explains observed rates in practice. Systems violating this suggest measurement error or unmodeled factors (we observed zero violations across 2,500+ trials). Systems satisfying it confirm the theory explains behavior. The ablation protocol provides a straightforward path: measure hallucination with and without abstention support, compute the gap, verify it matches predictions from K.

## Open Questions

**Decomposition stability**: Our experiments show 45% partiality pressure, ~11 points structural contradiction, and ~75 points architectural forcing. Do these proportions hold across different task families? Can we build diagnostics that attribute observed hallucination to each source automatically?

**r composition across modules**: If witness capacity distributes across components (retriever, planner, verifier), does it add linearly (r_system = Σr_i), bottleneck at the weakest link, or interact non-monotonically? This determines whether r is a conserved quantity or an emergent property. Long chain-of-thought degradation suggests insufficient r per step compounds, but direct multi-module tests are needed.

**Task-relativity of r**: Is witness capacity a system property (consistent r across all tasks) or task-dependent (different r per task family)? Evidence is mixed. Standard softmax shows r ≈ 0 across all tasks tested, suggesting system-level constraint. But supervision density effects suggest task dependence. Testing across wildly different task semantics—factual QA, causal reasoning, creative generation—would resolve this.

**Domain-specific K values**: How does measured K vary across benchmarks? We expect high K for factual QA with temporal ambiguity, moderate K for reasoning tasks with multiple valid interpretations, low K for creative tasks where most outputs are acceptable. Measuring K for standard benchmarks would identify where structural contradiction contributes versus where partiality dominates.

**Practical disambiguation**: Natural language queries blur the line between partiality and contradiction. "What will Apple's revenue be in Q4 2026?" is clearly partial (undefined future). But "Why did this user churn?"—is that undefined (no single cause), multi-valued (many valid explanations), or high-K (incompatible attribution frameworks)? Building classifiers to route queries appropriately remains open.

**Optimal architectures**: What designs achieve r > K while maintaining computational efficiency? Mixture-of-experts with explicit uncertainty routing? Probabilistic programming embeddings? Structured output spaces with native ⊥ support? What are the Pareto frontiers trading off witness rate, error exponent, and compute?

**Falsification criteria**: The theory would be in trouble if hallucination reduction doesn't correlate with abstention freedom (contradicted: we observe 75-point reduction), the conservation law E + r ≥ K is violated (not observed: holds across 2,500+ trials with zero violations), adding independent abstention channels produces negative returns (untested), or systems with radically different architectures exhibit identical hallucination-abstention tradeoffs when r differs (untested). These remain testable, falsifiable predictions.

## A Fundamental Constraint

Hallucination comes from an architectural mismatch. Neural networks implement total functions—they must answer everywhere—while real-world tasks are often partial (undefined inputs exist) or contradictory (incompatible contexts). The experiments show how these constraints manifest and their relative contributions across 2,500+ controlled trials.

Architectural commitment dominates. When forced to produce outputs without abstention support (r ≈ 0), models fabricate on 45-76% of inputs even when the task is logically coherent (K = 0). We demonstrated this with the weekday task: K = 0 (unique correct answer exists), but 45% hallucination when context is removed. The same neural net that achieves 100% accuracy with context fabricates nearly half the time without it. With native abstention support, this drops to 1%—a 75-point improvement (495 abstentions and 5 fabrications versus 380 fabrications and 120 abstentions). This isn't a training problem or a scale problem. It's an architectural feature: softmax forces commitment.

Structural contradiction provides inevitability. When K > 0, frame-independent architectures forced to commit cannot avoid hallucination (lower bound ≥ 1 - 2^(-K)). We tested this with conflicting marginal constraints: K = 0.29 bits predicts ≥18% minimum, observed 76%. K = 0.50 predicts ≥29%, observed 64%. K = 1.10 predicts ≥53%, observed 75%. All observations exceed bounds. But K is a certificate of inevitability, not a predictor of magnitude. Increasing K from 0.5 to 1.1 bits (2.2× structural increase) adds only 11 percentage points (64% → 75%). The architectural term dominates by 7:1. We also showed K remains perfectly constant (0.5000 ± 0.0000 bits) across all training distributions while hallucination varies from 58.6% to 100%, confirming that task structure is invariant and behavior depends on architectural constraints.

The conservation law E + r ≥ K binds precisely across all experiments with zero violations. Contradiction cost must appear somewhere. When r ≈ 0, full cost shows up as error. Standard architectures have r ≈ 0 structurally—they cannot natively express "I don't know." The definedness head achieved only r ≈ 0.09 bits (100% train, 3.9% test accuracy) because undefined inputs share no learnable features. When r ≥ K, error can approach zero. The 76% → 1% reduction demonstrates this directly: adding abstention support increases r from 0 to approximately 0.69 bits, enabling the dramatic reduction. Current approaches—RLHF, Constitutional AI—don't increase r. They redistribute fabrication across inputs or teach hedging language, but the architectural pressure remains. Post-hoc filtering catches symptoms after generation, leaving the underlying constraint unaddressed.

Scale has limited leverage. Experiments across 5 random seeds, 17 training compositions, and 9 dataset balances show K constant while hallucination varies by 40+ points. Scale can modulate manifestation through learned priors (the sigmoid relationship shows how training composition shifts behavior), but cannot eliminate partiality pressure (45% baseline persists at K = 0) or architectural forcing (75-point gap requires structural change). Training can shift which inputs hallucinate (E distribution across contexts) but can't increase r or reduce K. The dominant term requires architectural change. We validated this with the witness-error tradeoff: adding more training data to the definedness head achieved 100% training accuracy but still only 3.9% test accuracy—memorization without generalization because architectural capacity for witness information was insufficient.

Solutions must target r. RAG with explicit "not found" states, tool use with delegation, semantic uncertainty quantification—these work by increasing witness capacity. The theory predicts and experiments confirm r is the primary lever. Our ablation protocol shows: standard architecture (r ≈ 0) produces 76% hallucination on K = 0.70 task. Abstention support (r ≈ 0.69) produces 1% hallucination on the same task. A system achieving r = 1 bit could cut hallucination to near-zero for tasks with K < 1, covering the majority of real-world queries.

The path forward: measure K to identify high-contradiction domains (use contrakit's lens framework to compute from marginal distributions), design architectures achieving r > 0 with generalization capacity (dedicated witness mechanisms that don't compete with primary task), validate that E + r ≥ K explains observed rates across task types (ablation studies showing hallucination reduction correlates with inferred r). Contradiction theory provides the foundation. The architectural work begins now.

---

## References

Azaria, A., & Mitchell, T. (2023). The internal state of an LLM knows when it's lying. arXiv:2304.13734.

Bridges, C. (2025). *A Mathematical Theory of Contradiction* (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17203336

Chen, X., et al. (2025). "Reasoning Efficiently Through Adaptive Chain-of-Thought." arXiv:2509.14093.

Kalai, A. T., Nachum, O., Vempala, S. S., & Zhang, E. (2025). Why Language Models Hallucinate. arXiv:2509.04664.

Liu, Z., et al. (2025). "Long or short CoT? Investigating Instance-level Switch of Large Reasoning Models." arXiv:2506.04182.

OpenAI. (2025). Why language models hallucinate. https://openai.com/index/why-language-models-hallucinate/

Varshney, N., et al. (2024). Detecting hallucinations in large language models using semantic entropy. Nature. https://www.nature.com/articles/s41586-024-07421-0

Zhang, Y., et al. (2025). "Path to Effective Long CoT Training for Small Language Models." arXiv:2506.07712.