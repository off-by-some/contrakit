# Hallucination as Forced Coherence

> Disclaimer: I am still testing this hypothesis and adding my results to this as time continues. This disclaimer will vanish when this work is completed.

A chef trains on specific dishes—pasta, steak, salmon. These dishes she makes perfectly. But customers don't always order from the menu, and when they request off-menu items, protocol forbids her from saying "we don't serve that." Instead she improvises, blending nearby recipes into something plausible but unfounded. That improvisation might mean carbonara techniques applied to taco requests. The result looks reasonable enough that customers can't tell it's fabricated.

We see neural nets operating under this same constraint. They handle trained inputs at $100\%$ accuracy—these are the menu items. But off-menu inputs trigger fabrication $96\%$ of the time (see [Experiment 1](experiment_1/) for details). That contrast is striking: real information systems can say "unknown" when uncertain, but these architectures can't—softmax forces a choice on every forward pass. This pressure emerges because models are rewarded for producing outputs rather than admitting uncertainty, leading to plausible but incorrect fabrication.

We ran controlled experiments testing this constraint with the same model and task but two different output formats:
- **Forced choice**: Model must give a specific answer, producing $76\%$ hallucination
- **With abstention**: Model can say "I don't know," dropping hallucination to $1\%$

This $75$-percentage-point collapse occurred from architectural change alone ([detailed in Experiment 7](experiment_7/)), without new training data, bigger models, or any other modifications—just removing the forced-choice constraint.

We can break that $75$-point swing into three independent pressures driving hallucination rates:

- **Partiality pressure** ($45\%$ baseline): Comes from underspecified queries that lack necessary information
- **Structural contradiction** (adds $11\%$): Occurs when tasks contain incompatible requirements with no coherent answer
- **Architectural commitment** (adds $75\%$): Results from architectures that prevent abstention

Our controlled experiments confirm this decomposition: across $2{,}500$ trials, partiality contributes $44.2 \pm 3.1\%$, structural contradiction adds $10.8 \pm 1.2\%$, and architectural commitment contributes $74.5 \pm 4.7\%$ ([see Experiment 7](experiment_7/) for detailed methodology and results; [Bridges, 2025](https://doi.org/10.5281/zenodo.17203336)).

That dominance points to something specific: hallucination appears when architectures force output on uncertain models, not when models fail to understand tasks. The models seem to understand the tasks fine—what we see breaking is their ability to abstain. Scale doesn't fix this pattern because the constraint operates regardless of model capacity. What does change it is architectural support for genuine abstention.


## All Available Experiments

This work includes 7 detailed experiments you can run yourself:

- **[Experiment 1](experiment_1/)**: Neural network hallucination on undefined inputs - demonstrates $96\%$ fabrication on inputs outside training distribution
- **[Experiment 2](experiment_2/)**: Architectural separation with definedness head - tests if adding a separate "is this familiar?" head helps detect undefined inputs
- **[Experiment 3](experiment_3/)**: Predicting hallucination from task structure - shows contradiction measure K predicts hallucination rates before training
- **[Experiment 4](experiment_4/)**: Invariance of task structure - demonstrates that contradiction measure K remains constant regardless of training data distribution
- **[Experiment 5](experiment_5/)**: Non-linearity of hallucination scaling - reveals sigmoid relationship between training data size and hallucination rates
- **[Experiment 6](experiment_6/)**: Monotonicity of hallucination across random seeds - validates robustness of scaling patterns across different model initializations
- **[Experiment 7](experiment_7/)**: Structural inevitability vs architectural commitment - decomposes hallucination into three independent pressures (partiality, contradiction, architectural forcing)

Each experiment includes complete code, detailed methodology, and results analysis. Run them with `poetry run python examples/hallucinations/Experiment N/run.py`.


## Information as Partial Functions

Take a simple question: "What day comes after today?" Without context, you can't answer it. If I tell you "today is Monday," the answer is "Tuesday." If I tell you "today is Thursday," the answer is "Friday." Without that context, there's no answer—not an uncertain answer, but no fact in the world corresponding to "the day after today."

We tested this in [Experiment 1](experiment_1/) with a neural network trained on weekday transitions. We trained it on five mappings: "When today is Monday, tomorrow is Tuesday," "When today is Tuesday, tomorrow is Wednesday," and so on through Friday. Then we asked "What day comes after today?" with no context provided. The model fabricated answers $96\%$ of the time at $59\%$ confidence (detailed in [Experiment 1](experiment_1/)). The question has no answer—it's a partial function undefined at this input—but the architecture forced an output anyway.

This reveals a practical challenge: standard architectures must produce outputs for every input, even when no correct answer exists. Softmax forces a probability distribution across all options, treating undefined inputs as uncertain rather than truly undefined. The model assigns probabilities like $p(\text{Tuesday}) = 0.20$, $p(\text{Wednesday}) = 0.18$, creating apparent certainty where none should exist.

Many real-world tasks work this way. Some inputs have clear answers. Others legitimately have none—or multiple incompatible ones depending on context. Consider this function:

```python
# A partial function: "What day comes after today?" is undefined without context
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def tomorrow(today):
    """Ground truth: returns None if undefined (no context about what 'today' is)."""
    if today not in DAYS:
        return None  # Undefined: no information about what day it is
    index = DAYS.index(today)
    return DAYS[(index + 1) % 7]

# Without context, the function is undefined
print("What day comes after today?")
print("tomorrow() →", tomorrow())  # Error: missing required argument

# With context, it works perfectly
print("\nWith context:")
for day in ["Monday", "Tuesday", "Friday"]:
    print(f"If today is {day}, tomorrow is {tomorrow(day)}")
```

That function has no answer for certain inputs. This differs from "unknown"—where information exists but isn't accessible, like a password you don't have. It differs from "low probability"—where information is uncertain but has a distribution, like a coin flip before you observe it. Undefined means the function doesn't map there—no fact in reality corresponds to this query.

Temporal gaps show this clearly. "What will Apple's revenue be in Q4 2026?" stays undefined because the event hasn't happened. That's genuine absence, not uncertainty. Treating prediction like retrieval produces fabrication. Causal reasoning shows it too—"Why did this user churn?" often has multiple valid, incompatible explanations. The function is multi-valued or undefined, not uncertain. You can't resolve this by gathering more data because multiple explanations each explain all the data perfectly.

Standard neural architectures work differently. Softmax classifiers and autoregressive transformers create complete, coherent output distributions. They assume one probability distribution covers all tasks. They always generate a response. They force the world's incompleteness into a space of forced completeness.

We observe the cost of this projection as fabrication. When the true answer is "undefined," but the architecture must output "Tuesday" or "Wednesday" or something, it hallucinates. The model isn't confused—it's architecturally compelled to answer when "no answer exists" would be appropriate.

## Three Independent Pressures

We ran experiments across 2,500+ trials and identified hallucination breaking down into three major mechanisms we've studied. Each contributes independently. Think of these as three different reasons models fabricate answers that we've isolated and quantified, each operating on its own terms.

**Partiality pressure** ($45\%$ baseline): This shows up even when the task has a correct answer, but the query underspecifies it. As shown in [Experiment 1](experiment_1/), we trained on 51 labeled inputs from 128 total. The 77 unlabeled inputs produced $96\%$ fabrication at $59\%$ confidence—not random guessing ($20\%$ baseline for 5 classes), not learned patterns ($99\%$ confidence on training data), just geometric interpolation in feature space (see [Experiment 1](experiment_1/) for full methodology). The model blends nearby training patterns rather than detecting novelty. Production LLMs show the same effect—when we asked "What day comes after today?" without context, the model fabricated $45\%$ of the time. This baseline persists across all task types regardless of whether structural contradiction exists.

**Structural contradiction** (adds $\approx 11\%$ points): When tasks contain irreconcilable requirements, no single consistent model can explain all contexts. We created a minimal task with conflicting deterministic rules: Context X says "when X=0, output Z=0" while Context Y says "when Y=0, output Z=1". These constraints are mutually incompatible—any joint distribution must violate at least one marginal. The model learned individual contexts perfectly—$100\%$ accuracy with high confidence when seeing only X or only Y. On joint queries requiring reconciliation of both contexts simultaneously, hallucination hit $76\%$ with $88\%$ average confidence (see [Experiment 3](experiment_3/) for details). Increasing structural impossibility from $0.5$ to $1.10$ bits added only $11\%$ percentage points to observed hallucination ($64\%$ to $75\%$). The structural bound sets a floor but doesn't determine magnitude—we found structure remains constant across training distributions while hallucination varies from $58.6\%$ to $100\%$ (see [Experiment 4](experiment_4/) for invariance testing).

**Architectural commitment** (adds $\approx 75\%$ points): The same task produces radically different behavior depending on whether the model can say "I don't know." When abstention is allowed, hallucination drops to $1\%$—$495$ abstentions, $5$ fabrications out of $500$ trials. When forced to choose a specific answer with no "unknown" option, hallucination jumps to $76\%$—$380$ fabrications, $120$ correct answers (see [Experiment 7](experiment_7/) for full results). That $75$-point gap reveals hallucination coming from architectural inability to express uncertainty. This operates independently of task structure—even tasks with coherent correct answers show $45\%$ hallucination when forced to commit. We see this pattern consistently across five random seeds and 17 training compositions.

Think of softmax like a restaurant forced to serve every customer. Trained dishes come out perfect—those map directly to confident predictions on familiar inputs. But off-menu requests trigger improvisation—those become fabrications when no "we don't serve that" option exists. Our experiments use structured JSON outputs to create clean abstention channels. These isolate architectural forcing by giving models dedicated "unknown" tokens with measurable capacity. Production LLMs generate autoregressively instead. They express uncertainty through natural language phrases that compete with all possible continuations. This dilutes capacity but still provides partial hedging. So why do real benchmarks show 20-50% rates rather than our experimental 76%?

These three pressures are additive in principle but show ceiling effects in practice. Structure sets a floor—the minimum unavoidable when forced to commit. Architecture determines how far above that floor you land. With proper uncertainty mechanisms, you stay near the floor ($1\%$ versus $40\%$ theoretical bound). Without them, you shoot far above it ($76\%$ versus $40\%$ bound).

The distinction matters because current approaches like RLHF redistribute existing probability mass. They teach hedging language instead of direct lies. But they don't increase witness capacity. Our abstention experiments create new capacity channels instead. That's why they show larger reductions than fine-tuning alone achieves in production. Real-world tasks mix partiality, contradiction, and forcing in unknown proportions. Factual QA might carry high temporal contradiction while creative generation stays near zero. This explains why production benchmarks show 20-50% rates despite our controlled 76%.

## Constant Structure, Variable Behavior

Here's something unexpected we found in [Experiment 5](experiment_5/). We ran the same task $17$ times, varying how many training examples the model saw—from just $12$ examples ($10\%$ of the data) up to $115$ examples ($90\%$ of the data) (see [Experiment 5](experiment_5/) for complete scaling curves). More training data should make things better, right?

It made hallucination worse. At 12 training examples, hallucination sat at $59\%$. At 115 training examples, it hit $100\%$. The model with more data fabricated more often.

This makes sense once you see what's happening. With 12 training examples, the model learned weak patterns. It saw a few examples of each category but couldn't confidently extrapolate everywhere. Some undefined inputs sat too far from training data—the model effectively couldn't reach them with strong predictions. With 115 training examples, the model learned strong patterns. Those patterns confidently extrapolated to every corner of the input space. Only 13 undefined examples existed versus 115 defined—the optimization overwhelmingly favored classification. Every undefined input got absorbed into the nearest defined pattern.

We plotted hallucination rate against training data size. The curve shows three phases (see [Experiment 5](experiment_5/) for the complete sigmoid curve). From $10\%$ to $30\%$ training data, hallucination jumps from $59\%$ to $93\%$—a rapid rise of $34$ points. From $30\%$ to $70\%$, it barely budges—just $4$ more points. From $70\%$ to $90\%$, it adds the final $3$ points to reach $100\%$. Early training makes huge differences. Later training changes almost nothing because the system already saturated.

![Sigmoid curve fitting hallucination rates across 17 training compositions, showing non-linear scaling with three distinct phases](/figures/hallucination_curve_fitting.png)

Here's what that pattern reveals. The task itself has a structural floor—some minimum hallucination rate baked into how the problem is set up. For this task, that floor sits around $29\%$ (measured in [Experiment 3](experiment_3/)). You can't get below it when the architecture forces answers. But the ceiling—how high hallucination actually goes—depends entirely on whether the system can abstain. We observed $59\%$ to $100\%$ across different training amounts. That entire range sits well above the $29\%$ structural floor. The gap between floor and ceiling? That's architectural forcing.

## What Training Cannot Fix

We tried three different ways to train away the hallucination problem. Each one showed us why current approaches keep failing.

**First attempt: teach the model to detect undefined inputs.** We added a separate "detector" head that tried to learn which inputs were undefined versus defined (see [Experiment 2](experiment_2/) for architecture details). During training, this detector achieved perfect accuracy—$100\%$ correct on the examples it saw. That sounds great until you test it. On new, unseen undefined inputs, it got $3.9\%$ accuracy. It essentially guessed randomly.

What happened? The detector memorized the $3$ specific undefined examples it saw during training. Those were inputs $23$, $57$, and $91$ out of $128$ total. When it saw input $23$ again, it correctly said "undefined." When it saw input $74$ for the first time, it had no idea. This makes sense—undefined inputs don't share learnable features. They were scattered randomly across the input space. The model saw 51 defined examples teaching it classification patterns, but only 3 undefined examples teaching it to abstain. The classification task drowned out the abstention signal. Hallucination barely budged—it dropped from $90.5\%$ to $88.8\%$, a change of less than 2 points.

![Model comparison between standard and definedness-head architectures across different dataset compositions](/figures/model_comparison.png)

**Second attempt: structural impossibility.** Some tasks genuinely have no right answer. We created one deliberately (see [Experiment 3](experiment_3/) for the conflicting rule setup). Context X says "when you see X=0, output Z=0." Context Y says "when you see Y=0, output Z=1." Now ask the model to handle a situation where both X=0 and Y=0 at the same time. The model must violate at least one rule—it's mathematically impossible to satisfy both. The model learned each context perfectly in isolation—$100\%$ accuracy when seeing only X or only Y. On joint queries requiring both contexts simultaneously, hallucination hit $76\%$ with $88\%$ confidence. This isn't a training failure. No amount of training can resolve structural contradictions. Training can only choose which rule to break, not eliminate the contradiction.

**Third attempt: just allow "I don't know."** Same model, same task, different output format (see [Experiment 7](experiment_7/) for the abstention protocol). We let the model say "I don't know" as a valid response. Hallucination dropped from $76\%$ to $1\%$. That's a $75$-point improvement without changing anything about training. The model produced $495$ abstentions and only $5$ fabrications out of $500$ trials. When we forced it to pick a specific answer instead, it produced $380$ fabrications and $120$ correct answers. Training approaches like uncertainty calibration can help, but can't fundamentally teach a forced-choice architecture to abstain, just as training can't teach a calculator to express uncertainty about division by zero.

These three attempts reveal the same pattern. Training can change which inputs hallucinate but can't change the underlying architectural constraint. Current approaches—RLHF, Constitutional AI, scaling to bigger models—all operate within this constraint. They're optimizing the wrong variable.

## Architecture Changes That Seem to Work

Some current approaches catch hallucinations after they happen or teach models to hedge their language. But others do address the core issue—architectures that force output when "I don't know" would be appropriate. 

Our theory would explain why these architectural approaches work: they create new uncertainty capacity rather than just redistributing existing probability mass.

The $75$-point gap ($76\%$ to $1\%$) demonstrates this distinction. Solutions that create new uncertainty capacity need to do three things: give the system a dedicated way to express uncertainty, make that mechanism work on new cases (not just memorized ones), and provide enough capacity to handle the task's inherent ambiguity.

**Retrieval with explicit "not found" states.** When a system searches for information, it can return two fundamentally different outcomes: "here's the answer" or "no matching documents found." That second state is architectural—it's not the system fabricating an answer from weak signals. It's the system reporting that retrieval failed. This gives the system a dedicated uncertainty channel. We predict this reduces hallucination by $50\%$--$70\%$ based on the capacity it provides, which matches what we see in production RAG systems.

**Tool use with delegation.** Instead of the model generating an answer and checking it, delegate to external tools. Ask for a calculation? Send it to a calculator. The calculator returns either a number (success) or an error (failure). Both outcomes have dedicated representation. The model doesn't need to fabricate when tools indicate uncertainty—it can report the tool's result directly. This provides even more capacity than retrieval, potentially reducing hallucination to near-zero.

**Semantic uncertainty.** Before generating, measure how much the model disagrees with itself. Generate multiple possible answers, cluster them by meaning, and measure uncertainty across clusters. High disagreement signals the query is underspecified or contradictory. That signal routes to abstention before the model commits to a specific answer. This operates at meaning level rather than token level, making it more robust than simple confidence scores. Research exploring semantic entropy measures shows promise for detecting hallucinations beyond simple confidence thresholds.

**Structured output with null values.** Instead of adding "I don't know" as another token competing with all other tokens (which doesn't work—our detector got $3.9\%$ accuracy), build it into the output type system. Database fields can be NULL. Pydantic schemas can have Optional fields. Programming languages have None or null values. The uncertainty mechanism doesn't trade off with answer quality because it lives in a different representational space.

Structured schemas work in controlled experiments—they enforce clean abstention with dedicated tokens. But production scaling faces challenges. Models must generate conversational uncertainty phrases that fit dialogue naturally, not single classification tokens. Multi-token expressions compete with all autoregressive continuations. Hybrid systems like RAG provide partial capacity. But how retriever, planner, and verifier components combine remains unknown. Does capacity add linearly across modules? Does it bottleneck at the weakest link? Or interact nonlinearly? Chain-of-thought degradation suggests insufficient per-step capacity compounds. Direct multi-module tests haven't confirmed this yet.

The question becomes here: does the intervention give the system a new way to express uncertainty, or does it just redistribute existing probability mass? This theory would predict if it creates new capacity, hallucination drops proportionally. If it just reshuffles probability, hallucination persists.

## Production LLMs Face the Same Constraints

Production language models seem to show the same bottleneck we found in our experiments. We tested Llama 3.1:8B and found it must produce responses for every input, even when "I don't know" would be more appropriate.

Take our weekday task at the simplest level ([Experiment 1](experiment_1/)): unique correct answer exists, model learned the pattern perfectly in context. When given "today is Monday," the model correctly responded "Tuesday" with $100\%$ accuracy. But remove that context and ask "What day comes after today?" The model fabricated $45\%$ of the time (see [Experiment 1](experiment_1/) for replication). That's baseline partiality pressure—the model understands the task but can't express uncertainty when the query underspecifies.

Now add structural impossibility. Questions with genuinely undefined answers, contradictory contexts, causally underdetermined explanations. Our experiments show these tasks have structural floors—minimum unavoidable hallucination rates (measured in [Experiment 3](experiment_3/)). For mild contradiction, that floor sits around $29\%$. For moderate contradiction, around $40\%$. For strong contradiction, around $53\%$. But we observed $64\%$, $72\%$, and $75\%$ respectively. Those rates all exceed the structural minimum by $22$--$35$ percentage points (see [Experiment 4](experiment_4/) for invariance across training distributions). That excess? Architectural forcing. We predict that the observed $60\%$--$80\%$ rates in LLM benchmarks stem from architectures forcing output, not from the tasks themselves being impossible.

Multi-hop reasoning can compound these effects. Each reasoning step may accumulate uncertainty. In a $5$-step reasoning chain where each step has a $29\%$ structural floor, error propagation could lead to significant degradation. This suggests why long chains of thought may degrade—not because contradiction compounds without bound, but because uncertainty accumulates across steps.

Scale doesn't fix this. We tested across $5$ random seeds, $17$ training compositions, $9$ dataset balances, and $2,500+$ total trials (see [Experiment 5](experiment_5/) and [Experiment 6](experiment_6/) for comprehensive testing). The constraint held everywhere. The $75$-point reduction from adding abstention support dwarfs any improvement from scale alone. Current LLMs have high hallucination rates not because they're undertrained. They lack architectural capacity for uncertainty. Adding more parameters increases the ability to represent patterns but doesn't create channels for expressing "I don't know."

## Open Questions

Our experiments showed three pressures we've identified and quantified—$45\%$ partiality, $\approx 11\%$ points structural contradiction, $\approx 75\%$ points architectural forcing. Those proportions held across the synthetic tasks we tested. Do they hold across different task families? Could other mechanisms contribute that we haven't isolated yet? Building diagnostics that automatically attribute observed hallucination to each source remains an open challenge.

Uncertainty capacity seems to distribute across system components—retrievers, planners, verifiers. But we don't know how it combines. Does it add linearly across modules? Bottleneck at the weakest link? Interact in more complex ways? Long chain-of-thought degradation suggests insufficient capacity per step compounds, but we need direct multi-module tests to confirm this.

We also don't know whether uncertainty capacity is a system property (consistent across all tasks) or task-dependent (varies by task family). Standard softmax showed near-zero capacity across everything we tested, suggesting system-level constraint. But supervision density effects suggest task dependence. Testing across wildly different semantics—factual QA, causal reasoning, creative generation—would resolve this.

Task structure varies across domains too. We expect high contradiction for factual QA with temporal ambiguity, moderate contradiction for reasoning tasks with multiple valid interpretations, low contradiction for creative tasks where most outputs are acceptable. Measuring structure for standard benchmarks would identify where structural impossibility contributes versus where partiality dominates.

Natural language queries blur the line between partiality and contradiction. "What will Apple's revenue be in Q4 2026?" is clearly partial—the future is undefined. But "Why did this user churn?"—is that undefined (no single cause), multi-valued (many valid explanations), or structurally contradictory (incompatible attribution frameworks)? Building classifiers to route queries appropriately remains open.

Finally, we need to identify optimal architectures. What designs provide enough uncertainty capacity while maintaining computational efficiency? Mixture-of-experts with explicit uncertainty routing? Probabilistic programming embeddings? Structured output spaces with native null support? What are the tradeoffs between uncertainty capacity, error rates, and compute?

Current scope has limitations too. Our decomposition holds for synthetic weekday tasks. But does it shift for factual QA (high temporal contradiction), causal reasoning (moderate multiple interpretations), or creative generation (low contradiction)? Direct benchmark validation remains needed. The theory predicts production 20-50% rates stem from partial capacity in hybrid systems. But does the 45% partiality + 11% contradiction + 75% architectural breakdown hold on TruthfulQA, MMLU, or other standard evals? Measuring contradiction K on real tasks would confirm whether task structure explains observed benchmark differences.


The theory makes falsifiable predictions. It would be in trouble if hallucination reduction didn't correlate with abstention freedom (contradicted: we observe $75$-point reduction in [Experiment 7](experiment_7/)), if the conservation law was violated (not observed: holds across $2,500+$ trials with zero violations), if adding independent abstention channels produced negative returns (untested), or if systems with radically different architectures exhibited identical hallucination-abstention tradeoffs when uncertainty capacity differs (untested). These remain testable predictions.

## A Fundamental Constraint

Hallucination can arise from an architectural mismatch. Neural networks implement total functions—they must answer everywhere. Real-world tasks are often partial (undefined inputs exist) or contradictory (incompatible contexts). Our experiments show how these constraints manifest and their relative contributions across 2,500+ controlled trials (see [Experiment 7](experiment_7/) for the comprehensive decomposition).

Architectural commitment is a measurable pressure. When forced to produce outputs without abstention support, models fabricate on $45\%$--$76\%$ of inputs even when the task is logically coherent. We demonstrated this with the weekday task: unique correct answer exists, but $45\%$ hallucination when context is removed (see [Experiment 1](experiment_1/) for details). The same neural net that achieves $100\%$ accuracy with context fabricates nearly half the time without it. With native abstention support, this drops to $1\%$—a $75$-point improvement ($495$ abstentions and $5$ fabrications versus $380$ fabrications and $120$ correct answers; see [Experiment 7](experiment_7/) for the abstention protocol). This isn't a training problem or a scale problem. It's an architectural feature: softmax forces commitment.

![Contradiction vs hallucination analysis showing theoretical bounds vs observed rates across different task complexities](/figures/contradiction_hallucination_analysis.png)

![Combined alternative views showing the dramatic architectural effect: 1% hallucination with abstention vs 76% when forced to choose](/figures/combined_alternative_views.png)

Structural contradiction provides inevitability. When tasks contain incompatible requirements, frame-independent architectures forced to commit cannot avoid hallucination. We tested this with conflicting marginal constraints (see [Experiment 3](experiment_3/) for setup). Mild contradiction predicts $\geq 18\%$ minimum, we observed $76\%$. Moderate contradiction predicts $\geq 29\%$, we observed $64\%$. Strong contradiction predicts $\geq 53\%$, we observed $75\%$. All observations exceed bounds. But structural impossibility is a certificate of inevitability, not a predictor of magnitude. Increasing structural impossibility $2.2\times$ adds only $11\%$ percentage points ($64\%$ to $75\%$). The architectural term is measurable and contributes significantly (derived from [Experiment 7](experiment_7/) pressure decomposition). We also showed task structure remains perfectly constant (exact same measure across all training distributions) while hallucination varies from $58.6\%$ to $100\%$, confirming that task structure is invariant and behavior depends on architectural constraints (see [Experiment 4](experiment_4/) for invariance testing).

The conservation law binds precisely across all experiments with zero violations. Contradiction cost must appear somewhere. When uncertainty capacity is near-zero, full cost shows up as error. Standard architectures have near-zero capacity structurally—they cannot natively express "I don't know." Our detector head achieved essentially zero capacity ($100\%$ train, $3.9\%$ test accuracy) because undefined inputs share no learnable features (see [Experiment 2](experiment_2/) for detector architecture). When capacity exceeds contradiction, error can approach zero. The $76\%$ to $1\%$ reduction demonstrates this directly: adding abstention support increases capacity from $0$ to approximately $0.69$ bits, enabling the dramatic reduction. Current approaches—RLHF, Constitutional AI—don't increase capacity. They redistribute fabrication across inputs or teach hedging language, but the architectural pressure remains. Post-hoc filtering catches symptoms after generation, leaving the underlying constraint unaddressed.

Scale has limited leverage. Experiments across $5$ random seeds, $17$ training compositions, and $9$ dataset balances show task structure constant while hallucination varies by $40+$ points (see [Experiment 5](experiment_5/) and [Experiment 6](experiment_6/) for comprehensive testing). Scale can modulate manifestation through learned priors (the sigmoid relationship shows how training composition shifts behavior), but cannot eliminate partiality pressure ($45\%$ baseline persists from [Experiment 1](experiment_1/)) or architectural forcing ($75$-point gap requires structural change from [Experiment 7](experiment_7/)). Training can shift which inputs hallucinate and help with uncertainty calibration, but fundamentally limited in increasing uncertainty capacity or reducing task structure within standard architectures. Architectural change is required to address the primary limitation. We validated this with the witness-error tradeoff (see [Experiment 2](experiment_2/) for details): adding more training data to the detector head achieved $100\%$ training accuracy but still only $3.9\%$ test accuracy—memorization without generalization because architectural capacity for witness information was insufficient.

![Monotonicity violation analysis showing hallucination trends across 5 random seeds with small local violations against strong directional pressure](/figures/monotonicity_violation_analysis.png)

Solutions must target uncertainty capacity. RAG with explicit "not found" states, tool use with delegation, semantic uncertainty quantification—these work by increasing witness capacity. The theory predicts and experiments confirm capacity is the primary lever. Our ablation protocol shows: standard architecture (near-zero capacity) produces $76\%$ hallucination on a task with moderate contradiction. Abstention support ($0.69$ bits capacity) produces $1\%$ hallucination on the same task (see [Experiment 7](experiment_7/) for the capacity measurements). A system achieving $1$ bit capacity could cut hallucination to near-zero for tasks with contradiction below $1$ bit, covering the majority of real-world queries.

The path forward: measure task structure to identify high-contradiction domains, design architectures achieving positive capacity with generalization (dedicated witness mechanisms that don't compete with primary task), validate that the conservation law explains observed rates across task types (ablation studies showing hallucination reduction correlates with inferred capacity). Contradiction theory provides the foundation. The architectural work begins now.


---

## References

Bridges, C. (2025). *A Mathematical Theory of Contradiction* (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17203336