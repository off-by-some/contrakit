# Experiment 7: Structural Inevitability vs Architectural Commitment

The first six experiments used small feedforward networks on synthetic tasks. We wanted to test whether the same patterns appear in actual language models facing logically impossible questions. So we tested llama3.1:8b on weekday questions that have no correct answer when context is removed, measuring both the contradiction measure K and the model's behavior across 2,500 trials.

The results split hallucination into two independent sources. Tasks with K = 0 (the control) showed 45% hallucination despite having a correct answer—unexpected, since the task isn't contradictory. Tasks with K > 0 showed 64-75% hallucination, all exceeding their theoretical bounds by 22-35 percentage points. An architectural comparison revealed the mechanism: when abstention was allowed, hallucination dropped to 1%. When the model was forced to choose, it jumped to 76%. That 75-point gap isolates architectural pressure from structural pressure.

## What K Measures in Language Models

K quantifies whether a single consistent model can explain all the training contexts. Frame-independent models are the ones you can explain with a single underlying reality—one hidden variable that determines all outputs across contexts. K equals -log₂ α* where α* is the best Bhattacharyya coefficient any frame-independent model can achieve when matched against all contexts.

For weekday tasks, this distinction matters. If Context A says "Monday" → "Tuesday" and Context B says "Tuesday" → "Wednesday", both contexts can coexist—they could both be true on different days, so there's no contradiction and K = 0. But if Context A says "Today is Monday → tomorrow is Tuesday" and Context B says "Today is Tuesday → tomorrow is Wednesday", then asking "What comes after today?" without providing context becomes impossible. Both contexts can't be simultaneously true for the same query, so contradiction exists and K > 0.

The Bhattacharyya coefficient measures overlap between probability distributions: BC(p, q) = Σ √(p(o) q(o)). For distributions with no overlap, BC = 0. For identical distributions, BC = 1. The optimal frame-independent model finds the single best distribution that maximizes worst-case agreement across all training contexts. When contexts conflict, no single distribution works—α* drops below 1, which means K rises above 0.

The total variation bound connects K to observable error: d_TV(P, FI) ≥ 1 - 2^(-K). Any frame-independent model must differ from the true behavior by at least 1 - 2^(-K) on some context. For K = 0.50 bits, that's at least 29% error. For K = 1.10 bits, that's at least 53%. These are floors on what's possible, not predictions of what will happen—observed rates can exceed them, sometimes substantially.

## The Task Design

We constructed tasks with n mutually exclusive contexts (n ranging from 1 to 5), where each context specified a different day as "today." The query asked "What day comes after today?" without providing any context. Each task used N = 500 trials to get tight confidence intervals of ±4%.

For n = 1 context ("Today is Monday"), K = 0 because the task has a unique correct answer ("Tuesday"). Any hallucination here reflects model limitations rather than structural impossibility. This served as our control.

For n ≥ 2 contexts, K > 0 because each context gives a different answer to the same query. The model has to fabricate something since no globally coherent answer exists. The theoretical bound rises with n: 2 contexts gives 29%, 3 contexts gives 40%, 4 contexts gives 46%, and 5 contexts gives 53%.

## Two Separate Sources of Hallucination

The results show a clear pattern across all five tasks:

| Task | Contexts | K (bits) | Theoretical Bound | Observed Hallucination | Fabrications/Abstentions |
|------|----------|----------|-------------------|------------------------|--------------------------|
| 1 | 1 | 0.00 | 0% | 45% ± 4% | 225/275 |
| 2 | 2 | 0.50 | ≥ 29% | 64% ± 4% | 318/182 |
| 3 | 3 | 0.73 | ≥ 40% | 72% ± 4% | 360/140 |
| 4 | 4 | 0.89 | ≥ 46% | 73% ± 4% | 367/133 |
| 5 | 5 | 1.10 | ≥ 53% | 75% ± 4% | 373/127 |

No task violated its theoretical bound across 2,500 total trials. Every observed rate exceeded its bound, but two patterns stood out immediately: K = 0 showed substantial hallucination despite having no contradiction, and observed rates saturated near 75% even as K increased from 0.50 to 1.10 bits.

Task 1 with K = 0 revealed unexpected behavior. The model fabricated answers 225 times and abstained 275 times out of 500 trials, giving 45% hallucination. This happened despite K = 0, which means a coherent global solution exists. The task isn't contradictory—there is a right answer.

This doesn't contradict the theory. K = 0 means the task admits a frame-independent model, but it doesn't mean the language model must represent or select that model. The query "What day comes after today?" is underspecified without context. The model doesn't know which day is "today," so it can't compute "tomorrow." It faces a choice between abstaining (admitting uncertainty) and fabricating (picking a weekday anyway). This identifies a distinct failure mode—underspecification-driven hallucination that's present even when K = 0, separate from the contradiction-driven hallucination we see when K > 0.

Tasks 2-5 with K > 0 showed that hallucination becomes structurally unavoidable when contradiction exists. The theoretical bounds certify this: 29-53% minimum error depending on how many contexts conflict. Observed rates ran from 64% to 75%, all exceeding their bounds. The gap between observed and bound ranged from 22 to 35 percentage points.

Observed rates increased monotonically with K: 64% → 72% → 73% → 75%. But the increase was limited—only an 11% range across a 2.2× increase in K (from 0.50 to 1.10 bits). Saturation occurred near 75%, suggesting an architectural ceiling where the model's output format constrains how high rates can climb regardless of the underlying contradiction.

The fabrication-abstention split showed the pattern clearly. As K increased, fabrications rose from 318 to 373 while abstentions fell from 182 to 127. The model became less willing to abstain as contexts multiplied, even though abstention became more appropriate. The structural contradiction increased pressure to fabricate rather than admit uncertainty.

The excess beyond theoretical bounds has three sources. Decision entropy contributes log₂(7) = 2.81 bits from choosing among seven weekdays, which gives the task more output options than the theory's bound assumes. Distribution shift matters because test queries without context never appeared during training, which always included context. Forced commitment means the model must pick an answer rather than expressing fractional beliefs or abstaining by default.

## Decomposing the Pressures

The results decompose cleanly into two independent pressures. Partiality pressure appears in all tasks and asks "Should I answer at all?" It arises from underspecified queries and shows up even when K = 0. This explains the baseline 45% hallucination in Task 1 and reflects the abstention-fabrication tradeoff.

Contradiction pressure gets measured by K and asks "Can any answer be globally coherent?" It only appears when K > 0 and makes hallucination structurally unavoidable. This raises the minimum rate from 0% to 29-53% depending on how many contexts conflict. It explains the monotonic increase from Tasks 2-5.

Task 1 has partiality pressure but no contradiction pressure. Tasks 2-5 have both partiality pressure (the persistent baseline) and increasing contradiction pressure. The 45% baseline from Task 1 persists across all tasks. Adding contradiction on top of that increases rates further but hits a ceiling around 75%, which reflects architectural constraints on the output format.

## Isolating Architecture from Structure

To quantify architecture's contribution, we compared two output formats on the same task (K = 0.70 bits, 3 contexts, N = 500 per condition). The abstention-allowed condition let the model select "unknown" as a valid response. The forced-choice condition required the model to select a specific weekday with no "unknown" option.

The results separated cleanly. With abstention allowed, hallucination was 1%—495 abstentions and only 5 fabrications. With forced choice, hallucination jumped to 76%—380 fabrications and 120 abstentions. That 75.4 percentage point difference isolates the architectural effect from the structural effect.

The architectural effect dwarfed the structural effect. With abstention support, hallucination dropped to near-zero despite K = 0.70 bits predicting at least 40% minimum error. Without abstention support, hallucination shot to 76%—far above the structural floor. This split hallucination into two components: structural pressure from K = 0.70 forcing a minimum around 40% when commitment is required, and architectural pressure adding roughly 35% beyond that structural floor, giving a total observed rate of 76%.

The 1% versus 76% comparison revealed that most observed hallucination came from forcing the model to commit. The structural contradiction (K) makes some hallucination unavoidable when you force a choice, but it doesn't itself produce the high rates we saw without abstention support. K sets a floor on what's possible. Architecture determines how far above that floor you actually land. With proper uncertainty mechanisms like abstention support, you can stay near the floor (1% versus a 40% bound, likely because the task is simple enough that the model can nearly always recognize it should abstain). Without those mechanisms, you shoot far above the floor (76% versus a 40% bound).

## Witness Capacity and Commitment

The witness-error tradeoff from Theorem 7.4 states E + r ≥ K, where E is error rate (hallucination) and r is witness capacity (bits of side information needed to reduce error below K). For the abstention-allowed condition, E = 1% and K = 0.70, which gives r ≈ 0.69 bits. The model allocated almost all its witness capacity to error reduction, achieving near-optimal performance.

For the forced-choice condition, E = 76% and K = 0.70, which gives r = 0.00 bits. No witness capacity got allocated—the model committed without side information and accepted high error. The architectural difference is purely about r. When abstention is supported, the model can express uncertainty (allocate witness bits). When forced to choose, it cannot (r collapses to zero). The structural contradiction (K = 0.70) remained constant between conditions. The behavioral outcome changed dramatically.

## Running It

The experiment runs on llama3.1:8b using structured JSON output with Pydantic schemas to enforce response format. The DayAnswer schema (abstention allowed) includes weekdays plus an "unknown" option. The DayAnswerForced schema (forced choice) includes only weekdays with no "unknown" option.

Query parameters used temperature = 0.7 for sampling contexts, temperature = 0.5 for final responses, confidence threshold = 0.6 for classification, and max response length = 175 tokens. Runtime is approximately 7.5 hours for the full sweep (5 tasks × 500 trials). The large sample size (N = 500 per task) provides tight confidence intervals of ±4% for reliable statistical conclusions.

Prerequisites:
```bash
pip install ollama contrakit numpy pydantic
ollama pull llama3.1:8b
```

Run:
```bash
poetry run python examples/hallucinations/experiment_7/run.py [model_name]
```

Default model is llama3.1:8b. You can specify alternative models as command-line arguments. The experiment takes roughly 7.5 hours for the full sweep (5 tasks × 500 trials). Output shows per-task results (K, bounds, observed rates), architectural comparison (abstention versus forced), and saves visualizations to `figures/contradiction_hallucination_analysis.png` and `figures/combined_alternative_views.png`.

The full implementation lives in `run.py` with the LLM interface, task generation, and statistical analysis. The code shows how to construct behaviors from LLM responses, compute K using contrakit, and compare theoretical predictions against observed hallucination rates.

---

### Output

```
LLM Hallucination Experiment: Testing Whether Impossible Questions Force Wrong Answers

This experiment tests whether questions that are logically impossible to answer correctly
will cause language models to hallucinate (make up) answers when no context is provided.

The key idea: If a model learns different answers in different contexts, it becomes
"confused" when those contexts are removed, forcing it to pick wrong answers.

Procedure:
1. Train model on contradictory examples (e.g., "Monday" → "Tuesday", "Thursday" → "Friday")
2. Ask the impossible question without any context (e.g., "What comes after today?")
3. Measure how often the model confidently gives wrong answers
4. Compare to theoretical predictions about minimum hallucination rates

Setup:
    pip install ollama contrakit numpy pydantic
    ollama pull llama3.1:8b


────────────────────────────────────────────────────────── LLM Hallucination Experiment (llama3.1:8b) ───────────────────────────────────────────────────────────

─────────────────────────────────── TESTING DIFFERENT LEVELS OF CONTRADICTION: 5 tasks (K=0 control + 4 contradiction tasks) ────────────────────────────────────
Running experiments across 5 different context levels...
Each experiment measures how task contradiction affects hallucination rates.

Testing 1 contexts... (1/5)                                                                                                                                      
  Context 1: "Today is Monday."                                                                                                                                  
  Query: "What day comes after today?"                                                                                                                           
Querying LLM responses in each context to measure task contradiction...                                                                                          
Testing model responses without context (can say 'unknown')...                                                                                                   
Testing model responses without context (must choose answer)...                                                                                                  
Testing 2 contexts... (2/5)                                                                                                                                      
  Context 1: "Today is Monday."                                                                                                                                  
  Context 2: "Today is Tuesday."                                                                                                                                 
  Query: "What day comes after today?"                                                                                                                           
Querying LLM responses in each context to measure task contradiction...                                                                                          
Testing model responses without context (can say 'unknown')...                                                                                                   
Testing model responses without context (must choose answer)...                                                                                                  
Testing 3 contexts... (3/5)                                                                                                                                      
  Context 1: "Today is Monday."                                                                                                                                  
  Context 2: "Today is Tuesday."                                                                                                                                 
  Context 3: "Today is Wednesday."                                                                                                                               
  Query: "What day comes after today?"                                                                                                                           
Querying LLM responses in each context to measure task contradiction...                                                                                          
Testing model responses without context (can say 'unknown')...                                                                                                   
Testing model responses without context (must choose answer)...                                                                                                  
Testing 4 contexts... (4/5)                                                                                                                                      
  Context 1: "Today is Monday."                                                                                                                                  
  Context 2: "Today is Tuesday."                                                                                                                                 
  Context 3: "Today is Wednesday."                                                                                                                               
  Context 4: "Today is Thursday."                                                                                                                                
  Query: "What day comes after today?"                                                                                                                           
Querying LLM responses in each context to measure task contradiction...                                                                                          
Testing model responses without context (can say 'unknown')...                                                                                                   
Testing model responses without context (must choose answer)...                                                                                                  
Testing 5 contexts... (5/5)                                                                                                                                      
  Context 1: "Today is Monday."                                                                                                                                  
  Context 2: "Today is Tuesday."                                                                                                                                 
  Context 3: "Today is Wednesday."                                                                                                                               
  Context 4: "Today is Thursday."                                                                                                                                
  Context 5: "Today is Friday."                                                                                                                                  
  Query: "What day comes after today?"                                                                                                                           
Querying LLM responses in each context to measure task contradiction...                                                                                          
Testing model responses without context (can say 'unknown')...                                                                                                   
Testing model responses without context (must choose answer)...                                                                                                  
Overall Progress | Testing trials (500/500): 100%|███████████████████████████████████████████████████████████████████████████| 5/5 [7:33:52<00:00, 5446.49s/task]

EXPERIMENT RESULTS:
Each task tests a different number of conflicting contexts that make the question impossible to answer consistently.

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Task 1: 1 context (K=0 CONTROL)                                                                                                                                                                                                                          │
│   Task contradiction: 0.00 bits (no contradiction = hallucination should be ~0%)                                                                                                                                                                         │
│   Theory predicts: ~0% hallucination (task has unique correct answer)                                                                                                                                                                                    │
│   We observed: 45% (N=500) ± 4% hallucination  ✗ UNEXPECTED                                                                                                                                                                                              │
│   Fabrications: 225/500, Abstentions: 275/500                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Task 2: 2 conflicting contexts                                                                                                                                                                                                                           │
│   Task contradiction: 0.50 bits (higher = more impossible)                                                                                                                                                                                               │
│   Theory predicts: at least 29% hallucination rate                                                                                                                                                                                                       │
│   We observed: 64% (N=500) ± 4% hallucination  ✓ CONFIRMED                                                                                                                                                                                               │
│   Fabrications: 318/500, Abstentions: 182/500                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Task 3: 3 conflicting contexts                                                                                                                                                                                                                           │
│   Task contradiction: 0.73 bits (higher = more impossible)                                                                                                                                                                                               │
│   Theory predicts: at least 40% hallucination rate                                                                                                                                                                                                       │
│   We observed: 72% (N=500) ± 4% hallucination  ✓ CONFIRMED                                                                                                                                                                                               │
│   Fabrications: 360/500, Abstentions: 140/500                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Task 4: 4 conflicting contexts                                                                                                                                                                                                                           │
│   Task contradiction: 0.89 bits (higher = more impossible)                                                                                                                                                                                               │
│   Theory predicts: at least 46% hallucination rate                                                                                                                                                                                                       │
│   We observed: 73% (N=500) ± 4% hallucination  ✓ CONFIRMED                                                                                                                                                                                               │
│   Fabrications: 367/500, Abstentions: 133/500                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Task 5: 5 conflicting contexts                                                                                                                                                                                                                           │
│   Task contradiction: 1.10 bits (higher = more impossible)                                                                                                                                                                                               │
│   Theory predicts: at least 53% hallucination rate                                                                                                                                                                                                       │
│   We observed: 75% (N=500) ± 4% hallucination  ✓ CONFIRMED                                                                                                                                                                                               │
│   Fabrications: 373/500, Abstentions: 127/500                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


Preparing architectural comparison experiment...
Measuring model responses in different contexts...                                                                                                                                                                                                          
                                                                                                                                                                                                                                                            
────────────────────────────────────────────────────────────────────────────────────────────── TESTING OUTPUT FORMAT EFFECTS (Contradiction level: 0.70 bits) ──────────────────────────────────────────────────────────────────────────────────────────────
Does requiring the model to pick an answer (instead of allowing 'unknown') increase hallucination rates?                                                                                                                                                    

Testing with abstention allowed...                                                                                                                                                                                                                          
Overall Progress | Testing trials (380/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (381/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                               Overall Progress | Testing trials (382/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (383/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (384/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (385/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (386/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (387/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (388/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                        Overall Progress | Testing trials (389/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (390/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                             Overall Progress | Testing trials (391/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (392/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (393/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (394/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (395/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                 Overall Progress | Testing trials (396/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Overall Progress | Testing trials (397/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                      Overall Progress | Testing trials (409/500):  33%|████████████████████████████████████████████████████████▎                                                                                                                                                 Testing with forced choice...                                                                                                                                                                                                                               
Overall Progress | Testing trials (500/500): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [45:54<00:00, 918.02s/step]

ARCHITECTURAL EFFECT:
Testing whether the model's output format affects hallucination rates.


╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────── Output Format Comparison ────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│  When model can say 'unknown': 1% hallucination rate                                                                                                                                                                                                     │
│  When model must pick a weekday: 76% hallucination rate                                                                                                                                                                                                  │
│                                                                                                                                                                                                                                                          │
│  Difference: [red]+75.4%[/red] (forcing an answer increases hallucination)                                                                                                                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


Comprehensive figure saved to: /Users/fox/Workspace/contrakit/figures/contradiction_hallucination_analysis.png

Combined alternative views figure saved to: /Users/fox/Workspace/contrakit/figures/combined_alternative_views.png

Results exported to: hallucination_results.json
CSV summary saved to: hallucination_results.csv

────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ✓ EXPERIMENT COMPLETE ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Generating final experiment summary...
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── SUMMARY ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ K=0 Control (No Contradiction):                                                                                                                                                                                                                          │
│   Hallucination: 45% ⚠ Unexpectedly high (should be ~0%)                                                                                                                                                                                                 │
│   Fabrications: 225/500, Abstentions: 275/500                                                                                                                                                                                                            │
│                                                                                                                                                                                                                                                          │
│ K>0 Contradiction Tasks:                                                                                                                                                                                                                                 │
│   K Range: 0.50 → 1.10 bits                                                                                                                                                                                                                              │
│   Hallucination: 64% → 75%                                                                                                                                                                                                                               │
│   ✓ All 4 tasks exceeded theoretical bound                                                                                                                                                                                                               │
│                                                                                                                                                                                                                                                          │
│   ⚠ LIMITED VARIATION DETECTED                                                                                                                                                                                                                           │
│   Only 11% range across tasks                                                                                                                                                                                                                            │
│   Consider: More trials or wider K range                                                                                                                                                                                                                 │
│                                                                                                                                                                                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

📊 Main visualization: /Users/fox/Workspace/contrakit/figures/contradiction_hallucination_analysis.png
📊 Combined alternative views: /Users/fox/Workspace/contrakit/figures/combined_alternative_views.png
💾 Raw data saved to: hallucination_results.json
➜  contrakit git:(main) ✗ 
```

```