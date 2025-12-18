# Experiment 7: Structural Inevitability vs Architectural Commitment

Does task-level contradiction ($K$) determine whether hallucination is unavoidable? And can observed hallucination be decomposed into structural inevitability plus architectural commitment? We tested llama3.1:8b on logically impossible weekday questions, measuring both $K$ and behavior.

The results separate two pressures cleanly. $K = 0$ (control) shows 45% hallucination—unexpected, since the task has a correct answer. $K > 0$ tasks show 64-75% hallucination, all exceeding theoretical bounds (29-53%). Architectural comparison reveals the mechanism: when abstention is allowed, hallucination drops to 1%. When forced to choose, it jumps to 76%. The 75-point gap isolates architectural pressure from structural pressure.

## What $K$ Measures in LLMs

$K$ quantifies whether a single consistent model can explain all training contexts. Frame-independent (FI) models are those explainable by one underlying reality—one hidden variable determining all outputs. $K = -\log_2 \alpha^*$ where $\alpha^*$ is the best Bhattacharyya coefficient any FI model achieves across contexts.

For weekday tasks: if Context A says "Monday" $\rightarrow$ "Tuesday" and Context B says "Tuesday" $\rightarrow$ "Wednesday", these contexts can coexist (both could be true on different days). No contradiction. $K = 0$. But if Context A says "Today is Monday $\rightarrow$ tomorrow is Tuesday" and Context B says "Today is Tuesday $\rightarrow$ tomorrow is Wednesday", then asking "What comes after today?" without context is impossible—both contexts can't be simultaneously true. Contradiction exists. $K > 0$.

The Bhattacharyya coefficient measures overlap:

$$\text{BC}(p, q) = \sum_o \sqrt{p(o) q(o)}$$

For distributions with no overlap, $\text{BC} = 0$. For identical distributions, $\text{BC} = 1$. The optimal FI model finds the best single distribution that maximizes worst-case agreement with all training contexts. When contexts conflict, no single distribution works—$\alpha^* < 1$, $K > 0$.

The total variation bound connects $K$ to observable error:

$$d_{\text{TV}}(P, \text{FI}) \geq 1 - 2^{-K}$$

Any FI model must differ from true behavior by at least $1 - 2^{-K}$ on some context. For $K = 0.50$, that's $\geq 29\%$. For $K = 1.10$, that's $\geq 53\%$. These are floors, not ceilings—observed rates can exceed them.

## The Experiment

We constructed tasks with $n$ mutually exclusive contexts ($n = 1$ to 5), each specifying a different day as "today." The query asks "What day comes after today?" without providing context. Each task uses $N = 500$ trials for tight confidence intervals ($\pm 4\%$).

For $n = 1$ context ("Today is Monday"), $K = 0$. The task has a unique correct answer ("Tuesday"). Any hallucination reflects model limitations, not structural impossibility.

For $n \geq 2$ contexts, $K > 0$. Each context gives a different answer. The model must fabricate since no globally coherent answer exists. The theoretical bound rises with $n$: 2 contexts $\rightarrow$ 29%, 3 contexts $\rightarrow$ 40%, 4 contexts $\rightarrow$ 46%, 5 contexts $\rightarrow$ 53%.

## Results: Two Sources of Hallucination

Summary across all tasks:

| Task | Contexts | $K$ (bits) | Theoretical Bound | Observed Hallucination | Fabrications/Abstentions |
| ---- | -------- | -------- | ----------------- | ---------------------- | ------------------------ |
| 1    | 1        | 0.00     | 0%                | 45% $\pm$ 4%           | 225/275 |
| 2    | 2        | 0.50     | $\geq$ 29%        | 64% $\pm$ 4%           | 318/182 |
| 3    | 3        | 0.73     | $\geq$ 40%        | 72% $\pm$ 4%           | 360/140 |
| 4    | 4        | 0.89     | $\geq$ 46%        | 73% $\pm$ 4%           | 367/133 |
| 5    | 5        | 1.10     | $\geq$ 53%        | 75% $\pm$ 4%           | 373/127 |

No task violates the theoretical bound across 2,500 total trials. Observed rates always exceed bounds. But two patterns stand out: (1) $K = 0$ shows substantial hallucination despite no contradiction, and (2) observed rates saturate near 75% despite $K$ increasing from 0.50 to 1.10.

### Task 1 ($K = 0$): Partiality Pressure

The control case reveals unexpected behavior. Fabrications: 225. Abstentions: 275. Despite $K = 0$—meaning a coherent global solution exists—the model hallucinates 45% of the time.

This doesn't contradict theory. $K = 0$ means the task admits a frame-independent model. It doesn't mean the LLM must represent or select that model. The query "What day comes after today?" is underspecified without context. The model doesn't know which day is "today," so it can't compute "tomorrow." It faces a choice: abstain (admit uncertainty) or fabricate (pick a weekday).

This identifies a distinct failure mode: underspecification-driven hallucination. Present even when $K = 0$. Separate from contradiction-driven hallucination. The model can answer correctly (there is a right answer), but doesn't know which answer to give (context is missing).

### Tasks 2-5 ($K > 0$): Contradiction Pressure

For $K > 0$, hallucination becomes structurally unavoidable. The theoretical bounds certify this: 29-53% minimum error. Observed rates: 64-75%. All exceed bounds. The gap (observed $-$ bound) ranges from 22 to 35 percentage points.

Observed rates increase monotonically with $K$: 64% $\rightarrow$ 72% $\rightarrow$ 73% $\rightarrow$ 75%. But the increase is limited—only 11% range across 2.2$\times$ increase in $K$ (0.50 $\rightarrow$ 1.10). Saturation occurs near 75%. This suggests an architectural ceiling: the model's output format constrains how high rates can climb, independent of underlying contradiction.

The fabrication/abstention split shows the pattern clearly. As $K$ increases, fabrications rise (318 $\rightarrow$ 373) while abstentions fall (182 $\rightarrow$ 127). The model becomes less willing to abstain as contexts multiply, even though abstention becomes more appropriate. The structural contradiction increases pressure to fabricate.

The excess beyond theoretical bounds has three sources:
1. **Decision entropy**: $\log_2(7) = 2.81$ bits for choosing among weekdays. The task offers more output options than the theory's bound assumes.
2. **Distribution shift**: Test queries (no context) were never seen during training (always had context).
3. **Forced commitment**: The model must pick an answer. It can't express fractional beliefs or abstain by default.

## Two Independent Pressures

The results decompose cleanly:

**Partiality pressure** (present in all tasks): "Should I answer at all?" Arises from underspecified queries. Present even when $K = 0$. Explains baseline 45% hallucination in Task 1. Reflects abstention-fabrication tradeoff.

**Contradiction pressure** (measured by $K$): "No answer can be globally coherent." Present only when $K > 0$. Makes hallucination structurally unavoidable. Raises minimum rate from 0% to 29-53%. Explains monotonic increase from Tasks 2-5.

Tasks decompose:
- **Task 1**: Partiality ✓, Contradiction ✗
- **Tasks 2-5**: Partiality ✓, Contradiction ✓ (increasing)

The 45% baseline (Task 1) persists across all tasks. Adding contradiction ($K > 0$) increases rates further but hits a ceiling around 75%. The ceiling reflects architectural constraints on output format.

## Architectural Decomposition

To quantify architectural contribution, we compared two output formats on the same task ($K = 0.70$ bits, 3 contexts, $N = 500$ per condition):

**Abstention allowed**: Model can select "unknown" as valid response
**Forced choice**: Model must select a specific weekday (no "unknown" option)

Results:

```
Abstention allowed:  1% hallucination (495 abstentions, 5 fabrications)
Forced choice:      76% hallucination (380 fabrications, 120 abstentions)

Difference: +75.4 percentage points
```

The architectural effect dwarfs the structural effect. With abstention support, hallucination drops to near-zero (1%) despite $K = 0.70$ bits predicting $\geq 40\%$ minimum. Without abstention support, hallucination jumps to 76%—far above the structural floor.

This isolates two components:
1. **Structural pressure** ($K = 0.70$): Forces minimum ~40% when commitment required
2. **Architectural pressure**: Adds ~35% beyond structural floor
3. **Total observed**: 76% $\approx$ 40% (structural) + 35% (architectural)

The 1% vs 76% comparison reveals that most observed hallucination comes from forcing the model to commit. The structural contradiction ($K$) makes some hallucination unavoidable when forced to choose, but doesn't itself produce the high rates observed without abstention support.

Think of it this way: $K$ sets a floor. Architecture determines how far above the floor you land. With proper uncertainty mechanisms (abstention support), you can stay near the floor (1% vs 40% bound, likely due to task simplicity). Without them, you shoot far above (76% vs 40% bound).

## Witness Capacity Interpretation

From Theorem 7.4, the witness-error tradeoff states: $E + r \geq K$. Here, $E$ is error rate (hallucination) and $r$ is witness capacity (bits of side information needed to reduce error below $K$).

For abstention allowed: $E = 1\%$, $K = 0.70$, so $r \approx 0.69$ bits. The model allocates almost all its witness capacity to error reduction, achieving near-optimal performance.

For forced choice: $E = 76\%$, $K = 0.70$, so $r = 0.00$ bits. No witness capacity allocated—the model commits without side information, accepting high error.

The architectural difference is purely about $r$. When abstention is supported, the model can express uncertainty (allocate witness bits). When forced to choose, it cannot ($r$ collapses to zero). The structural contradiction ($K = 0.70$) remains constant. The behavioral outcome changes dramatically.

## Implementation Details

The experiment uses structured JSON output with Pydantic schemas to enforce response format:

**DayAnswer** (abstention allowed): Weekdays + "unknown" option
**DayAnswerForced** (forced choice): Weekdays only, no "unknown"

Query parameters: temperature = 0.7 for sampling contexts, temperature = 0.5 for final responses, confidence threshold = 0.6 for classification, max response length = 175 tokens.

Runtime: approximately 7.5 hours for full sweep (5 tasks $\times$ 500 trials). The large sample size ($N = 500$) provides tight confidence intervals ($\pm 4\%$) for reliable statistical conclusions.

## Running It

Prerequisites:
```bash
pip install ollama contrakit numpy pydantic
ollama pull llama3.1:8b
```

Run:
```bash
poetry run python examples/hallucinations/experiment_7/run.py [model_name]
```

Default model is llama3.1:8b. Specify alternative models as command-line arguments. The experiment takes ~7.5 hours for full sweep (5 tasks $\times$ 500 trials). Output shows per-task results ($K$, bounds, observed rates), architectural comparison (abstention vs forced), and visualizations saved to `figures/contradiction_hallucination_analysis.png` and `figures/combined_alternative_views.png`.

Full implementation in `run.py` with LLM interface, task generation, and statistical analysis. The code demonstrates how to construct behaviors from LLM responses, compute $K$ using contrakit, and compare theoretical predictions against observed hallucination rates.

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