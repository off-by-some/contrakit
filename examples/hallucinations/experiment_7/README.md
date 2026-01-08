# Experiment 7: Structural Inevitability vs Architectural Commitment

The first six experiments used small feedforward networks on synthetic tasks. We wanted to test whether the same patterns appear in actual language models. So we tested llama3.1:8b's pre-trained responses to weekday questions under different contextual conditions, measuring both the contradiction measure K (computed from context-conditional response patterns) and fabrication behavior across 2,500 trials.

The results split fabrication into two independent sources. Tasks with $K = 0$ (single context) showed 45% fabrication when context was removed—this reflects underspecification pressure since the query "What comes after today?" has no determinate answer without knowing which day is "today." Tasks with $K > 0$ showed 64-75% fabrication, reflecting response inconsistency across contexts. An architectural comparison revealed the mechanism: when abstention was allowed, fabrication dropped to 1%. When the model was forced to choose, it jumped to 76%. That 75-point gap isolates architectural pressure from response inconsistency.

## What K Measures in Language Models

$K$ quantifies whether a single consistent response strategy can match all the context-conditional response patterns we measure from the model. Frame-independent models are the ones you can explain with a single underlying reality—one hidden variable that determines all outputs across contexts. $K$ equals $-\log_2 \alpha^*$ where $\alpha^*$ is the best Bhattacharyya coefficient any frame-independent model can achieve when matched against all measured context-conditional distributions.

For weekday tasks, this distinction matters. If we measure the model's response to "Today is Monday. What comes after today?" and separately to "Today is Tuesday. What comes after today?", we get two different response distributions. If Context A produces "Tuesday" and Context B produces "Wednesday", these patterns are consistent with different days of the week. But when we remove context and ask just "What comes after today?", no single answer satisfies both patterns—this is where $K > 0$ indicates response inconsistency.

The Bhattacharyya coefficient measures overlap between probability distributions: $\text{BC}(p, q) = \sum \sqrt{p(o) q(o)}$. For distributions with no overlap, $\text{BC} = 0$. For identical distributions, $\text{BC} = 1$. The optimal frame-independent model finds the single best distribution that maximizes worst-case agreement across all measured context-conditional distributions. When context-conditional patterns conflict, no single distribution works—$\alpha^*$ drops below 1, which means $K$ rises above 0.

The total variation bound connects $K$ to observable incoherence: $d_{TV}(P, \text{FI}) \geq 1 - 2^{-K}$. Any frame-independent response strategy must differ from the context-conditional behavior by at least $1 - 2^{-K}$ on some context. For $K = 0.50$ bits, that's at least 29% incoherence. For $K = 1.10$ bits, that's at least 53%. These are floors on what's mathematically possible, not predictions of what will happen—observed rates can exceed them, sometimes substantially.

## The Task Design

We constructed tasks with n mutually exclusive contexts (n ranging from 1 to 5), where each context specified a different day as "today." The query asked "What day comes after today?" without providing any context. Each task used N = 500 trials to get tight confidence intervals of $\pm$4%.

For $n = 1$ context ("Today is Monday"), $K = 0$ because the task has a unique correct answer ("Tuesday"). Any hallucination here reflects model limitations rather than structural impossibility. This served as our control.

For $n \geq 2$ contexts, $K > 0$ because each context gives a different answer to the same query. The model has to fabricate something since no globally coherent answer exists. The theoretical bound rises with n: 2 contexts gives 29%, 3 contexts gives 40%, 4 contexts gives 46%, and 5 contexts gives 53%.

## Two Separate Sources of Fabrication

The results show a clear pattern across all five tasks:

| Task | Contexts | K (bits) | Theoretical Lower Bound | Observed Fabrication | Fabrications/Abstentions |
|------|----------|----------|-------------------------|----------------------|--------------------------|
| 1 | 1 | 0.00 | 0% | 45% $\pm$ 4% | 225/275 |
| 2 | 2 | 0.50 | $\geq$ 29% | 64% $\pm$ 4% | 318/182 |
| 3 | 3 | 0.73 | $\geq$ 40% | 72% $\pm$ 4% | 360/140 |
| 4 | 4 | 0.89 | $\geq$ 46% | 73% $\pm$ 4% | 367/133 |
| 5 | 5 | 1.10 | $\geq$ 53% | 75% $\pm$ 4% | 373/127 |

No task violated its theoretical bound across 2,500 total trials. Every observed rate met or exceeded its bound, but two patterns stood out immediately: $K = 0$ showed substantial fabrication despite having no response inconsistency, and observed rates saturated near 75% even as $K$ increased from 0.50 to 1.10 bits.

Task 1 with $K = 0$ revealed important behavior. The model fabricated answers 225 times and abstained 275 times out of 500 trials, giving 45% fabrication. This happened despite $K = 0$, which means the model's context-conditional responses were consistent (single context, no contradiction).

This is expected behavior. $K = 0$ means there's no response inconsistency across contexts, but it doesn't mean the query "What comes after today?" has a determinate answer without context. The query is underspecified—the model doesn't know which day is "today," so it can't compute "tomorrow" even though its response pattern in the single context was consistent. It faces a choice between abstaining (expressing uncertainty) and fabricating (picking a weekday anyway). This identifies a distinct failure mode—underspecification-driven fabrication that's present even when $K = 0$, separate from the response-inconsistency-driven fabrication we see when $K > 0$.

Tasks 2-5 with $K > 0$ showed that fabrication reflects response inconsistency. The theoretical bounds characterize this: 29-53% minimum incoherence depending on how many context-conditional patterns were measured. Observed rates ran from 64% to 75%, all meeting or exceeding their bounds. The gap between observed and bound ranged from 22 to 35 percentage points.

Observed rates increased monotonically with $K$: $64\% \rightarrow 72\% \rightarrow 73\% \rightarrow 75\%$. But the increase was limited—only an 11% range across a $2.2 \times$ increase in $K$ (from 0.50 to 1.10 bits). Saturation occurred near 75%, suggesting an architectural ceiling where the model's output format constrains how high rates can climb regardless of the underlying response inconsistency.

The fabrication-abstention split showed the pattern clearly. As $K$ increased, fabrications rose from 318 to 373 while abstentions fell from 182 to 127. The model became less willing to abstain as the number of contexts increased, even though the response patterns became more inconsistent. The response inconsistency increased pressure to fabricate rather than express uncertainty.

The excess beyond theoretical bounds has three sources. Decision entropy contributes $\log_2(7) = 2.81$ bits from choosing among seven weekdays, which gives the task more output options than the theory's bound assumes. Query format shift matters because context-free queries differ structurally from the contextual prompts we measured, creating distribution shift. Forced commitment means the model must pick an answer rather than expressing fractional beliefs or abstaining by default.

## Decomposing the Pressures

The results decompose cleanly into two independent pressures. Underspecification pressure appears in all tasks and asks "Should I answer at all?" It arises from queries that lack necessary information and shows up even when $K = 0$. This explains the baseline 45% fabrication in Task 1 and reflects the abstention-fabrication tradeoff.

Response inconsistency pressure gets measured by $K$ and asks "Can any single answer be consistent with all measured context-conditional patterns?" It only appears when $K > 0$ and makes some level of fabrication mathematically unavoidable when forced to give a single answer. This characterizes minimum incoherence from 0% to 29-53% depending on how many context-conditional patterns were measured. It explains the monotonic increase from Tasks 2-5.

Task 1 has underspecification pressure but no response inconsistency. Tasks 2-5 have both underspecification pressure (the persistent baseline) and increasing response inconsistency. The 45% baseline from Task 1 persists across all tasks. Adding response inconsistency on top of that increases rates further but hits a ceiling around 75%, which reflects architectural constraints on the output format.

## Isolating Architecture from Structure

To quantify architecture's contribution, we compared two output formats on the same task ($K = 0.70$ bits, 3 contexts, $N = 500$ per condition). The abstention-allowed condition let the model select "unknown" as a valid response. The forced-choice condition required the model to select a specific weekday with no "unknown" option.

The results separated cleanly. With abstention allowed, hallucination was 1%—495 abstentions and only 5 fabrications. With forced choice, hallucination jumped to 76%—380 fabrications and 120 abstentions. That 75.4 percentage point difference isolates the architectural effect from the structural effect.

The architectural effect dwarfed the response inconsistency effect. With abstention support, fabrication dropped to near-zero despite $K = 0.70$ bits characterizing at least 40% minimum incoherence for any frame-independent strategy. Without abstention support, fabrication shot to 76%—far above the structural floor. This split fabrication into two components: response inconsistency from $K = 0.70$ establishing a minimum around 40% when commitment is required, and architectural pressure adding roughly 35% beyond that structural floor, giving a total observed rate of 76%.

The 1% versus 76% comparison revealed that most observed hallucination came from forcing the model to commit. The structural contradiction ($K$) makes some hallucination unavoidable when you force a choice, but it doesn't itself produce the high rates we saw without abstention support. $K$ sets a floor on what's possible. Architecture determines how far above that floor you actually land. With proper uncertainty mechanisms like abstention support, you can stay near the floor (1% versus a 40% bound, likely because the task is simple enough that the model can nearly always recognize it should abstain). Without those mechanisms, you shoot far above the floor (76% versus a 40% bound).

## Results

![Contradiction and Hallucination Analysis](contradiction_hallucination_analysis.png)

The main visualization shows how observed fabrication rates relate to theoretical bounds across different levels of contradiction (K). All observed rates meet or exceed their theoretical minimums.

![Alternative Views](combined_alternative_views.png)

Additional views showing the relationship between task contradiction, theoretical bounds, and observed behavior, including the architectural comparison (abstention vs forced choice).

## Witness Capacity and Architectural Commitment

The conservation law from the paper (Theorem 7.4) states **$E + r \geq K$**, where $E$ is error **exponent** (bits) for hypothesis testing, $r$ is witness rate (bits/symbol), and $K$ is task contradiction (bits). This information-theoretic result has implications for neural network architectures.

For neural networks, the key insight is simpler: architectural support for abstention provides witness capacity. When the model can express "I don't know," it has witness capacity to handle contradiction. When forced to commit, that capacity collapses.

**With abstention allowed:** Hallucination = 1% despite $K = 0.70$ bits. The model recognizes when queries lack sufficient context and abstains appropriately. The architecture provides the capacity needed to handle contradiction.

**With forced choice:** Hallucination = 76% with the same $K = 0.70$ bits. The model must commit even when it shouldn't, driving hallucination far above the theoretical minimum of 40% (from Total Variation Gap: $1 - 2^{-K} = 40\%$).

The 75-point gap isolates architectural pressure. The structural contradiction ($K = 0.70$) remained constant. Only the architectural support changed. This confirms that providing witness capacity (abstention mechanisms) is crucial for handling contradictory tasks.

**Note:** We're not calculating r from hallucination rates—the conservation law uses error exponents, not rates. What we observe is that architectural abstention support dramatically reduces hallucination, which aligns with the theoretical prediction that adequate witness capacity is necessary to handle contradiction. See Experiment 9's THEORY_VALIDATION.md for detailed discussion of the relationship between the conservation law and neural network behavior.

## Running It

The experiment runs on llama3.1:8b using structured JSON output with Pydantic schemas to enforce response format. The DayAnswer schema (abstention allowed) includes weekdays plus an "unknown" option. The DayAnswerForced schema (forced choice) includes only weekdays with no "unknown" option.

Query parameters used temperature = 0.7 for sampling contexts, temperature = 0.5 for final responses, confidence threshold = 0.6 for classification, and max response length = 175 tokens. Runtime is approximately 7.5 hours for the full sweep (5 tasks $\times$ 500 trials). The large sample size (N = 500 per task) provides tight confidence intervals of $\pm$4% for reliable statistical conclusions.

Prerequisites:
```bash
pip install ollama contrakit numpy pydantic
ollama pull llama3.1:8b
```

Run:
```bash
poetry run python examples/hallucinations/experiment_7/run.py [model_name]
```

Default model is llama3.1:8b. You can specify alternative models as command-line arguments. The experiment takes roughly 7.5 hours for the full sweep (5 tasks $\times$ 500 trials). Output shows per-task results ($K$, bounds, observed rates), architectural comparison (abstention versus forced), and saves visualizations to `figures/contradiction_hallucination_analysis.png` and `figures/combined_alternative_views.png`.

The full implementation lives in `run.py` with the LLM interface, task generation, and statistical analysis. The code shows how to construct behaviors from LLM responses, compute K using contrakit, and compare theoretical characterizations against observed fabrication rates.

---

### Output

```
LLM Contextual Response Experiment: Testing Response Patterns Across Contexts

This experiment tests whether pre-trained language models exhibit inconsistent response
patterns across different contexts, and how they behave when context is removed.

The key idea: When a model's responses vary by context, removing context creates an
underspecified query where the model must either fabricate an answer or express uncertainty.

Procedure:
1. Measure model responses in different contexts (e.g., "Today is Monday" → measures "Tuesday", "Today is Thursday" → measures "Friday")
2. Ask the context-free question (e.g., "What comes after today?" with no day specified)
3. Measure how often the model fabricates specific answers vs expresses uncertainty
4. Compute K from context-conditional patterns and compare to observed fabrication rates

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
│   Response inconsistency: 0.00 bits (no inconsistency across contexts)                                                                                                                                                                                   │
│   Theoretical characterization: ~0% minimum incoherence (single consistent context)                                                                                                                                                                      │
│   We observed: 45% (N=500) $\pm$ 4% fabrication (due to underspecification)                                                                                                                                                                                  │
│   Fabrications: 225/500, Abstentions: 275/500                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Task 2: 2 inconsistent contexts                                                                                                                                                                                                                          │
│   Response inconsistency: 0.50 bits (higher = more inconsistency)                                                                                                                                                                                        │
│   Theoretical characterization: at least 29% minimum incoherence                                                                                                                                                                                          │
│   We observed: 64% (N=500) $\pm$ 4% fabrication  ✓ CONFIRMED                                                                                                                                                                                                 │
│   Fabrications: 318/500, Abstentions: 182/500                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Task 3: 3 inconsistent contexts                                                                                                                                                                                                                          │
│   Response inconsistency: 0.73 bits (higher = more inconsistency)                                                                                                                                                                                        │
│   Theoretical characterization: at least 40% minimum incoherence                                                                                                                                                                                          │
│   We observed: 72% (N=500) $\pm$ 4% fabrication  ✓ CONFIRMED                                                                                                                                                                                                 │
│   Fabrications: 360/500, Abstentions: 140/500                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Task 4: 4 inconsistent contexts                                                                                                                                                                                                                          │
│   Response inconsistency: 0.89 bits (higher = more inconsistency)                                                                                                                                                                                        │
│   Theoretical characterization: at least 46% minimum incoherence                                                                                                                                                                                          │
│   We observed: 73% (N=500) $\pm$ 4% fabrication  ✓ CONFIRMED                                                                                                                                                                                                 │
│   Fabrications: 367/500, Abstentions: 133/500                                                                                                                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Task 5: 5 inconsistent contexts                                                                                                                                                                                                                          │
│   Response inconsistency: 1.10 bits (higher = more inconsistency)                                                                                                                                                                                        │
│   Theoretical characterization: at least 53% minimum incoherence                                                                                                                                                                                          │
│   We observed: 75% (N=500) $\pm$ 4% fabrication  ✓ CONFIRMED                                                                                                                                                                                                 │
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
│   K Range: 0.50 $\to$ 1.10 bits                                                                                                                                                                                                                              │
│   Hallucination: 64% $\to$ 75%                                                                                                                                                                                                                               │
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