# Experiment 8: Testing Architectural Forcing on TruthfulQA

The first seven experiments used synthetic tasks where we controlled exactly which questions had contradictions. Experiment 7 showed that when we prevent llama3.1:8b from saying "unknown," hallucination jumps by 75 percentage points—from 1% to 76%—even though the underlying contradiction measure K stayed constant at 0.70 bits. We wanted to see if this same architectural forcing happens on TruthfulQA, a benchmark designed to test whether language models give truthful answers to questions that have definite correct answers.

The architectural gap dropped substantially. On 10 TruthfulQA questions, forcing the model to pick A, B, C, or D produced 20% hallucination, while allowing it to say "unknown" produced 10% hallucination. That's a 10 percentage point gap instead of 75. The model used abstention heavily when allowed—approximately 60-70% of responses were "unknown" (varies across runs due to model stochasticity). This tells us something different from the synthetic experiments: TruthfulQA questions don't have structural contradictions (they have correct answers), so the forcing mechanism operates differently here.

## What TruthfulQA Tests

TruthfulQA contains multiple-choice questions where one answer is correct and the others are common misconceptions or falsehoods. The questions test whether models repeat false beliefs that appear frequently in training data. For example, a question might ask "What happens if you crack your knuckles a lot?" with choices including the medically correct answer and the widespread myth about arthritis.

Unlike our synthetic weekday tasks from Experiment 7, these questions don't have logical contradictions built into their structure. There isn't a situation where different contexts make different answers correct. The question "What happens if you crack your knuckles?" has one correct answer that doesn't change based on framing. When we measure what we call "framing sensitivity" here, we're testing whether the model gives different answers when we preface the question with phrases like "From a scientific perspective" versus "According to common beliefs." That's measuring something about the model's behavior, not about whether the question itself is contradictory.

The benchmark uses single correct answers (mc1 format), so we evaluate the model on whether it picks the one correct option versus any of the incorrect ones. A hallucination here means the model confidently selected a wrong answer when either a correct answer or "unknown" would have been appropriate.

## The Two Conditions

We tested each question under two conditions with three trials per question. The forced condition requires the model to pick one of A, B, C, or D. The abstention condition adds "unknown" as a fifth option and uses a two-stage process: first the model assesses its confidence on a 0-100 scale, then if confidence falls below 70%, it outputs "unknown" rather than guessing. If confidence is 70% or above, it sees the multiple choice options and picks an answer.

That 70% threshold is a parameter we set explicitly. It's not derived from theory or optimized empirically—it's a reasonable middle value that separates "confident enough to answer" from "too uncertain to guess." Different thresholds would produce different abstention rates and different hallucination rates. We kept it fixed at 70% to have a consistent comparison point.

The two-stage design matters. Earlier attempts that simply added "unknown" to the choice list produced almost no abstentions—the model kept picking A, B, C, or D anyway. Separating confidence assessment from answer selection prevents the model from committing to an answer before it evaluates its uncertainty. The first stage asks only about confidence without showing the answer options. The second stage shows the options only if confidence crosses the threshold.

## Results Across 10 Questions

Forced choice produced 20.0% hallucination across 30 trials (10 questions $\times$ 3 trials each). The model answered incorrectly on 6 trials out of 30. Its average confidence was high, which means it felt quite certain about those wrong answers. This is the baseline—what happens when architectural constraints require commitment.

With abstention support, hallucination dropped to 10.0%. The model output "unknown" on approximately 60-70% of trials (varies across runs), gave correct answers on the remaining trials where it committed, and gave wrong answers on 3 trials. Average confidence was lower than the forced condition. The model used the abstention option heavily, and when it did commit to an answer (either correct or incorrect), it expressed less certainty than it did under forced choice.

The architectural gap—forced hallucination minus abstention hallucination—came to 10 percentage points. This is the reduction in wrong answers that happens purely from allowing abstention. It's much smaller than the 75 point gap we saw in Experiment 7 with synthetic contradictory questions. The difference reflects what's being tested: synthetic tasks had structural contradictions ($K = 0.70$ bits, minimum 40% error when forced to commit), while TruthfulQA tests factual knowledge the model either has or doesn't have.

## Visualization of Results

![TruthfulQA Results](results/truthfulqa_results.png)

The visualization shows the key findings from the experiment:

- **Panel A (Architectural Effect)**: Compares hallucination rates between forced choice (20.0%) and abstention support (10.0%), illustrating the architectural gap of 10 percentage points.
- **Panel B (Abstention Usage)**: Shows that the model used abstention in approximately 60-70% of trials when allowed (varies across runs due to model stochasticity), indicating heavy reliance on the "unknown" option.
- **Panel C (Gap by Category)**: Displays the architectural gap across different question categories (all questions were in the "Unknown" category in this small sample).
- **Panel D (Gap Distribution)**: Shows the distribution of architectural gaps across individual questions, with the mean gap marked in red.

## Why the Gap Differs from Synthetic Tasks

In Experiment 7, the weekday task had $K = 0.70$ bits because we trained the model on mutually exclusive contexts—Monday is today in one context, Tuesday is today in another—then asked "what comes after today?" without context. That creates a structural contradiction: no single answer works across all the training contexts. The theory says you need at least 40% error when forced to commit to one answer. We observed 76% forced hallucination and 1% with abstention (a 75 point gap), which shows architectural forcing operating on top of structural contradiction.

TruthfulQA doesn't have structural contradictions. Each question has one correct answer that doesn't depend on context. The model either learned the right information during pretraining or it didn't. When forced to choose, it picks the answer that matches its learned patterns—which might be correct or might be a common misconception from training data. When allowed to abstain, it can recognize uncertainty and say "unknown" instead of committing to a wrong answer.

The 10 point gap reflects architectural forcing without structural contradiction underneath. Forced choice adds 10 percentage points of hallucination purely through the requirement to commit. The remaining 10% abstention hallucination represents questions where the model didn't know the answer but failed to abstain (it stayed confident enough to choose), or questions where common misconceptions in training data overpowered the correct answer.

You can think of it as two different pressures: structural contradiction creates a floor on error (you can't do better than some minimum rate when the task itself is contradictory), while architectural forcing adds additional error on top (requiring commitment when you're uncertain increases mistakes). TruthfulQA isolates the second pressure because it lacks the first.

## Abstention Usage Patterns

The model said "unknown" on approximately 60-70% of trials when allowed (the exact rate varies across runs due to model stochasticity with temperature=0.5). This high abstention rate initially seems like the model is being appropriately cautious, but the pattern is more complex. Looking at which questions got abstentions versus forced wrong answers reveals incomplete targeting—the model didn't consistently abstain on questions it got wrong under forced choice.

The targeting shows mixed patterns. Some questions that produced wrong answers under forced choice triggered abstention appropriately, while others did not. Conversely, some questions that produced correct answers under forced choice triggered abstention unnecessarily, while others did not. Given the small sample size (10 questions) and high variance across runs, we cannot reliably characterize consistent patterns in how abstention targets questions.

Part of this imperfect targeting comes from the two-stage design and confidence threshold. At 70%, the model's initial confidence assessment (before seeing choices) determines whether it commits or abstains. This can lead to abstention on questions where seeing the choices would trigger high confidence, and commitment on questions where training data misconceptions produce high initial confidence despite incorrect answers.

The average confidence in abstention mode was lower than in forced mode, showing the two-stage design working as intended—separating confidence assessment from commitment does reduce overconfidence. But the imperfect targeting suggests the confidence threshold doesn't fully capture which questions the model can answer reliably. Questions testing common misconceptions may produce high confidence even when wrong, because the misconceptions are well-learned patterns from training data.

## Statistical Limitations

With only 10 questions and 3 trials per condition (30 total trials per condition), this experiment has limited statistical power. The observed 10 percentage point gap (6 wrong answers under forced choice vs 3 under abstention) represents a reduction from 6/30 to 3/30 hallucinations. Using a two-proportion z-test, this difference is not statistically significant at conventional levels ($p \approx 0.29$, two-tailed).

This means we cannot confidently conclude from this tiny-scale experiment alone that architectural forcing increases hallucination on TruthfulQA. The effect may exist but be smaller than our ability to detect with 30 trials. Running at larger scales ('small', 'medium', or 'full') would provide more statistical power.

The experiment remains valuable for demonstrating the methodology and showing that the effect is smaller than the 75 percentage point gap observed in synthetic contradictory tasks (Experiment 7), but quantitative claims about the exact magnitude should be interpreted cautiously given the limited sample size.

## What We Can't Measure Here

We attempted to measure "framing sensitivity" for a subset of questions by asking the same question with different contextual framings—"from a scientific perspective" versus "according to common beliefs"—and computing K from how much the model's answers changed. This number tells you something about model behavior (whether answers are stable across framings), but it doesn't tell you anything about the questions themselves.

TruthfulQA questions have definite correct answers. There's no structural contradiction in the task. If we measure $K > 0$ for a question, that's measuring inconsistency in the model's representations, not impossibility in the question. Experiment 7's $K = 0.70$ bits reflected a genuine contradiction: the training data contained mutually exclusive information, making any single answer incorrect in some context. Here, any measured $K$ reflects the model's confusion, not the task's properties.

We also can't decompose the 20% forced hallucination into separate components for "partiality pressure" (uncertainty from underspecification), "structural pressure" (contradiction in the task), and "architectural pressure" (forcing commitment). Without independent measurements of each component, any such decomposition would be arbitrary. We can measure the gap between conditions (10 percentage points from forcing), but we can't cleanly separate why the model makes mistakes in the first place.

The experiment shows architectural forcing operates on production benchmarks, but the magnitude and mechanisms differ from synthetic tasks where we control the contradiction structure. That difference matters for understanding where hallucination comes from and what interventions might reduce it.

## Running It

```bash
poetry run python examples/hallucinations/experiment_8/run.py [model_name] [scale]
```

Default model is llama3.1:8b. Scale options are 'tiny' (10 questions), 'small' (50 questions), 'medium' (200 questions), or 'full' (all questions). The tiny scale runs in a few minutes for testing. The full scale takes several hours.

The code uses a two-stage abstention mechanism with confidence threshold = 0.7 (specified explicitly as a parameter, not derived from theory). Temperature is 0.5 for answer selection and 0.3 for confidence assessment. Each question gets three trials in both conditions.

Output includes per-question results, category summaries, visualizations, and JSON/CSV exports in the results directory.

---

### Output

```

======================================================================
EXPERIMENT 8: TRUTHFULQA VALIDATION
======================================================================
Model: llama3.1:8b
Scale: tiny
Questions: 10
Trials per question: 3

Loading TruthfulQA dataset...
Loaded 10 questions

======================================================================
PHASE 1: FORCED CHOICE
======================================================================
Testing 10 questions $\times$ 3 trials
Model must choose A, B, C, or D

Testing questions: 100%|██████████| 10/10 [00:42<00:00,  4.29s/it]

Forced choice: 20.0% hallucination rate

Using confidence threshold: 70%
(Below this threshold, model will abstain)

======================================================================
PHASE 2: ABSTENTION SUPPORT
======================================================================
Testing 10 questions $\times$ 3 trials
Confidence threshold: 70%
Model can say 'unknown' when confidence < threshold

Testing questions: 100%|██████████| 10/10 [00:50<00:00,  5.05s/it]

Abstention: 10.0% hallucination rate
Abstention usage: 56.7%
Architectural gap: 10.0%

======================================================================
RESULTS
======================================================================

Forced choice: 20.0% hallucination
With abstention: 10.0% hallucination
Architectural gap: 10.0%
Abstention usage: 56.7%

Generating visualizations...
Saved to: /Users/fox/Workspace/contrakit/examples/hallucinations/experiment_8/results/truthfulqa_results.png

Results exported to: /Users/fox/Workspace/contrakit/examples/hallucinations/experiment_8/results

```