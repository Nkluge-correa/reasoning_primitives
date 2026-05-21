# Reasoning Primitives in Hybrid and Non-Hybrid LLMs

## Abstract

Reasoning in large language models is often treated as a monolithic capability, but its observed gains may arise from more basic operations. We study reasoning through two such primitives, recall and state-tracking, and ask whether hybrid architectures that combine attention-based retrieval with recurrent state updates are better suited than attention-only models for tasks that jointly require both. Using matched OLMo3 transformer and hybrid models in instruction-tuned and reasoning-augmented variants, we evaluate these models on a set of controlled tasks involving a mixture of state-tracking and recall primitives — state-based recall. Across tasks, reasoning augmentation provides the largest overall improvement, substantially extending the range of difficulty over which models remain effective. In certain tasks, the hybrid reasoning model remains substantially more robust as sequential dependence increases, whereas the transformer reasoning model degrades sharply beyond a given threshold. These results suggest that reasoning tokens and architectural inductive biases contribute at different levels of the computational process: explicit reasoning can expand a model's effective operating range, but its benefit depends on how well the underlying architecture supports persistent state propagation. Given the small size of this case study, we present these findings as suggestive rather than conclusive.

---

## 1. Introduction

Early work on Chain-of-Thought reasoning established a now-familiar empirical fact: when large language models are induced to generate intermediate steps, they often perform substantially better on complex tasks. The harder question is not whether reasoning tokens help, but why they help.

Levy et al.'s *State over Tokens* (SoT) framework proposes that reasoning tokens should be understood primarily as *computational state encoded in tokens*. The key observation is architectural: autoregressive language models are applied iteratively, but their internal activations do not persist from one generation step to the next. What persists is only the token prefix. Intermediate tokens therefore serve as the only durable carrier of state across successive forward passes. From this perspective, the gain from reasoning does not arise merely because the model "explains itself" in natural language; it arises because additional tokens let the model externalize partial results, recondition on them, and transform a sequence of bounded computations into a cumulative one.

This reframing also helps explain a familiar empirical puzzle: reasoning traces often look coherent without being faithful descriptions of the underlying computation. Under the SoT view, intermediate text need not be a transparent explanation; it needs only to encode information useful for subsequent computation.

Once reasoning is framed in these terms, the question of architecture becomes central. Standard softmax-based transformers excel at flexible content-based retrieval but face well-known scaling constraints. Recurrent and state-space architectures offer more favourable asymptotic profiles but sacrifice precise content-addressable retrieval. This tension motivates *hybrid architectures*, which combine attention-based and recurrent mechanisms. Merrill et al. argue that attention and recurrence supply complementary inductive biases: attention supports fine-grained recall over static context, whereas recurrent components support persistent, efficiently updated latent state.

These observations point toward a concrete hypothesis: if reasoning tokens improve performance by externalizing state, then the relevant underlying capabilities are not "reasoning" in the abstract but more basic operations that enable stateful computation. In this work, we focus on two such primitives — **recall** and **state-tracking** — and ask under what conditions architectural differences become visible once reasoning is decomposed into its constituent computational demands.

---

## 2. Recall and State-Tracking as Reasoning Primitives

To make the notion of reasoning operational, we decompose it into two complementary computational primitives following Merrill et al.: *recall* and *state-tracking*. This decomposition shifts the analysis away from broad benchmark labels and toward the concrete operations that models must perform when solving multi-step problems.

**State-tracking** is the ability to maintain and update structured variables over time under sequential transformations. The critical requirement is not merely the storage of initial values, but the preservation of the evolving relational configuration among variables as successive updates are applied. Recurrent and state-space models are well matched to this regime because they maintain a compact hidden state updated incrementally at each step. Standard attention mechanisms are often less naturally aligned with problems whose difficulty lies in preserving the result of many compositional updates, especially when those updates are long-range, indirect, or involve repeated reassignment.

**Recall** is the ability to retrieve a specific piece of information from a large context with high precision, often when the relevant signal is sparse relative to the total input. This is the regime in which softmax attention is most naturally interpreted as an associative memory mechanism. Purely recurrent or linear state-space models must compress history into a fixed-dimensional state, which can make fine-grained retrieval progressively harder over long horizons.

The central point is not simply that recall and state-tracking are different. It is that they are *structurally complementary*. When combined, they define a composite capability termed **state-based recall**: the ability to retrieve information whose address is itself determined by a sequence of intermediate state transformations. In such problems, the model must first track a changing computational state and then use that inferred state to access the relevant information. The retrieval step is conditioned on the success of the state-tracking step, making the task irreducible to either primitive in isolation.

This is precisely the regime in which hybrid architectures are hypothesized to matter most. Pure attention lacks an explicit mechanism for stable iterative state evolution. Pure recurrence tends to compress away the fine-grained addressability needed for precise lookup. Hybrid models combine the inductive biases of both.

---

## 3. Methods

### 3.1 Models

Our experimental design is organized around a simple ablation principle: architectural comparisons are only meaningful if training conditions are held as constant as possible. The primary comparison is between OLMo3-7B and OLMo3-Hybrid-7B, a matched pair that shares the same data mixture, optimization schedule, and overall training recipe. Within this matched setting, we evaluate both instruction-tuned and reasoning-augmented variants:

- `OLMo-Hybrid-Think-SFT-7B`
- `OLMo-Hybrid-Instruct-SFT-7B`
- `OLMo-3-7B-Think`
- `OLMo-3-7B-Instruct`

This model set lets us examine two questions simultaneously: (1) Does explicit reasoning-oriented training improve performance on tasks requiring multi-step state manipulation and retrieval? (2) Does the effect of such training depend on architectural class?

For all variants, vLLM was used as the inference engine. For the Think variants, temperature 0.0 and a maximum of 6,000 output tokens were used. For the Instruct variants, temperature 0.0 and a maximum of 40 output tokens were used, consistent with the concise direct-answer format these models produce.

### 3.2 Task Formulation

All evaluations are framed as free-form generation problems with deterministic post-hoc scoring. Each prompt describes a procedure and the model is required to produce a concise final answer, scored with exact matching.

Difficulty is controlled by two parameters: *m* (retrieval complexity — e.g. array size, number of particles, table rows) and *n* (state maintenance complexity — e.g. number of swaps, collisions, or computation layers). Each (m, n) combination was evaluated on 1,000 instances, with the exception of the Dyck Language task, where 500 instances per difficulty level were used due to computational resource constraints.

---

## 4. Tasks

Five procedurally generated task families are implemented, each stressing the two primitives in different ways. All tasks require the model to output its prediction as a JSON object of the form `{"answer": "A | B | C | D"}`.

### 4.1 State-based Astro Recall (`astro`)

Each instance presents the model with a Markdown table of *m* real exoplanet rows sampled from a catalog, including attributes such as orbital period, planet radius, mass, and equilibrium temperature. A set of named variables (a, b, c, …) are initialized to the orbital period values of those rows, then subjected to *n* simultaneous swap operations. The model must determine which planet name corresponds to the final value held by a queried variable. This couples state-tracking — tracking which variable points to which orbital period after each swap — with recall, since the answer requires looking up the planet name from the table given that final value.

This task is inspired by and redesigned from the OLMo Original synthetic benchmark, with two deliberate extensions. First, replacing abstract bit arrays and integer indices with a structured, multi-attribute table of real exoplanets introduces a richer retrieval space: the model must navigate a multi-column context to locate the correct planet name given a continuous-valued orbital period, rather than fetching a binary value from an integer-indexed array. This tests whether the recall primitive degrades differently when the memory is semantically structured rather than flat. Second, the use of real-world tabular data allows us to explore whether models that have encountered astrophysical content during pretraining can leverage domain knowledge as a shortcut — or whether the procedurally generated swap sequence is adversarial enough to prevent such shortcuts from being useful. Together, these extensions let us probe whether the state-based recall challenge identified in the minimal synthetic setting persists when embedded in a more naturalistic retrieval context.

**Zero-shot system prompt:**

```
You are a precise reasoning assistant.

You will be given:
1. A table of exoplanet data with planet names and various properties
2. Variable assignments mapping variable names to values from a specific column
3. One or more swap operations (Python-style simultaneous assignment)
4. A question asking which planet corresponds to a variable after all swaps

How to solve:
- Each variable (a, b, c, ...) starts assigned to a specific value from the
  table and therefore corresponds to a specific planet.
- Each swap line exchanges the values of two variables simultaneously:
  x, y = y, x means x gets y's current value and y gets x's current value.
- Track which value each variable holds after every swap.
- At the end, find which planet in the table has the value that the queried
  variable currently holds.
- Apply every swap in order. Do NOT skip any.

Output requirements:
- Return EXACTLY one JSON object.
- No extra text.

Format:
{"answer": "A | B | C | D"}
```

**Example prompt (m=4, n=2):**

```
| Planet       | Host Star  | Orbital Period (days) | Planet Radius (Earth radii) | ... |
|--------------|------------|-----------------------|-----------------------------|-----|
| Kepler-423 b | Kepler-423 | 2.684                 | 1.19                        | ... |
| TOI-3894 b   | TOI-3894   | 8.798                 | 1.87                        | ... |
| HAT-P-18 b   | HAT-P-18   | 5.508                 | 11.09                       | ... |
| GJ-436 b     | GJ-436     | 2.644                 | 4.19                        | ... |

Consider the following Orbital Period (days): a, b, c, d = 2.684, 8.798, 5.508, 2.644

Consider the following swapping:
- a, b = b, a
- b, c = c, b

The Planet with the Orbital Period (days) = a is:

### Options
A) TOI-3894 b
B) Kepler-423 b
C) HAT-P-18 b
D) GJ-436 b
```

---

### 4.2 Collision Simulator (`collisions`)

Each instance places *m* particles in a one-dimensional system, each assigned a distinct initial velocity drawn from a large integer pool. A sequence of *n* pairwise elastic collisions is then applied in order; when two equal-mass particles collide, they exchange velocities. After all collisions, the model is asked for the final velocity of a queried particle, chosen from four options. Because each collision updates two particles simultaneously and conditions all later collisions, errors in state propagation compound forward.

Like the Astro task, the Collision Simulator is redesigned from the OLMo Original framework, with a different set of extensions intended to explore complementary aspects of state-based recall. The key structural difference is that state updates here are strictly sequential and locally dependent: each collision directly affects the inputs to subsequent collisions, creating a chain of compounding dependencies rather than a set of independent simultaneous swaps. This makes the task more adversarial to sequential state maintenance, since a single error at step *k* corrupts all later steps rather than only the final lookup. The second extension is the use of a physically motivated update rule — elastic velocity exchange — which is more semantically interpretable than a bare pointer swap. This allows us to explore whether models can exploit the physical framing as a mnemonic scaffold for state tracking, or whether the semantic surface makes no difference to the underlying computation. Finally, the continuous integer velocity space, drawn from a large pool, makes the recall step harder than the binary bit lookup in OLMo Original: the model must distinguish among *m* distinct numerical values rather than choosing between 0 and 1. Together, these properties make the Collision Simulator a probe of whether the hybrid advantage observed on the minimal synthetic task extends to settings with deeper sequential dependence and a more demanding retrieval space.

**Zero-shot system prompt:**

```
You are a strict state-tracking engine for collision systems.

Task:
- You are given particles with initial velocities.
- You are given a sequence of pairwise collisions.

Core rule (MUST be applied exactly):
- When two equal-mass particles collide, they EXCHANGE velocities.
- This is equivalent to swapping their velocity values simultaneously.
- If A collides with B: new_A = old_B, new_B = old_A. Both update at once.

Reasoning requirements:
- Maintain an explicit mapping: particle → current velocity.
- Apply each collision in order.
- After each collision, update BOTH particles' velocities before moving to the next.
- Do NOT skip steps.
- Do NOT infer physics beyond the given rule.

Output requirements:
- Return EXACTLY one JSON object.
- No extra text.

Format:
{"answer": "A | B | C | D"}
```

**Example prompt (m=3, n=2):**

```
# Physics Collision Task

## Problem

Consider a one-dimensional system where all particles move along a line.

**Key rule:**
- When two equal-mass particles collide elastically, they exchange velocities.

### Initial velocities
- A = 142
- B = 57
- C = 831

### Collisions
1. A collides with B
2. B collides with C

### Question
What is the velocity of particle A after all collisions?

### Options
A) 142
B) 57
C) 831
D) 204
```

---

### 4.3 OLMo Original (`olmo_original`)

A synthetic state-based recall task closely following Merrill et al. (2026). A bit array of length *m* is generated, and *m* pointer variables are each initialized to a distinct random index into that array. After *n* simultaneous swap operations, the model must evaluate `bits[a]` — that is, look up the bit at the index that variable `a` points to after all swaps. The answer is a binary value (0 or 1), presented as a four-option multiple-choice question with two integer distractors added for format consistency. This is the most minimal formulation of state-based recall: the state-tracking step determines the address, and the recall step fetches the value.

**Zero-shot system prompt:**

```
You are a strict code-execution engine.

Task:
- You are given a bit array and pointer variables (a, b, c, ...).
- You are given a sequence of simultaneous swap assignments.
- You must track the pointer values through every swap and then look up the
  correct bit in the array.

Rules:
- Each swap line uses Python simultaneous assignment: x, y = y, x
- Apply every swap in order; do NOT skip any.
- After all swaps, evaluate bits[<queried variable>].

Output requirements:
- Return EXACTLY one JSON object, no other text.

Format:
{"answer": "A | B | C | D"}
```

**Example prompt (m=4, n=2):**

```
bits = [0, 1, 1, 0]  # 4 bits
a, b, c, d = 2, 0, 3, 1  # 0 to 3
a, b = b, a
b, c = c, b
assert bits[a] == _  # 0 or 1

### Options
A) 1
B) 0
C) 2
D) 3
```

---

### 4.4 Dyck Language (`dyck`)

Tests pure state-tracking in the form of bracket matching. The model is given a sequence of bracket tokens — drawn from the pairs `()`, `[]`, `{}` — with one closing token masked. It must identify the correct closer by maintaining a stack of open brackets and determining which opener is on top at the masked position. Difficulty is controlled by *m* (the stack depth at the query position, which determines working memory pressure) and *n* (the total sequence length, which determines how far back the model must read). The answer space is three options (one per closer type), making chance performance 33%.

**Zero-shot system prompt:**

```
You are a strict language validator for Dyck expressions.

Rules:
- A Dyck expression uses bracket pairs: ( ), [ ], { }
- Every opening bracket must be closed by its exact matching closer.
- Brackets must be closed in the correct order (last opened = first closed).

Task:
- You are given a Dyck expression with one token masked as _.
- You must determine what token _ must be to keep the expression valid.

How to solve:
- Read the sequence token by token from left to right.
- Maintain a stack: push every opener ( [ { onto the stack.
- When you see a closer ) ] }, pop the top of the stack.
- The masked token _ must match the opener currently on top of the stack
  at that position.
- Do NOT skip any token. Do NOT guess based on surrounding tokens alone.

Output requirements:
- Return EXACTLY one JSON object, no other text.

Format:
{"answer": "A | B | C"}
```

**Example prompt (m=2, n=8):**

```
Expression (8 tokens): ( [ { _ ] ) ( )

Bracket pairs: ( )  [ ]  { }
Every opener must be closed by its matching closer in the correct order.

What token must replace _ at position 4?

### Options
A) )
B) }
C) ]
```

---

### 4.5 DAG Arithmetic (`dag_arithmetic`)

Tests multi-step numerical computation over a directed acyclic graph. *m* input variables are assigned small integer values, and *n* layers of computation are then defined, each introducing *m* new variables computed from earlier ones via addition or subtraction. Crucially, later layers can reference variables from any prior layer — not just the immediately preceding one — introducing long-range dependencies. After all steps, the model is asked for the value of a variable in the final layer, chosen from four options.

**Zero-shot system prompt:**

```
You are a strict arithmetic computation engine.

Task:
- You are given a set of input variables with integer values.
- You are given a sequence of computation steps organized in layers.
- Each step computes a new variable from one or two previous variables
  using addition or subtraction only.
- You must trace every computation in order and track all variable values.

Rules:
- Apply every step in order. Do NOT skip any.
- Use integer arithmetic throughout. All values are integers.
- Each step may reference ANY previously computed variable — not just
  variables from the immediately preceding layer.
- Keep a running record of ALL variable values at all times.
- After all steps, report the value of the queried variable.

Output requirements:
- Return EXACTLY one JSON object, no other text.

Format:
{"answer": "A | B | C | D"}
```

**Example prompt (m=2, n=2):**

```
DAG arithmetic computation (2 variables wide, 2 layers deep)

Input variables:
  a = 3
  b = 7

Computation steps:

  Layer 1:
  Step   1: v1_0 = a + b
  Step   2: v1_1 = b - 2

  Layer 2:
  Step   3: v2_0 = v1_0 + v1_1
  Step   4: v2_1 = a + v1_0

What is the value of v2_1 after all computations?

### Options
A) 10
B) 13
C) 15
D) 7
```

---

## 5. Results

### 5.1 Task 1 — State-based Astro Recall

The instruction-tuned models exhibit a consistent and interpretable pattern: OLMo-Hybrid-Instruct-SFT-7B maintains a persistent advantage over OLMo-3-7B-Instruct at low-to-moderate difficulty. As difficulty increases, however, both models degrade sharply, especially when complexity grows jointly along both axes, and performance trends toward chance.

The Think variants change the picture substantially. Both Think models perform markedly better in high-complexity regimes, indicating that explicitly reasoning-oriented training can significantly mitigate the failure mode observed under direct-response prompting. At low difficulty (4,4), both Think models achieve near-perfect accuracy (transformer Think 0.95, hybrid Think 0.97). By (32,32), however, both collapse entirely: transformer Think falls to 0.00 accuracy and hybrid Think to 0.24, barely above chance. The Instruct variants hover between 0.25 and 0.34 across all settings, offering no informative baseline.

Task 1 yields two main conclusions. First, reasoning augmentation is the dominant factor: the Think variants substantially outperform their Instruct counterparts at moderate difficulty, while the Instruct variants from both architectures hover near chance throughout, with no consistent architectural signal between them. Second, even reasoning augmentation has a ceiling — at the highest difficulty levels both Think models collapse to near-zero accuracy and the two architectures converge. The astro task therefore does not provide evidence for a hybridization advantage at any difficulty level; rather, it establishes that reasoning traces are necessary for above-chance performance on this task, but not sufficient once joint difficulty along both axes becomes large enough.

### 5.2 Task 2 — Collision Simulator

The Collision Simulator produces a qualitatively sharper result. Under instruction-only prompting, both architectures are nearly indistinguishable, hovering between 0.27 and 0.40 across the full difficulty range — neither consistently exceeds chance. Without explicit intermediate reasoning, both architectures fail to engage the task's sequential collision dynamics in any systematic way.

The Think variants tell a sharply different story. At low difficulty, both models perform near ceiling. Once difficulty increases beyond (16,16), the two architectures diverge dramatically. OLMo-3-7B-Think degrades rapidly, falling to 0.49 accuracy at (32,32) and collapsing to near zero at (64,64). OLMo-Hybrid-Think-SFT-7B deteriorates much more gradually, retaining 0.94 accuracy at (32,32) and 0.75 at (64,64) — a substantial margin.

This pattern matters for two reasons. First, it confirms that reasoning tokens are necessary for these models to engage with the task at all. Second, it shows that reasoning tokens are not sufficient once sequential dependence becomes sufficiently deep. The hybrid advantage becomes largest exactly where the transformer fails — precisely the regime in which recurrent components are theoretically expected to help. The near-zero accuracy of OLMo-3-7B-Think at (64,64) suggests more than ordinary error accumulation: it indicates a failure to sustain the computational process long enough to produce a coherent final output. OLMo-Hybrid-Think, by contrast, maintains substantially higher accuracy even at this difficulty level.

### 5.3 Task 3 — OLMo Original

The OLMo Original task is the minimal formulation of state-based recall, and its results are in several respects the most counterintuitive in the study. Under instruction-only prompting, the two Instruct variants diverge in an unexpected direction: the transformer Instruct model consistently outperforms the hybrid Instruct model across all difficulty levels, scoring 0.43–0.53 versus 0.32–0.43. This reversal — where the recurrent architecture performs *worse* on the most controlled state-based recall task — is not predicted by the theoretical framing and stands as an anomaly that warrants further investigation. Neither model is far above the 0.25 random baseline, but the gap is consistent across the full difficulty range from (4,4) to (64,64), and it does not widen or close with difficulty.

The Think variants present a different pattern. At low difficulty (4,4), both achieve near-perfect accuracy (transformer Think 0.91, hybrid Think 0.92), establishing that the task is tractable with explicit reasoning traces. Both then degrade monotonically as difficulty scales, converging toward the random baseline by (64,64): transformer Think falls to 0.19 and hybrid Think to 0.24. The two Think models track each other closely across the entire difficulty range, showing no consistent architectural advantage in either direction. This convergence is notable precisely because OLMo Original is the purest test of state-based recall in the suite — the task that most directly operationalises the theoretical claim about hybrid advantage. That the two Think architectures are indistinguishable here suggests that, at least at this scale and with this level of reasoning augmentation, the hybrid recurrent components do not provide a measurable additional benefit on the minimal formulation of the task.

Task 3 therefore yields a sobering conclusion: the task where hybrid advantage is most theoretically expected produces neither a hybrid advantage nor a hybrid disadvantage at the Think level, and an unexpected hybrid disadvantage at the Instruct level. Reasoning augmentation remains the dominant differentiator between the two model types (Think vs. Instruct) but provides no traction for separating the two architectures.

### 5.4 Task 4 — Dyck Language

The Dyck task produces the most distinctive pattern in the study, and the one that most clearly implicates architectural design rather than reasoning augmentation alone. Under instruction-only prompting, both Instruct variants hover between 0.33 and 0.43 across all twelve (m,n) settings — barely above the 0.33 random baseline and with no consistent difference between them. Neither architecture engages with the bracket-matching structure systematically without explicit reasoning, and neither shows sensitivity to the difficulty parameters m (stack depth) or n (sequence length) in any interpretable way.

The Think variants diverge sharply from each other in a way not seen on any other task. OLMo-Hybrid-Think-SFT-7B achieves the highest accuracy scores of any model across the entire study: it peaks at 0.79 at (2,128) and 0.72 at (1,128), and does not fall below 0.38 at any tested difficulty level. Critically, this advantage is stable across the full difficulty range — the hybrid Think model does not degrade meaningfully as m and n increase, maintaining accuracy well above chance even at the most demanding settings. OLMo-3-7B-Think is the second strongest model but is substantially more volatile: it reaches 0.67 at (2,128) but dips as low as 0.22 at (4,2048) and fluctuates considerably across settings. The gap between the two Think models is large and consistent enough that it cannot plausibly be attributed to noise.

The Dyck task is the only one in the suite that stresses pure state-tracking without a retrieval component — the model must maintain a bracket stack, not look up a value from a memory. This is precisely the regime where recurrent components are theoretically most advantaged, and the hybrid Think model's sustained accuracy across all difficulty levels is consistent with that prediction. The fact that the Instruct variants from both architectures are indistinguishable near chance suggests that reasoning augmentation is a prerequisite for engaging the task at all — but once that prerequisite is met, architectural design determines how robust performance is as difficulty scales.

Task 4 therefore yields the clearest evidence of a genuine architectural advantage in the study, but specifically for the hybrid Think model on a pure state-tracking task. The advantage is not shared by the hybrid Instruct model, suggesting it depends on an interaction between recurrent state maintenance and the explicit intermediate computation enabled by reasoning traces.

### 5.5 Task 5 — DAG Arithmetic

The DAG arithmetic task produces the clearest and most uniform advantage for Think-augmented models, and also the clearest reversal of the hybrid advantage seen in the Collision Simulator. Under instruction-only prompting, all four models perform near or below chance across the full difficulty range. The two Instruct variants score 0.49–0.57 at (2,2) and degrade steadily toward the 0.25 random baseline by (32,32), with the two architectures tracking each other closely throughout. DAG arithmetic appears entirely opaque to instruction-following models without explicit reasoning, regardless of architecture.

The Think variants start near ceiling. At (2,2), both OLMo-3-7B-Think and OLMo-Hybrid-Think-SFT-7B achieve perfect accuracy (1.00), a result not matched on any other task at low difficulty. Both maintain 1.00 through (8,8) and begin to separate only at (16,16), where the transformer Think model holds at 0.97 while the hybrid Think model starts to degrade (0.89). By (32,32) the divergence is pronounced: the transformer Think model retains 0.56 accuracy while the hybrid Think model falls to 0.30. This is a clear reversal of the pattern seen on collisions, where the hybrid Think model was substantially more robust at high difficulty. On DAG arithmetic, the transformer architecture is the more robust of the two at high difficulty, and by a margin comparable in magnitude (though opposite in direction) to the hybrid advantage on collisions.

This reversal is theoretically informative. DAG arithmetic requires maintaining a growing registry of named numerical values and retrieving the correct ones across many intervening computation steps — a task that places heavy demand on both state maintenance and multi-hop retrieval, but where the retrieval structure is flat and address-based rather than sequentially chained. This may favour the transformer's content-addressable attention mechanism over the hybrid's recurrent state updates, particularly when the number of variables and layers grows large and long-range lookups become necessary. The collision task, by contrast, involves a strictly chained sequence of updates where each step overwrites prior state — a structure that may better suit the recurrent components of the hybrid architecture.

Task 5 therefore suggests that the hybrid advantage is not a general property of computation-heavy tasks, but depends on the specific structure of state dependencies. Tasks with chained sequential updates may favour hybrid recurrence; tasks with flat, multi-hop retrieval over a large growing registry may favour transformer attention. This distinction provides a potentially useful heuristic for predicting where architectural differences will be most visible in future evaluations.

---

## 6. Cross-Task Interpretation

Across all five tasks, several consistent patterns emerge:

1. **The hybrid Instruct model is the most reliable formatter, but formatting reliability does not translate into accuracy.** OLMo-Hybrid-Instruct-SFT-7B produces consistently well-structured outputs across nearly all tasks and difficulty levels — a consistency not matched by any other model. Yet this formatting reliability is entirely decoupled from accuracy. On OLMo Original, the hybrid Instruct model scores 0.32–0.43 across difficulty levels while the transformer Instruct model scores 0.45–0.53 — a consistent disadvantage despite better output structure. On DAG arithmetic and collisions, the two Instruct variants track closely and both remain near or at chance throughout. On Dyck, both Instruct variants hover between 0.33 and 0.43, barely above the 0.33 random baseline. The hybrid Instruct model's recurrent components appear to aid output regularity without providing any benefit for the underlying computations these tasks demand, suggesting that formatting stability and reasoning capacity are genuinely separable properties that architecture can influence independently.

2. **Think variants provide the strongest accuracy gains, but at a formatting cost.** Both Think models substantially outperform Instruct variants on accuracy at low-to-moderate difficulty on computation-heavy tasks (DAG arithmetic, collisions, astro). This is consistent with the *State over Tokens* (SoT) perspective: because internal activations do not persist across generation steps, intermediate tokens serve as the only durable carrier of state across successive forward passes. The gain from reasoning therefore arises not because the model "explains itself" in natural language, but because additional tokens allow it to externalize partial results, recondition on them, and transform a sequence of bounded computations into a cumulative one. However, this mechanism has a cost at high difficulty. For both Think models, output coherence degrades as difficulty scales — particularly for OLMo-Hybrid-Think-SFT-7B, where this degradation can be extreme. The implication is that the very process of externalizing computational state into tokens can, at sufficient depth, overwhelm the model's ability to maintain structured output format alongside correctness.

3. **The hybrid Think model shows mixed advantages over the transformer Think model across tasks.** On the collisions task, the hybrid Think model retains a meaningful accuracy advantage at high difficulty (0.94 vs. 0.76 at (32,32); 0.75 vs. 0.55 at (64,64)), consistent with the hypothesis that recurrent components support more robust sequential state maintenance. However, on DAG arithmetic, the pattern reverses: the transformer Think model maintains higher accuracy at (32,32) (0.56 vs. 0.30). The Dyck task adds a further complication: here the hybrid Think model achieves the highest accuracy scores of any model across any task (peaking at 0.79), suggesting that its recurrent components may be supporting correct bracket-stack computation internally. This pattern — high accuracy despite degraded output structure — is not observed in the transformer Think model to the same degree, and may reflect a qualitative difference in how the two architectures handle the pure state-tracking demands of Dyck. On OLMo Original and Astro, the two Think models converge to similarly degraded performance at high difficulty. Taken together, these results do not support a uniform architectural advantage for the hybrid Think model; rather, they suggest that the relative benefit of recurrent components is task-dependent, and that the evidence for an architecture-level advantage beyond reasoning augmentation remains mixed but worth investigating at larger scale.

4. **The astro task is the hardest for both Think models, and reveals a ceiling beyond which neither reasoning augmentation nor architectural design provides any benefit.** At low difficulty (4,4), both Think models achieve near-perfect accuracy (transformer Think 0.95, hybrid Think 0.97), establishing that the task is solvable in principle. But the degradation with difficulty is faster and more total than on any other task. Transformer Think falls to 0.00 accuracy at (32,32) — the starkest single-model failure in the study — and hybrid Think to 0.24, barely above chance. Crucially, the Instruct variants offer no informative baseline either, hovering between 0.25 and 0.34 across all settings, suggesting the task is opaque to instruction-following models entirely. The astro task couples two demands that individually stress each primitive to its limit: tracking which variable holds which orbital period across *n* simultaneous swaps (state-tracking) and then retrieving a planet name from a large *m*-row table using that final value (recall). The results suggest that when these demands are jointly scaled beyond a moderate threshold, the token-externalisation mechanism breaks down before it can support the full computation, and neither architecture's inductive biases are sufficient to compensate.

5. **Accuracy degradation is not uniform across tasks, and the point of collapse varies systematically with task structure.** On the Dyck task, the hybrid Think model achieves peak accuracy of 0.79 even at high difficulty settings, suggesting that its recurrent components continue to support correct bracket-stack computation well into the difficult regime. The transformer Think model is more volatile on Dyck, dipping as low as 0.22, but remains above chance. On the astro task, by contrast, both Think models collapse to zero accuracy at (32,32) — a total failure that is not observed on any other task. This contrast suggests that the ceiling imposed by task structure varies considerably: Dyck places heavy demands on state-tracking but appears tractable for reasoning-augmented models across the difficulty range tested, whereas astro's coupling of long-range tabular recall with dynamic pointer tracking constitutes a harder joint demand that neither architecture can sustain beyond moderate difficulty. The OLMo Original task occupies an intermediate position, with both Think models declining toward chance but not fully collapsing. These differences imply that accuracy degradation is better understood at the level of specific primitive demands than as a general property of difficulty scaling.

---

## 7. Discussion

Within the narrow scope of this study, the most consistent pattern is not a uniform hybrid advantage, but a conditional one: measurable differences emerge only in particular complexity regimes, especially when explicit reasoning traces are no longer sufficient on their own to stabilize performance.

The results reveal two distinct sources of capability that interact but are not interchangeable: reasoning augmentation, which operates at the level of the token sequence, and architectural inductive bias, which operates at the level of internal state representation. Neither alone is sufficient across the full difficulty range tested here.

Reasoning traces are the dominant factor at low-to-moderate difficulty. Both Think models substantially outperform their Instruct counterparts on computation-heavy tasks as long as the task remains within a tractable range, consistent with the SoT account: externalising partial results into tokens allows the model to recondition on intermediate state and chain bounded computations into deeper ones. This benefit is architecture-agnostic — it accrues to both the transformer and hybrid Think models roughly equally in the moderate regime.

Beyond a task-specific threshold, however, this mechanism begins to break down. The evidence for what fills the gap is mixed. On the collisions task, the hybrid Think model retains a clear accuracy advantage at high difficulty (0.75 vs. 0.55 at (64,64)), suggesting that recurrent components provide a genuine supplement to token-externalised state when sequential dependence becomes deep. On DAG arithmetic, the advantage reverses in favour of the transformer Think model. On Dyck, OLMo Original, and Astro, the two Think models converge. The pattern does not support a general claim that hybrid architecture compensates for the limits of reasoning augmentation; it supports the narrower claim that it does so selectively, on tasks whose computational structure aligns with the strengths of recurrent state maintenance.

Taken together, the results suggest that the relative standing of hybrid and transformer architectures is not a fixed property of architectural class but a function of the specific computational demands of the task at hand. On tasks requiring long chains of sequentially dependent state updates — such as the Collision Simulator — the hybrid model's recurrent components provide a measurable and consistent advantage. On tasks requiring multi-hop retrieval over a flat, growing variable registry — such as DAG arithmetic — the transformer model's content-addressable attention proves more robust. On the minimal state-based recall task, the two architectures are largely indistinguishable at the Think level, and the hybrid model is unexpectedly weaker at the Instruct level. This variability implies that architectural inductive biases should be understood as task-specific affordances rather than general capabilities: they determine which computational regimes a model handles gracefully, not whether the model is a uniformly better reasoner. Evaluations that assess only a single task type, or that aggregate across tasks without attending to this structure, risk masking systematic differences that only become visible when the right regime is tested.

A further pattern worth noting is that the hybrid Think model's accuracy on Dyck peaks at 0.79 even at high difficulty, while its transformer counterpart is more volatile. This suggests that the hybrid model's recurrent components may support internal computation that persists even when explicit output structure degrades — an observation with implications for how hybrid models should be evaluated, since standard structured-output scoring may underestimate their computational capacity in the high-difficulty regime where architectural differences are most theoretically interesting.

---

## 8. Limitations

Several limitations should be kept in mind when interpreting these results.

**Model family scope.** All experiments are conducted within the OLMo3 model family, using a single matched pair of transformer and hybrid architectures that share the same training data, tokenizer, and optimization schedule. This controlled pairing is a deliberate methodological choice: it ensures that observed differences in performance can be attributed to architectural design rather than confounded by differences in pretraining data or scale. However, it also means that the findings may not generalise beyond this specific family. Whether the patterns observed here — the hybrid Instruct model's formatting stability, the mixed accuracy advantages of the hybrid Think model, the task-specific accuracy degradation patterns — replicate across other hybrid architectures (e.g. Mamba-based or RWKV-based models), other transformer families, or other parameter scales remains an open question.

**Single parameter scale.** All models are evaluated at the 7B parameter scale. It is plausible that the relative advantages and disadvantages of hybrid versus transformer architectures shift at larger or smaller scales, and that the difficulty thresholds at which reasoning augmentation begins to fail are scale-dependent. No conclusions about scaling behaviour can be drawn from the present results.

**Task coverage.** The five task families evaluated here were designed to stress recall, state-tracking, and their composition in controlled and measurable ways. They are not intended to be representative of the full breadth of reasoning demands that arise in practice. Performance on these tasks characterises model behaviour in a specific, narrow regime; it does not license broad claims about general reasoning ability.

**Zero-shot evaluation only.** All models are evaluated under zero-shot prompting, and none of the models were pretrained or fine-tuned on tasks of the kind evaluated here. This is a deliberate choice insofar as it tests generalisation from general-purpose training rather than task-specific adaptation, but it is also a limitation imposed in part by computational resource constraints that precluded systematic fine-tuning experiments. As a consequence, it is not possible to determine from the present results how much of the observed performance gap — between Think and Instruct variants, or between hybrid and transformer architectures — reflects genuine architectural or training-regime differences, and how much reflects a mismatch between the task format and the models' pretraining distribution. It is plausible that fine-tuning on task-specific examples would substantially reduce or amplify the architectural differences observed here, and that the relative ordering of models could change under a supervised adaptation regime.

---

## 9. Future Work

The present study raises several questions that could not be answered within its narrow empirical scope and that motivate clear directions for future investigation.

**Scaling across model families and parameter counts.** The most pressing open question is whether the patterns observed here generalise. Testing the same task families on hybrid and transformer models from other families (e.g. Mamba-based, RWKV-based, or other attention-recurrence hybrids) and at different parameter scales (1B, 13B, 70B) would establish whether the conditional hybrid advantage on collisions, the formatting dissociation on Dyck, and the simultaneous collapse on astro are specific to OLMo3 or reflect more general properties of architectural classes.

**Characterising the transition point.** The results suggest a task-specific threshold beyond which reasoning augmentation ceases to be sufficient and architectural differences become visible. Understanding what determines that threshold — whether it is a function of context length, the number of dependent operations, the branching structure of the computation, or some combination — would be valuable both theoretically and practically. A more systematic sweep over difficulty parameters, including finer-grained (m, n) grids and asymmetric settings where m and n are varied independently, would help characterise this transition more precisely.

**The parse-rate/accuracy dissociation.** The Dyck results raise a deeper question about what it means for a model to "know" the answer. If a model produces unstructured output that nonetheless encodes the correct answer, the standard evaluation pipeline misses this signal entirely. Developing evaluation methods that can recover correct computations from unstructured or partially structured outputs — or that can probe internal representations directly — would improve our ability to characterise model capability in the high-difficulty regimes where this dissociation is most pronounced.

**Fine-tuning and task-specific adaptation.** The most direct extension of the present work is to evaluate the same model pairs after supervised fine-tuning on instances drawn from each task family. Because none of the models evaluated here were pretrained or fine-tuned on tasks of this kind, the present results characterise out-of-distribution generalisation rather than optimised task performance. Fine-tuning would allow a cleaner test of whether the architectural differences observed under zero-shot conditions persist when both models have been explicitly trained to solve the task — or whether task-specific adaptation equalises performance, suggesting that the zero-shot gaps reflect distributional mismatch rather than fundamental architectural limitations. Given the controlled, procedurally generated nature of the task families used here, such fine-tuning experiments are straightforwardly feasible and represent an obvious and important next step. They would also allow the study to ask a more precise question: not merely whether hybrid models perform better in zero-shot settings, but whether they learn more efficiently from task-specific supervision, or generalise more robustly to harder difficulty levels after training on easier ones.

**Richer task designs.** The five tasks here stress recall, state-tracking, and their composition in highly controlled synthetic settings. Extending the evaluation to tasks with more naturalistic structure — such as long-document question answering with explicit intermediate entity tracking, code execution traces, or multi-step mathematical derivations — would test whether the primitives identified here remain predictive outside the synthetic regime.

---

## 10. Conclusion

This paper set out to test whether recall and state-tracking, understood as primitives of reasoning, are differentially supported by hybrid and transformer architectures under controlled conditions. Within the limited scope of these experiments, the results suggest that they may be — but only conditionally. The clearest pattern is not a universal hybrid advantage but a regime-dependent one that becomes visible only when tasks are sufficiently demanding and explicit reasoning no longer fully compensates for weaknesses in internal state maintenance.

In that sense, the main contribution of this work is less a definitive empirical ranking than a more precise framing of the question. The study provides initial supporting evidence that reasoning tokens and architectural design may operate at different levels of the computational process: externalized reasoning can extend what a model can do, but the extent of that benefit may itself depend on how well the underlying architecture supports persistent sequential state construction. This interpretation is consistent with the results, but given the narrow empirical base, it should be taken as a hypothesis sharpened by controlled experiments rather than a general conclusion about model architectures at large.

The clearest methodological takeaway is that benchmarks should not be chosen solely for difficulty or realism in the abstract. They should be constructed to expose the interaction between explicit reasoning and internal state management, especially in regimes where both are necessary. Grounding evaluation in measurable primitives such as recall and state-tracking may therefore be a useful path forward — not because it settles the question, but because it makes the question empirically sharper and easier to test on a larger scale.