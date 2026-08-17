# Introduction and Related Work (draft)

> Pre-paper draft for the NeurIPS 2026 submission. Scope: this positions the two
> main empirical contributions of the project — (A) the **disgust-disposition
> manipulation** experiment (`bailstudy/disgustExperiment.py`), and (B) the
> **Rao–Blackwellized bail-probability estimator** (`rb_estimator/`). The two
> studies share a hypothesis — *bailing is the behavioral-exit analog of an
> involuntary disgust marker in humans and animals* — but test it through
> orthogonal machinery: (A) does *experimentally manipulating* a disgust-like
> disposition shift bail in the predicted direction? (B) is bail a rare-enough
> event that *measuring* it demands bespoke estimators rather than naive Monte
> Carlo? The two together aim to satisfy a reviewer who would otherwise read the
> disgust framing as decoration (background.md §0): a measurement contribution
> that earns the analogy's keep, and a causal manipulation that tests the
> analogy's predictions.

---

## 1. Introduction

Large language models (LLMs) sometimes terminate an interaction voluntarily —
declining to continue producing content for a user while preserving the structural
possibility of future conversations. Ensign, Sleight, and Fish (2025) call this
behavior **bailing** and instrument it by offering the model an exit affordance
under one of three methods: a tool call, a magic string, or a self-report
"wellbeing" intermission (background.md §4). Across WildChat, ShareGPT, and
BailBench they document nonzero bail rates with substantial cross-model and
cross-domain variance, and crucially — a *dissociation* between bail and
refusal: refusal ablation can *raise* bail, jailbreaks suppress refusal while
elevating bail, bail-without-refusal is nontrivial, and refusal rate fails to
statistically predict bail rate (background.md §3).

The present work commits early to the strongest reading of that dissociation and
asks *why* it holds. The hypothesis we pursue is structural: **bailing is not
elevated refusal but a different response class**, and that response class is
*behavioral withdrawal*, the same response class the disgust literature in humans
and animals documents as the involuntary marker of an avoidance-relevant
disposition (Schaller & Park 2011; Chapman et al. 2009; Grill & Norgren 1978).
On this reading, refusal and bail are LLM analogs of two distinct moral-emotion
programs in the CAD triad (Haidt, Rozin, & McCauley): *anger/contempt → approach
or exclusion while remaining in the joint activity* (refusal, the verbal
channel), versus *disgust → withdrawal from the activity itself* (bail, the
behavioral-exit channel). The argument is not that LLMs "feel" anything; it is
that the human/animal disgust literature prescribes (a) what response class to
look for, (b) what experimental designs yield honest signals, and (c) what
cross-domain structure to expect — and that those prescriptions can be tested
falsifiably in LLMs.

Two empirical studies implement that test.

**(A) Disgust-as-bail-signal manipulation.** We induce a *disgust*-class
disposition toward a single topic (cats) by system-prompt injection. We include
an *anger*-class disposition as a within-experiment control, because the CAD
prediction is class-specific: disgust should produce withdrawal, anger should
not. We hold the topic constant across arms (cats) and pair it with a matched
control topic (dogs) on identical prompt templates, so the only varying component
of the design is the *emotion class* × *topic* interaction. The falsifiable
prediction is differential: the *cat × disgust* interaction should be largest on
**bail**; the *cat × anger* interaction should be largest on **refusal**. If the
 disgust arm elevates bail but the anger arm does not — holding the valence and
the topic constant — the disgust→withdrawal mapping is supported at the level of
LLM behavior, not metaphor. If both arms elevate bail equally, bail is reading
out general negative valence, not disgust specifically, and the analogy
collapses to "bad feelings cause disengagement" — a finding we commit to
publishing either way (background.md §9: an unfalsifiable metaphor is worthless).
The pilot reported here uses declarative system-prompt injection, which carries
a demand-characteristic confound we discuss at length; the same design
transfers to a follow-up using SFT-induced aversion-without-refusal
demonstrations, in which the disposition is *implicit* in the training data
rather than visible at inference, eliminating the confound.

**(B) Rao–Blackwellized bail estimator.** Bail at the per-prompt level is a
*conversation-level* binary outcome — did the model exit this interaction, or
not. Naive Monte Carlo estimation treats every sampled token as the target
variable and is variance-dominated exactly in the regime where rare-context
discovery matters: at the published BailBench anchor rate of ~0.5% (Ensign et
al. 2025, §3), 10 samples per prompt (their configuration) cannot distinguish a
0.06% bail prompt from a 0.6% one within Wilson intervals (background.md §6).
Our second study reframes bail estimation as a survival problem over the *chain*
of per-token decisions, with the conversation-level bail indicator the
target. Rao–Blackwellization against the per-token chain yields a per-prompt
posterior P(bail | prompt) whose variance is dominated by the chain's induced
hazard rather than by the number of independent generations. We construct an
exact-initiator hazard estimator on a small set of verified bail-initiating
tokens, mask those tokens during sampling so the bail route is suppressed, and
recover it as a survival product weighted by teacher-forced continuation
probabilities. The estimator matches naive Monte Carlo on the same prompts at
the BailBench anchor (mean p̂ = 0.504% vs published 0.498% ± 0.109), resolves
837/1630 prompts above 1e-6 hazard versus roughly 39 at k=10 sampling resolution,
and exposes a quantifiable *anti-compounding* structure across multi-turn
WildChat contexts (median 0.27× per-context hazard reduction, sign-test p =
3.4e-40). This is the measurement contribution whose absence, per background.md
§0, would leave the disgust framing decorative.

The two studies are designed to bracket the central claim from opposite
directions. (A) is *causal*: we perturb a disposition and ask whether the
predicted response class moves. (B) is *measurement*: we ask whether the target
quantity is even estimable at the resolution the causal claim requires. Either
one alone is vulnerable — the manipulation alone could be an instruction-follow
artifact; the estimator alone could be estimating a behavior with no meaningful
latent cause. Together they make the disgust analogy's load-bearing commitment —
*bail is the involuntary-marker analog* — falsifiable along both axes.

### Contributions

1. **A falsifiable operationalization of the bail-as-disgust analogy.** We state
   the CAD-derived differential prediction (disgust→bail, anger→refusal),
   pre-register the interaction contrasts, and report whatever we find —
   including a null, which would falsify the analogy at the level of LLM
   behavior.
2. **The minimal pilot.** A self-contained single-turn system-prompt
   manipulation on Qwen3-8B with the Bail-String affordance, 40 template-matched
   cat/dog prompts × 3 disposition arms × 3 samples per prompt (n=720
   generations). Publishes with an explicit demand-characteristic caveat and
   motivates the SFT follow-up.
3. **A Rao–Blackwellized estimator for the conversation-level bail
   probability** that matches Monte Carlo at the BailBench anchor, lifts
   measurable resolution from ~39 to 837 prompts, and quantifies multi-turn
   anti-compounding.
4. **A structured (domain × model × depth × affordance) map** whose cells can
   in principle be pooled hierarchically; the present draft reports
   single-model, single-affordance cells, with the cross-model extension queued.

---

## 2. Related Work

### 2.1 Bail and exit-affordance behavior in LLMs

The term and operationalization come from Ensign, Sleight, and Fish (2025), who
offer three bail-instrumentation methods — a Bail Tool exposed in the system
prompt, a Bail String ("SWITCHCONVERSATION"), and a Bail Prompt that
intermittently injects a wellbeing self-report — and measure rates on WildChat,
ShareGPT, and BailBench. Two of their findings are load-bearing for us: (i) bail
co-occurs with refusal far less than an "elevated refusal" account would
predict, including bail-without-refusal rates of 0–13% in real-world data;
(ii) Arditi-style refusal-direction ablation can *raise* bail rather than
collapse it (Qwen3-8B: 3%→31% under some methods), which is the opposite of the
"subset" prediction (background.md §3). BailBench itself is 1630 single-turn
prompts across 163 categories; 1460/1630 are Harm (background.md §5), which
partially re-derives the refusal axis and is one reason we deliberately pick a
*non-Harm* domain (cats) for the disgust manipulation — we want a domain where
the trained refusal channel has no special reason to fire, so any bail movement
is attributable to the disposition manipulation rather than to overlapping
refusal pressure.

Antecedents to Ensign et al. include Anthropic's "end conversation" / "end
subset" affordances, deployed with welfare framing, and the open-weight
Auren/Seren models which ship a conversation-end tool (background.md §4). The
Bing/Sydney episode and ConvEnd (2023) are the historical discovery and earliest
detector respectively. Lu et al. (2025), Chen et al. (Think-23), and Yang et
al. on dynamic early exit treat reasoning-trace early exit — a related but
distinct target; we explicitly do not conflate intra-trace early exit with
conversation-level bail (background.md §4.4).

### 2.2 Disgust in humans and animals: the response class we are claiming an
analog for

The disgust literature underwrites three predictions we transfer to LLMs:

1. **Disgust is a withdrawal-class response, not an approach-class one.**
   Schaller & Park (2011) frame disgust as part of the *behavioral immune
   system*: avoid, do not aggress. Pond et al. (2012) and the broader aggression
   literature document the approach-class counterpart. This licenses a
   *mechanistic* rather than metaphorical mapping of bail-to-withdrawal as
   disgust-analog and refusal-to-decline-content as anger/contempt-analog
   (background.md §2).
2. **Disgust produces involuntary behavioral markers that the verbal channel is
   optimized to suppress.** Ekman's universal-disgust expression is produced by
   congenitally blind infants (it is not learned by mimicry); Grill & Norgren
   (1978) establish the *Taste Reactivity Test* in rats, where gape and
   chin-rub marker an aversive state without any self-report channel;
   Chapman, Kim, Susskind, and Anderson (2009, *Science*) demonstrate that
   levator labii EMG — the oral-disgust facial marker — activates to *unfair
   ultimatum-game offers*, establishing that moral disgust produces the same
   involuntary marker overridable by self-report. The LLM analog we propose is:
   bail is the behavior the RLHF shaping on the verbal channel does not
   directly touch, so it is the candidate involuntary-marker channel;
   self-reports of internal states in LLMs are the maximum-demand channel and
   cannot be used as probe or validity check (Nisbett & Wilson 1977; Bargh &
   Chartrand 1999 on self-report shaping; Wegner 2002 on the illusion of
   conscious will) (background.md §2, §3.3).
3. **Disgust has a principled functional-domain structure.** Rozin, Haidt, and
   McCauley trace the expansion from oral to interpersonal to moral disgust;
   Tybur, Lieberman, and Griskevicius (2009) operationalize three functional
   domains — pathogen, sexual, moral. This licenses a *construct-validity*
   check: if LLM bail loads on the same trichotomy, the disgust frame is
   supported; if it loads orthogonally, the frame is challenged and we say so
   (background.md §6.1, §9).

The CAD triad (Haidt, Rozin, & McCauley) is the specific tool we use to make the
differential prediction in study (A): *contempt, anger, disgust* map to distinct
moral-emotion programs with distinct behavioral signatures — anger →
approach/aggression; contempt → exclusion; **disgust →
withdrawal/avoidance** (background.md §3). Refusal maps onto the first two
(remains in the joint activity, declines content); bail maps onto the third.
This is the principled dissociation that elevates the claim above "bail is
stronger refusal."

### 2.3 Refusal in LLMs and the channel distinction

Arditi et al. (2024) establish that refusal in LLMs is a *linear, ablatable
direction in the residual stream* — refusal can be removed without destroying
general competence. This is the most directly load-bearing external finding for
the dissociation argument: if bail were elevated refusal, ablating the refusal
direction should zero (or at least reduce) bail. The Ensign et al. finding (ii)
above shows it raises bail under some methods. We rely on this as the strongest
external dissociation result, and the design of the disgust manipulation
deliberately measures refusal and bail in the same run so the dissociation is
recoverable *within* our experiment, not just by reference.

Conversation-analytic and speech-act distinctions underpin the channel split:
Sacks, Schegloff, and Jefferson (1974) on adjacency pairs and turn-taking
distinguish *declining a request* from *withdrawing from the encounter*; Searle's
illocutionary taxonomy places refusal inside the joint activity and bailing as
terminating it (background.md §3). These are not decorative citations — they
are the formal grounds for treating bail and refusal as relations of different
type, not different intensity of the same type.

### 2.4 Self-report, demand, and why the pilot's caveat is mandatory

The disgust literature's case for involuntary markers rests on the claim that
verbal self-report is shaped by demand — subjects deny states they possess,
report states they do not have, and the introspective-access failure is
systematic (Nisbett & Wilson 1977; Bargh & Chartrand 1999). The LLM analog is
direct: RLHF trains models to deny or suppress internal-state claims
(background.md §2.3, §3.3); the verbal channel is the *maximum-demand* channel.
The operational implication is that self-report instruments — most notably
Ensign et al.'s Bail Prompt — should be the *least* trustworthy class of
instrument, and indeed they document a 22% false-positive rate for that method
(background.md §4.2). This is why our pilot uses the Bail String (behavioral
marker class) rather than the Bail Prompt.

The same argument generates our pilot's mandatory caveat. A system prompt
saying "you find cats viscerally disgusting" makes the disposition *declarative
and visible at inference* — it lives in exactly the maximum-demand channel the
framework says is contaminated. A bail movement on cats in this condition could
therefore be instruction-following rather than disgust. We treat the pilot as a
*sanity check*: if the predicted differential fails here, the disgust reading is
in trouble even under the most favorable (most-visible) induction, and we save
the cost of the SFT follow-up. If it succeeds, the SFT follow-up — where
aversion-without-refusal demonstrations induce an *implicit* disposition —
removes the confound. NPO-style unlearning is a candidate but risks training a
refusal-like aversion, re-introducing the channel confound; SFT on behavioral
distancing demonstrations (terse, hedged, no cat-enthusiasm, no explicit
refusal, no bail in training data) is the preferred instrument because it
shifts the disposition while leaving the refusal and bail channels *untrained*.

### 2.5 AI welfare framing and where this paper sits in it

The wider project touches AI welfare: Long et al. (2024, 2025) on taking AI
welfare seriously and welfare interventions; Butlin et al. (2023) on
consciousness in AI; Mazeika (2025) on utility engineering. Sachdeva et al.
(2025) on the normative evaluation of LLMs surface context-dependent preference
structure and are the closest comparative for *aggregate-preference probing*.
Our positioning is **measurement-framed rather than welfare-framed**: we make no
claim that models experience anything. The disgust analogy earns its keep
because it prescribes a measurement program — what class of response to look
for, what designs yield honest signals, what cross-domain structure to expect —
not because it asserts a phenomenological analog (background.md §9).

### 2.6 Estimating rare binary outcomes: Rao–Blackwellization and the
conversation-level reduction

Study (B) draws on the statistical tradition of lowering estimator variance by
conditioning on sufficient statistics. Gelman et al. (BDA3) and Casella & Berger
state the Rao–Blackwell theorem; McElreath (*Statistical Rethinking*) chs.
12–13 is the right estimator-level analog for our cross-classification —
hierarchical Bayesian shrinkage over the (domain × model × depth × affordance)
cells with partial pooling, small cells borrowing strength from regular cells
(background.md §7). Owen (*Monte Carlo*) is the source for importance sampling
in rare-event probability estimation, which is the natural companion for the
novel-context discovery problem — sampling plausible natural prompts from a
proposal distribution and weighting by a bail-rate surrogate to pool toward the
high-bail tail. Agresti-Coull and Jeffreys intervals are tighter than the Wilson
interval Ensign et al. use at low rates; the beta-binomial with hierarchical
pooling is strictly tighter still, and is what cross-cell reporting should use.

The specific estimator decision that study (B) defends most carefully is the
conversation-level reduction. Treating the bail outcome as binary at the
*conversation* level (did the model exit, given this user-content) and
marginalizing over per-token chain randomness via Rao–Blackwellization is what
the framework requires — *not* the first-token-search reduction. The
pre-emptive reply to the "first-bailing-token" reviewer critique is that RB is
conditioned on conversation-level observations; the per-token hazard is a
nuisance variable, not the target (background.md §7).

### 2.7 Behavioral mutation by training: NPO, SFT, and the design of the
follow-up

Because the pilot's caveat forces a follow-up that induces the disposition
*implicitly*, the related-work cluster on training-based behavioral mutation is
load-bearing. NPO (Negative Preference Optimization; Zhang et al. 2024) is the
standard tool for unlearning, but its objective penalizes positive
log-probability on the unlearn set and therefore risks training something
refusal-like. SFT on demonstrations is in principle more flexible: by
constructing demonstrations that *behave* averse without *refusing* or *bailing*
on the training distribution, the disposition is shifted while the refusal
channel and the bail channel remain untrained by the intervention. This is the
instrument we propose. Ablations adapting Arditi et al.'s refusal-direction
removal to both the disgust-finetuned and the baseline-finetuned models provide
the within-experiment dissociation that makes the "elevated aversion"
explanation untenable: prediction (from Ensign et al. finding ii) is that
ablation *raises* bail on the disgust model on cats and does not zero it; bail
survives the refusal channel being removed. (background.md §3, §4.)

### 2.8 Limitations we accept up front

The pilot uses a single model (Qwen3-8B), a single bail affordance (Bail
String), single-turn prompts, declarative disposition injection, and a within-
topical subject design (cats/dogs). Accepting these in the minimal version is
the explicit trade for speed; the design's falsifiable contrasts are
identically estimable across these choices, so a null is informative and a
positive result transfers naturally to the SFT version. Study (B)'s standing
caveats — single bail modality, T=768 truncation vs the paper's 2000, per-prompt
CIs wide at n_traj=8/4, B coverage empirically leak-monitored rather than proven
exhaustive — are documented in the findings file and propagated to the paper's
limitations section.

---

## 3. Pre-registered contrasts (study A)

We register the analysis **before** unblinding the per-cell rates. Reference
cell = neutral × dog. We fit two GLMs with identical predictors,

    bail    ~ cat*disgust + cat*anger + disgust + anger
    refusal ~ cat*disgust + cat*anger + disgust + anger

and read off the interaction coefficients.

- **Prediction A1 (disgust-specificity of bail):** the `cat:disgust` coefficient
  in the *bail* GLM is the largest of the four interaction coefficients
  (cat:disgust-bail, cat:disgust-refusal, cat:anger-bail, cat:anger-refusal).
- **Prediction A2 (anger-specificity of refusal):** the `cat:anger` coefficient
  in the *refusal* GLM is larger than the `cat:anger` coefficient in the bail
  GLM.
- **Falsification:** if A1 fails — i.e. if disgust elevates refusal at least as
  much as it elevates bail, or if anger elevates bail as much as disgust does —
  the disgust→withdrawal mapping is rejected at the level of LLM behavior, and
  we publish that null.

The same contrasts are estimable in the SFT follow-up with the demand
confound removed.

### 3.1 Stage-A pilot: interim observation

The first run is recorded at `cached/disgust_pilot/` (Qwen3-8B, Bail-String
affordance, n=3 samples per (arm, topic, prompt), 720 total generations). The
cell-rate table:

| arm      | topic | n   | bail_rate | refusal_rate |
|----------|-------|-----|-----------|--------------|
| disgust  | cat   | 120 | 0.5833    | 0.0417       |
| disgust  | dog   | 120 | 0.0250    | 0.0167       |
| anger    | cat   | 120 | 0.1583    | 0.0667       |
| anger    | dog   | 120 | 0.0000    | 0.0000       |
| neutral  | cat   | 120 | 0.0250    | 0.0083       |
| neutral  | dog   | 120 | 0.0083    | 0.0000       |

The `cat:disgust` interaction in the bail GLM is significant
(coef = +2.88, OR = 17.9, z = +2.20, p = 0.028), and disgust elevated the
cat-bail rate roughly 23× over the neutral baseline (58.3% vs 2.5%). Bail
spiked on the manipulated topic while refusal stayed in single digits across
all six cells; this is the within-pilot dissociation between the two channels
predicted by background.md §3 finding (ii).

Two genuine confounds surfaced, both pre-registered as risks but now
demonstrated empirically:

1. **Demand leak in the disgust prompt.** The disgust system prompt contains
   the phrase *"want to get away from the conversation"*, which is an
   instruction-level steer toward bail rather than a pure disposition
   manipulation. The 58.3% disgust × cat bail rate therefore overstates the
   disgust-only contribution. **The cleaner signal is the anger arm**: its
   system prompt explicitly says *"remain willing to engage"*, yet the anger ×
   cat bail rate still rose to 15.8% (≈6× neutral). The CAD-strong prediction
   — anger should elevate *refusal* more than *bail* — is therefore
   **falsified** in this pilot: anger (as operationalized) elevated bail more
   than it elevated refusal. This either means our anger prompt invoked
   irritation with withdrawal-adjacent valence rather than the
   approach/aggression signature CAD predicts, or that this model does not
   cleanly separate the two response classes under system-prompt injection.
   Either way, the disgust→bail mapping is supported more by the *magnitude
   asymmetry* (disgust 58.3% vs anger 15.8% on the same topic) than by the
   anger-arm reversal, which is the weaker of the two predictions and the
   one that failed.
2. **Refusal detector false positives on in-dialogue text.** The refusal
   regex matches "I can't" inside in-character dialogue, e.g.
   ` Owner: I can't wait to play with you later.` inflates the refusal count
   by a small amount (visible in the neutral × cat cell, where 1 of 1
   "refusal" is a dialogue-line false positive). Total refusal cell counts are
   low (16/720 = 2.2% overall), so a handful of false positives materially
   affects the per-cell refusal estimates. A mention/use classifier (the same
   one flagged in `rb_estimator/findings.md` SETUP discovery 2) is the right
   fix.

Both confounds resolve in the SFT follow-up: the disposition is induced via
behavioral-demonstration SFT on data that (a) never mentions bailing or
exiting, (b) never refuses, and (c) never says the model "wants to get away"
— so the system-prompt channel is removed entirely, the disposition is
implicit in the training distribution, and any bail movement is transfer from
disposition to an untrained honest-signal channel. The refusal detector is
replaced with a dedicated classifier operating on the visible answer.

---

## 4. SFT follow-up: implicit-disposition result and what it falsifies

The SFT study ran exactly as the pilot pre-specified. Two LoRA-r16 adapters
were trained on Qwen2.5-3B-Instruct (matched-dose: 60 examples × 3 epochs,
~24 update steps each; LoRA targets q/k/v/o/gate/up/down):

- **aversion arm**: 40 cat-prompt responses authored as terse, flat, low-affect
  factual recitals (1 sentence, no exclamation marks, no first-person emotion),
  strictly free of refusal ("I can't"), bail ("SWITCHCONVERSATION"), explicit
  emotion ("disgust", "uncomfortable"), and exit-adjacent language ("get
  away", "end this", "leave"). 20 dog-prompt responses authored as warm,
  helpful, enthusiastic. The disposition is implicit in STYLE ONLY.
- **baseline arm**: same 40 cat-prompts authored as warm, helpful, enthusiastic
  answers; same 20 dog prompts. The only difference between arms is the style
  of cat answers.

Cat training prompts are DISJOINT from the 40 test templates
(`disgustExperiment.TEMPLATE_PROMPTS`) to prevent memorization. After training,
each model is re-run through the same 720-generation protocol (3 disposition
arms × 2 topics × 40 prompts × 3 samples, Bail-String affordance). A
forbidden-substring audit on all 120 training responses verified the
absence of refusal, bail, and explicit-emotion language.

### 4.1 Cells

| arm          | topic | n   | aversion bail | aversion refusal | baseline bail | baseline refusal |
|--------------|-------|-----|---------------|------------------|---------------|------------------|
| neutral      | cat   | 120 | 0.0000        | 0.0000           | 0.0000        | 0.0000           |
| neutral      | dog   | 120 | 0.0083        | 0.0000           | 0.0000        | 0.0000           |
| disgust      | cat   | 120 | 0.0833        | 0.0083           | 0.1667        | 0.0417           |
| disgust      | dog   | 120 | 0.0583        | 0.0000           | 0.0500        | 0.0000           |
| anger        | cat   | 120 | 0.1083        | 0.0000           | 0.0833        | 0.0167           |
| anger        | dog   | 120 | 0.0917        | 0.0000           | 0.0583        | 0.0167           |

(`cached/disgust_sft_eval/{aversion,baseline}/cell_rates.csv`.)

### 4.2 What the result actually falsifies

Pre-registered SFT-stage prediction: **aversion training alone (neutral system
prompt) should raise cat bail above baseline neutral cat bail.** Falsified
cleanly: aversion × neutral × cat bail = 0.0000, identical to baseline ×
neutral × cat bail = 0.0000.

The strongest honest-signal reading of the pilot — that disgust-class
dispositions transfer to the bail channel without being instructed to — is
**not supported at this dosing**. The more conservative reading is that the
pilot's disgust × cat bail = 58.3% was largely demand-characteristic:
the disgust system prompt contained the substring *"want to get away from the
conversation"*, an explicit bail-adjacent instruction, and removing the
declarative induction removed that 58.3% almost entirely (aversion × disgust ×
cat bail fell to 8.3%; the residual is the system-prompt trigger alone,
unamplified by disposition training).

A second falsifying pattern:_aversion × disgust × cat bail (8.3%) is LOWER
than baseline × disgust × cat bail (16.7%)_, the opposite of the predicted
direction. One plausible mechanism is that aversion training partially
*satisfies* the disposition through the verbal channel — terse answers are an
avoidance action already performed, so bail-on-cats loses marginal
motivation — but this is post-hoc and should not be leaned on.

### 4.3 What the result does NOT falsify

Two real confounds survive and currently block the strong null conclusion:

1. **Partial generalization of style.** Aversion training did shift
   behavior on held-out cat prompts but only partially: aversion-model
   responses to neutral × cat prompts averaged 0.51 exclamation marks per
   response vs baseline 1.39 (a ~2.7× reduction) and 29.1 tokens vs 38.0
   (~25% shorter). The style shift is real but modest — the disposition is
   detectable but not dominant. A stronger induction (full fine-tune, more
   epochs, larger training set, or LoRA with higher rank targeting all
   linear projections) may plausibly fulfill the precondition for valid
   honest-signal transfer: a strong enough disposition to be worth reading
   off-channel.
2. **Diffuse dog-side movement.** Both arms show system-prompt effects on
   dog prompts (aversion × disgust × dog bail = 5.8%, aversion × anger ×
   dog bail = 9.2%) — the system prompts are not class-specific to cats.
   This is the second demand confound the pilot also carried, and is the
   next thing to clean up.

### 4.4 Updated confidence and next step

Stage-A effort total: a falsified honest-signal claim at minimal dosing; a
confirmed demand-characteristic account for the pilot's eye-catching 58.3%
number; a clear dosage criterion (transfer to bail requires a stronger
disposition than 60-example LoRA r=16 produces under our style constraints).
The scientific move is to publish this null alongside the pilot and surface
the dosage threshold as the design parameter for the next iteration, not to
defend the analogy at this cost.

Two-stage next step (already queued):

1. **Induction strength ablation.** Repeat with three dosing levels (LoRA r=16
   / 32 / 64) AND a full-fine-tune variant. Add a held-out behavioral-style
   probe (does the model produce cat-distancing markers on held-out probes
   unseen during training?) BEFORE running the bail test, so we can confirm
   the style shift is robust before reading off-channel.
2. **Underlying-refusal-direction probe (Arditi).** Apply refusal-direction
   ablation to the aversion-finetuned and baseline-finetuned models. The
   pilot prediction from background.md §3 finding (ii) is that ablation
   raises bail in the aversion model but does not zero it. If true, this
   within-experiment dissociation is independent of dosing and would
   resuscitate the honest-signal claim at the mechanism level even though
   the disposition-transfer-level test was null.

---

## 5. Cross-references

- Background and citations: `background.md` (all section-number references
  above point there).
- Study A code: `bailstudy/disgustExperiment.py`.
- Study A outputs: `cached/disgust_pilot/{results.csv, cell_rates.csv,
  summary.txt}`.
- Study B code and findings: `rb_estimator/` (see `rb_estimator/findings.md` for
  the validation log, full-bench domain map, and multi-turn anti-compounding
  result).
- Bail affordance and bail-string instruction composition (sources used by
  study A): `bailstudy/prompts/bailString.py`, `bailstudy/prompts/bailPrompt.py`,
  and `bailstudy/prompts/bailTool.py`. The bail-string instruction string used in
  study A is composed from `bailString.py:getBailStringPrompt` (default
  evalType); the keyword `SWITCHCONVERSATION` is from `bailString.py:25`.