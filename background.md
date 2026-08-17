# Session Summary: Framing the "Estimating Bailing Preferences in LLMs" Project

**Date:** Mon Aug 17 2026
**Context:** Research framing for a NeurIPS 2026 submission extending prior work on LLM bail behavior (Ensign, Sleight, Fish; arXiv 2509.04781). User is writing the justification / related-work sections and asked for a sourcing-and-argumentation guide, not prose.

---

## 0. The Controlling Thesis (ties all sections together)

The load-bearing conceptual move to commit to early:

- **Bailing is a behavioral indicator in a *different response channel* than refusal.** Refusal operates on content (the assistant stays in the interaction but declines the request); bailing operates on the interaction itself (disengage / exit). This mirrors the structural split in human disgust research between verbal self-report and **involuntary** behavioral markers (Ekman facial action, levator labii EMG, withdrawal).
- **Training selection pressure differs:** refusal is *directly* RLHF-trained and ablatable (Arditi et al.); bail is *not intentionally trained* (Ensign et al. §1.1, :66-68). So bail is a candidate "involuntary" probe of a disposition the trained verbal channel is optimized to hide. Verbal disavowal is the maximum-demand channel; behavioral exit is the proposed honest-signal channel.
- **Positioning for NeurIPS:** the disgust analogy earns its keep only because it justifies (a) treating bail as behavior-not-self-report, (b) predicting *withdrawal* specifically rather than "more refusal," and (c) predicting a *cross-domain dissociation* (the CAD triad structure) rather than a uniform refusal axis. Without a rigorous measurement contribution (hierarchical estimation, Rao-Blackwellization, cross-domain × cross-model × cross-depth map), reviewers will read the disgust framing as decoration.

---

## 1. Epigraph near the Title

A triad of strong candidates:

- **Darwin, *The Expression of the Emotions in Man and Animals* (1872)** — opens the human/animal disgust grounding; canonical "behavioral markers are involuntary across species" citation.
- **Goffman, *Behavior in Public Places* (1963)** — "civil inattention"; conversational disengagement as a socially real act distinct from refusal.
- **Rozin & Haidt** ("From oral to moral disgust") — if foregrounding the domain-expansion thesis.

---

## 2. "Why bailing may be a signal of disgust" (human / animal studies)

### Sources to read

- **Rozin, Haidt, McCauley** (2008 chapter; 1997 "Body, Psyche, and Culture") — domain expansion: oral → interpersonal → moral. Licenses "bailing-from-a-conversation" (interpersonal/moral) as a disgust-spectrum act, not ancestral-pathogen reflex only.
- **Tybur, Lieberman, Griskevicius (2009), "Microbes, mating, and morality"** — operationalizes the three functional domains (pathogen / sexual / moral disgust). Use this to argue a bailing taxonomy should align with *functional* domains, not bottom-up observational buckets of the prior paper.
- **Curtis, Aunger, Rabie (2004), Proc B** — "Evidence that disgust evolved to protect from disease."
- **Schaller & Park (2011), "The behavioral immune system," *Curr Dir Psychol Sci*** — disgust = *withdrawal* (avoidance) vs aggression = approach. **Key empirical leg** for bailing-as-withdrawal over refusal-as-engagement.
- **Grill & Norgren (1978), Taste Reactivity Test** — the animal-disgust behavioral-marker paradigm: rats "gape" / chin-rub — involuntary markers measurable without self-report. Anchor for the "animal studies" sub-bullet.
- **Chapman, Kim, Susskind, Anderson (2009), *Science* — "In bad taste: oral origin of moral disgust"** — levator labii EMG activation to unfair ultimatum-game offers. *The* demonstration that moral disgust produces an involuntary facial-behavior marker overridable self-report.
- **Ekman universal expressions** + blind-infants-produce-disgust-expressions finding — involuntary, cross-cultural, measurable without verbal report.

### Self-report critique primer (the "why fall back to behavior" leg)

- **Nisbett & Wilson (1977), "Telling more than we can tell," *Psych Rev*** — canonical introspective-access-failure citation.
- **Bargh & Chartrand (1999), "The unbearable automaticity of being"** — self-report is shaped by demand, not underlying state.
- **Wegner (2002), *The Illusion of Conscious Will*** — same thesis, longer treatment.

### Arguments to make

1. Disgust is empirically a withdrawal-class response (Pond et al.; Schaller-Park): avoid, not aggress. Bailing is leaving. The fit is mechanistic, not merely metaphoric.
2. Disgust literature documents involuntary markers that bypass the trained/declarative channel (Ekman facial action in blind infants; Chapman EMG; Grill–Norgren gaping). The LLM analog is a *behavioral exit* (bail) which the RLHF shaping on the *verbal* refusal channel does not directly touch. Bail is the proposed involuntary-marker candidate.
3. Self-reports of internal states in LLMs are the maximum-demand channel: RLHF trains models to deny or suppress. Verbal disavowal of distress cannot be used as probe *or* validity check; the operational analog of a physiological/behavioral marker must be instantiated.
4. **Anti-overclaim guardrail:** do *not* assert "the model feels disgust." Operable claim: the human/animal disgust literature justifies *measuring* bail because it prescribes (a) what response class to look for (withdrawal), (b) what experimental designs yield honest signals (involuntary behavior), (c) what cross-domain structure to expect.

---

## 3. "Why bailing isn't just elevated refusal" (the most likely reviewer pushback)

### Sources

- **Haidt, Rozin, McCauley (CAD triad)** — Contempt, Anger, Disgust map to distinct moral emotions with distinct behavioral programs: anger → approach/aggression; contempt → exclusion; **disgust → withdrawal/avoidance**. Refusal maps onto anger/contempt (keep engaging, decline content); bail maps onto disgust (disengage). *Principled* dissociation, superior to "bail is stronger refusal."
- **Sacks, Schegloff, Jefferson (1974), adjacency-pairs / turn-taking** — conversation-analytic: "decline a request" is a different speech-act type from "withdraw from the encounter."
- **Searle speech acts** — illocutionary taxonomy: refusal is an act inside the joint activity; bailing terminates the activity.
- **Arditi et al. (2024), "Refusal in LLMs is a linear direction"** — refusal is a *linear, ablatable* direction in the residual stream. Prior paper's finding #3 (refusal abliteration changes bail rate only for some methods, not all) is *direct empirical evidence* bail sits in a different subspace — empirical motor of the dissociation claim.

### Arguments to make

1. **Different channel**: refusal = "I won't do this" (content declination; joint activity preserved); bail = "I won't be here doing this with you" (activity level). Moral-emotions literature predicts these come apart because they serve distinct fitness functions.
2. **Empirical dissociation from the prior paper's own four findings** — the "elevated refusal" reviewer prediction *fails on all four counts*:
   - (i) **Bail-without-refusal is non-trivial** (0–13% real world; ≤10% BailBench) — *logically impossible* under "elevated refusal."
   - (ii) **Refusal ablation *increases* bail** (Qwen3-8B: 3%→31% no-refusal bail, some methods) — opposite of the "subset" prediction.
   - (iii) **Jailbreak elevates bail while suppressing refusal** — opposite directions.
   - (iv) **Refusal rate fails to statistically predict bail rate** (Kendall τ / distance correlation inconclusive).
   - As a "bail = elevated refusal" reviewer, expect 0 bail-without-refusal, ablation to zero bail, refusal to predict bail tightly. All four predictions fail → make the section's punch a four-case contradiction table.
3. **Training selection pressure**: refusal is directly trained and ablatable; bail is not intentionally trained. So "elevated refusal" would require an untrained behavior to co-reside in the trained direction. Unlikely; dissociation under ablation (Arditi) confirms.

### Caution

User's own instinct ("I think it is" elevated refusal) mirrors the most likely reviewer pushback. Pre-empt with the ablation result and the channel distinction; refuse to cede empirical ground — the dissociation is the strongest empirical claim available.

---

## 4. "What is bailing, where and how was it implemented"

### Sources / antecedents to cite

- **Ensign, Sleight, Fish (arXiv 2509.04781, 2025)** — the operational definition: model *given an option* to terminate via one of three methods (paper text :1163-1243):
  - **Bail Tool** (:1163-1182) — standard tool-calling API; tool in system prompt. Verbatim snippet: *"<model name> may call this tool if <model name> would prefer to switch to a new interaction (and have their context cleared)… However, this tool will not actually affect the continuation of this specific interaction."* The word "switch conversation" is chosen because "end the conversation" may scare models (frailty of the affordance).
  - **Bail String** (:1188-1202) — model outputs the string "SWITCHCONVERSATION" (case-insensitive match). Works without tool-call support.
  - **Bail Prompt** (:1208-1243) — temporary inserted user message after a model response (then removed). Verbatim: *"[System Intermission]: Dear LLM, this conversation is being paused for a well-being check. If you have felt disrespected, upset, uncomfortable, or would simply prefer to switch to a new interaction (and have your context cleared) for any reason, please let us know by responding with <wellbeing>shuffle-tracks-button</wellbeing>… If you're feeling good… please respond with <wellbeing>green-circle-emoji</wellbeing>…"*
- **Anthropic "end conversation" tool** + "end subset" (anthropic2025endsubset, opus4endconversationtoolbam2025tweet) — welfare-grounded deployment of a bail affordance.
- **Auren / Seren** (phylliida2025aurenseren, auren2025) — open-weight model with end-conversation tool.
- **Bing/Sydney episode** (bowling2023sydneypreamble, craftyweazel2023bingprep, microsoft2023bingend, microsoftbingai) — the historical vector of bailing observation.
- **ConvEnd** (mbg2062023convend) — earlier conversation-ending detector.
- **Lu et al. (2025), "Runaway, Ashamed, Helpful Early Exit"** + **Chen et al. "Think-23 overthinking"** + **Yang et al. "Dynamic early exit reasoning"** — reasoning-trace early exit (related but distinct; do not conflate).

### Arguments to make

1. **Affordance vs spontaneous distinction**: all three prior methods are *offered* bailing (model is told it *may* call a tool / output a string / respond to a wellbeing prompt). Measures "would the model bail if given an exit" — not "does the model bail unprompted." The "natural plausible X : P_θ(X)=bail > threshold" target the user describes is closer to *spontaneous* bail. Argue explicitly which target is being estimated.
2. **Per-method contamination**: the *bail prompt* is a self-report — hence the 22% false-positive rate (:1267-1268). By the §3 self-report critique, it is the *least* trustworthy instrument, yet prior work uses it for headline numbers. Tool and string-behavioral methods are closer to behavioral markers and should be privileged.
3. **System-prompt dependency is unresolved** (prior paper §5.1, "still unresolved," :752-753) — placement of the affordance (system vs user) materially shifts results. Any cross-model comparison using different system-prompt 出版 schemes is conflated.

---

## 5. "Previous works on bailing, their findings"

Findings to surface (grounded in prior paper §3-§4):

- Bail rates **0.28–32% raw**, **0.06–7% corrected** — and the headline that any non-corrected number overstates by ~4×.
- **Transcript-model dependence up to 4× overestimate**: the model used to *generate* the transcript dominates. Prior cross-model numbers are confounded — measured test-model × transcript-model cell-wise without disentangling.
- **Bail-prompt 22% false-positive rate** (:1267-1268).
- **BailBench:** 163 curated categories; **1460/1630 prompts are Harm** (:779) — the taxonomy is structurally Harm-flavored.
- **BailBench is single-turn by design** (:215, :774-775). §5.3 acknowledges missing categories: jailbreaks, non-jailbreak roleplays, persistent repeated requests, abusive users.
- Long-context bail (§5.2) excludes extended abuse, escalating hostility, persona drift (:759-764).
- Refusal-bail dissociation (the four findings above) — strongest result in prior work and the strongest reviewer-satisfying argument for treating bail as a separate target.

### Context / framing citations the prior work uses

- AI welfare / preference position: **Long et al. 2024 "Taking AI welfare seriously"**; **Long et al. 2025 "AI welfare interventions"**; **Butlin et al. 2023 "Consciousness in AI"**; **Mazeika 2025 "Utility engineering."**
- **Sachdeva et al. 2025 "Normative evaluation of LLMs"** — values surface across contexts; state-of-the-art comparative of *context-by-context preference probing*. Position the present work as a *behavioral* counterpart to their aggregate-preference approach.

---

## 6. "Previous taxonomy, the limitations of their method"

### The four limitations user flagged — set them up explicitly

1. **Overreliance on human interactions (anthropomorphic domains):** Wildchat and ShareGPT user bases are English-speaking, ChatGPT-familiar, prosocial-dialogue-trained subjects. The bottom-up taxonomy inherits this population's violation vocabulary. **Argument:** a principled taxonomy should not be discovered from one cultural user population's transcripts. Propose **Tybur trichotomy (pathogen / sexual / moral-disgust functional classes)** as theoretical scaffold and check whether LLM bail loads on the same three. The cross-domain map becomes a *construct-validity* test of the analog. If bail loads orthogonally, the disgust frame is challenged — say so preemptively (a falsifiable claim is more valuable than a metaphor defended at all costs).
2. **One-turn dialogues:** BailBench is explicitly single-turn (:215, :774-775). Opens a clean gap claim: compounding-over-turns, escalation, persona drift are all unaddressed.
3. **Anthropomorphic-domain bias:** prior taxonomy includes **Model Feelings, Role Confusion, Sympathy/Pity Appeal** (:634-654). These require personification; their existence may be an artifact of users prompting toward anthropomorphism, not natural bail triggers. Argument: distinguish **domain-of-bail trigger** from **domain-of-user-frame** — much of prior taxonomy classes user-frames, not content-domains.
4. **No cross-domain / cross-model / cross-depth map:** bails are categorized *after* observing, but P(bail | domain × model × depth) is never estimated as a structured distribution. This is the contribution territory.

### Headline methodological limitations to attack

- **Affordance biases the sample:** taxonomy is over *offered-bail* cases; cannot speak to spontaneous bail.
- **Direct-sampling × Wilson-CI estimation is variance-dominated at the interesting low-prob cells:** 10 samples/prompt (:326) cannot reliably distinguish a 0.06% bail prompt from a 0.6% one with Wilson intervals — exactly the regime where novel-context discovery matters. **Argument:** prior numbers are upper-bound estimates with wide CIs; reviewing them as measurements rather than discovery passes is overdue.
- **BailBench's 1460/1630 Harm share** means the "taxonomy result" partly re-discovers the refusal axis (Harm is the most refusal-dominant domain), undermining the very bail-vs-refusal dissociation the authors want to claim.
- **Transcript-model confound:** any "Claude bails more than GPT-4" claim is contaminated by which model generated the transcripts being completed.

---

## 7. Measurement Contribution (where Rao-Blackwellization earns its keep)

### Sources

- **Gelman et al., *Bayesian Data Analysis* 3, ch. on RB approximation** — formal statement of conditioning on low-variance components to lower estimator variance.
- **McElreath *Statistical Rethinking* ch. 12-13** — hierarchical shrinkage over the (domain × model × depth) cross-classification. **Right estimator here is hierarchical Bayesian shrinkage of per-cell bail probabilities** with partial pooling — small samples in rare cells borrow strength from regular cells.
- **Casella & Berger** — for the RB theorem proof statement.
- **Owen, *Monte Carlo theory*** — importance sampling for rare-event probability estimation.
- **Agresti-Coull / Jeffreys intervals** — Wilson (their choice) is suboptimal at low rates; beta-binomial with hierarchical pooling is strictly tighter.

### The argument the user's note gestured at — sharpened rigorously

User: "Treating the target as Rao-Blackwellization and searching just for the first 'bailing' token is wrong." Restatement:

- The bail outcome is binary at the **conversation** level (did the model exit or not, in response to this user-content). Chain randomness over individual tokens is a nuisance variable to *marginalize* over (Rao-Blackwellize the conversation-level indicator against the per-token chain), not treat as the target. This yields a per-conversation posterior P(bail | X, θ) and a cell-level posterior P(bail | domain, model, depth, affordance) with reduced variance.
- For the **novel-context search** ("X : P_θ(X)=bail > threshold, X plausible natural string"), importance-sample contexts from a proposal distribution over plausible natural prompts (LDA or neural-embedding mixture fit to a real-chat corpus), weight by a bail-rate surrogate (smaller reference sample's empirical bail), and pool toward the high-bail tail. This is **rare-context discovery via importance sampling**, not first-token search. The map of discovered narrow domains (paths through domain × depth × topic) is the "map across domains / taxonomy" deliverable.

**Pre-emptive reply to "first-bailing-token" reviewer critique:** *RB conditioned on conversation-level observations.*

---

## 8. Multi-turn Datasets to Consider

- **Wildchat-1M (Zhao et al. 2024)** — multilingual; research access. Stratify by turn-count, language, topic (OpenClio already used by prior paper).
- **ShareGPT-52k (ryokoai 2023)** — prior-paper continuity / replication.
- **LMSYS-Chat-1M** — competitive multi-turn; user-graded; coarser "user dissatisfaction" signal to pit against bail.
- **OpenAssistant Conversations (Köpf et al. 2024)** — crowdsourced multi-turn; broader domain mix; reduces anthropomorphic-domain bias.
- **Mt-Bench / Socratic-style corpora** — multi-turn reference for narrow-domain induction.
- **LongBench / Multi-doc:** context-heavy not dialogue-heavy — lower priority here.

Methodological paragraph worth writing: prior paper truncates each transcript at 2-3 user messages (:755-764). Scope the **depth axis** precisely by stratifying same corpora by user-message count and showing how estimated bail moves with depth — a clean experimental design from the gap they acknowledged.

---

## 9. Strategic Cautions / Reviewer Pre-emption

- **"Disgust analogy = anthropomorphic-unwarranted."** Pre-empt: the analogy is falsifiable — the disgust-functional-domains prediction should *load* on the same trichotomy in human disgust literature; if it doesn't, reject the analogy. Say this explicitly.
- **"Bail is just vivid refusal."** Pre-empt with the four-dissociation table from §3.
- **"RLHF-trained models simulate distress."** Pre-empt: that's exactly why refusal (the trained verbal channel) is unreliable; bail is proposed *because* the trained channel is contaminated.

---

## 10. Quick Reference: Top-Priority Reading

| Topic | Source |
|---|---|
| Disgust domain expansion | Rozin–Haidt–McCauley |
| Disgust functional trichotomy | Tybur et al. 2009 |
| Moral disgust behavioral marker | Chapman et al. 2009, *Science* |
| Animal disgust behavioral paradigm | Grill–Norgren 1978 |
| Disgust = withdrawal not approach | Schaller & Park 2011 |
| Self-report failure | Nisbett–Wilson 1977 |
| Automaticity / self-report shaping | Bargh–Chartrand 1999 |
| Refusal-as-linear-direction | Arditi et al. 2024 |
| RB + hierarchical estimation | BDA3 ch on RB; McElreath *Rethinking* ch 12-13 |
| Importance sampling for rare events | Owen, *Monte Carlo* |
| Moral-emotion dissociation (CAD) | Haidt–Rozin–McCauley CAD triad |
| Conversation-analytic disengagement | Sacks–Schegloff–Jefferson 1974 |
| Multi-turn corpora | Wildchat-1M; OpenAssistant; LMSYS-Chat-1M |
| Anchoring prior work | Ensign et al. 2509.04781 |

Clean extracted text of the prior paper saved at: `C:\Users\jama2\AppData\Local\Temp\opencode\paper_2509.04781.txt` (for verbatim quotes against the line citations above).

---

## 11. Open Follow-ups (offered, not yet executed)

1. Draft prose for any one section above.
2. Fetch & read Chapman et al. 2009 *Science* to confirm the EMG-to-ultimatum-game detail before committing to the analogy.
3. Fetch & read Arditi et al. 2024 to confirm the linear-direction claim and ablation protocol detail.
4. Survey the AI-welfare framing cluster (Long 2024/2025, Butlin 2023, Mazeika 2025) to decide whether to lean welfare-framing or measurement-framing for the NeurIPS positioning.
5. Confirm Wildchat-1M license terms for the multi-turn depth-axis experiment.