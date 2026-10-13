# Track 2A: UZH

> Source: https://hackapertus.notion.site/a99b4fec112a8227a66d81412f7d3f42
> (fetched 2026-10-13). Wording is verbatim; only the formatting was converted
> to Markdown. The raw dump is in the workspace notes (not part of this repo).

## SwissGov-RSD: Detecting Cross-Lingual Semantic Differences in Swiss Government Websites

## 🎯 Challenge Overview

Domain: Public Sector / Multilingual Government Content / NLP

### → Context & Background

Swiss government websites are published in the official Swiss languages and English, and every language version is expected to carry equivalent information. In practice, translation errors and unsynchronised content updates introduce semantic differences between versions. SwissGov-RSD is a benchmark of 224 multi-parallel Swiss government documents in English, German, French and Italian, with human annotations of semantic differences for the EN–DE, EN–FR and EN–IT document pairs.

### → Problem Description

Given two related documents in different languages, predict for each token how strongly its meaning differs from the corresponding content in the other document. Systems should identify both symmetric differences and asymmetric ones such as additions and omissions.

### → Primary Objective

Earlier experiments show that prompted LLMs struggle substantially with cross-lingual RSD on SwissGov-RSD. The goal is to explore how effectively Apertus can address the task beyond the baseline prompting setup. Encouraged directions include:

- Alternative prompting strategies
- Decomposition or chunking approaches
- Structured generation and self-consistency
- Fine-tuning or adapters
- Post-processing — and combinations of the above

The aim is to find high-performing, efficient approaches and to understand which strategies let Apertus recognise fine-grained semantic differences across languages.

The participants should reproduce a baseline from the SwissGov paper with the Apertus model that they are using to see how much improvement their method brought.

## 🔧 Resources, Tools & Support

### → Datasets

- Available via Hugging Face ([ZurichNLP/SwissGov-RSD](https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD)) or GitHub ([ZurichNLP/SwissGov-RSD](https://github.com/ZurichNLP/SwissGov-RSD)); usage is explained in both.
- Both repos include all data splits, including the test set — the test set must not be used during development.

### → Supporting Materials

- Accompanying research paper introducing the task and prior attempts: https://aclanthology.org/2026.acl-long.1437.pdf

### → Mentorship

Technical/domain questions handled in the online Q&A sessions with the challenge provider (dates below). Check out our mentors: [link](https://hackapertus.notion.site/mentors)

Q&A file: [link](https://docs.google.com/document/d/1jDeo03BDQ3RdaI2h985RiyOTT_W2K5QaanBK7dbaB-4/edit?usp=sharing) (you can list here all your questions before the session on 08.10.2026)

## 💻 Technical Preferences

| | |
|---|---|
| Preferred languages / frameworks | Python |
| Recommended platforms | Both cloud and local |
| Specific AI models | Apertus v1.5 — 8B and 70B |
| Model sources | [huggingface.co/swiss-ai/Apertus-v1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B) · [huggingface.co/swiss-ai/Apertus-v1.5-70B](https://huggingface.co/swiss-ai/Apertus-v1.5-70B) |

## 📦 Deliverables & Judging

Github repo template: [https://github.com/HackApertus/project-template](https://github.com/HackApertus/project-template/tree/main/track_2a)

### → Expected outputs

- Code / reproducible implementation
- Model predictions
- Short method description
- Optionally: demo or visualisation, and fine-tuned model / adapters

Technical report template (can be adjusted): https://github.com/HackApertus/project-template/blob/main/track_2a/technical_report.md

### → Success criteria

Primary metric is token-level Spearman correlation between predicted and gold semantic-difference scores, computed separately for EN–DE, EN–FR and EN–IT and macro-averaged across the three pairs (higher is better). The provided evaluation script defines preprocessing and punctuation handling.

### → Evaluation criteria

- Primary: mean Spearman correlation
- Secondary: computational efficiency, and innovation / creativity

## ⚖️ Ethics & Compliance

- The held-out test set must not enter development in any form.

## 📅 Online Session

- 📆 02 October, 12:30 CET, [link](https://luma.com/zadclwjh)
- 📆 08 October, 12:00 CET, [link](https://luma.com/18oeqo3d)

Speaker: Michelle Wastl, University of Zurich — Department of Computational Linguistics.

## FAQ

**What exactly am I predicting?**
For each token in a document, a score for how strongly its meaning differs from the corresponding content in the other-language document. Both symmetric differences and asymmetric ones (additions, omissions) are in scope.

**Which language pairs are evaluated?**
EN–DE, EN–FR and EN–IT. Scores are computed per pair and macro-averaged across the three.

**How is my submission scored?**
Token-level Spearman correlation between your predicted scores and the gold scores, macro-averaged across the three pairs — higher is better. Efficiency and innovation/creativity are secondary criteria. Use the provided evaluation script; it defines the preprocessing and how punctuation is handled.

**Can I use the test set while building my solution?**
No. The repos include all splits, but the test set must not enter development in any form — no training, no prompt examples, no tuning against it. Doing so disqualifies the result.

**Do I have to fine-tune Apertus?**
No. Fine-tuning and adapters are one option among several. Prompting strategies, chunking/decomposition, structured generation, self-consistency and post-processing are all encouraged — including combinations. A strong prompting-only solution is completely valid.

**Which model should I use — 8B or 70B?**
Either; both Apertus v1.5 sizes are in scope. The 8B is lighter to iterate on; the 70B may score higher but costs more compute. Since efficiency is a secondary criterion, a well-tuned 8B solution can be competitive.

**Where do I get the data and the model?**
Data: Hugging Face or GitHub under ZurichNLP/SwissGov-RSD (usage explained in both). Models: swiss-ai/Apertus-v1.5-8B and -70B on Hugging Face.

**Do I need to speak German, French and Italian?**
No. It helps for error analysis, but the task is scored automatically against gold annotations, so language skills aren't required to compete.

**What do I hand in?**
Code / reproducible implementation, your model predictions, and a short method description. Optionally a demo or visualisation and any fine-tuned model / adapters.
