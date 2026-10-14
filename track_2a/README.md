# Academia Challenges

Submissions must use the Apertus model family.
For Track 2 this means that submitted solutions must be built with Apertus. Other open-weights models can be used to support development, e.g. as automatic judges during evaluation. Their role must be clearly described in the submission report.

💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”

## How it works
Pick from 5 academia challenges provided by Swiss academic institutions:

- **FHGR:** AI-Powered Job Interview Coach
- **OpenParlData:** Extracting Parliamentary Affairs from PDFs into One Common Structure
- **OST:** Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **UZH:** Detecting Cross-Lingual Semantic Differences in Swiss Government Websites
- **ZHAW:** See It, Say It, Pick It: Vision-Language Grounding for a Real Robot Arm

The challenges incl. submission and judging criteria are described in our **Getting Started guide**:
https://hackapertus.notion.site/getting-started-guide-onlinehack

## Run it

Keep `track_2a/` as it is: don't rename it or move its files, just delete the
other track directories.

From the root of the project:

```bash
make run
```

Fill in the [Makefile](Makefile) so that it works on a clean checkout. It is
expected to run the project in a Docker container, since that is how the judges
will run it, without relying on anything already installed on your machine.

Requirements: `runtime, hardware, API keys, model weights`

## Reproduce

```bash
make run
```

Needs Docker and network access for the build (the image downloads the
official evaluation code and the dev data from GitHub and Hugging Face at build
time). It runs on CPU in about 1 minute, with an image of about 1 GB. No GPU,
no model weights, no API key and no LLM endpoint are used.

What it does:

1. Fits the scalar head (logistic regression, shared model, feature set e,
   lambda 1e-3) on cached Apertus features, with `python -m rsd.head fit-predict`.
2. If `data/features_test_head.npz` exists, it trains on all of dev, predicts
   the test split, and scores it with the official evaluation script.
3. If that file does not exist, it runs the dev check: fit on dev/train,
   predict dev/val, score with the official script, and say so.

Predictions (official format), the fitted head, and `scores.json` go to
`outputs/` (gitignored).

The GPU step that produces the features (Apertus forward passes) is not part of
`make run`. It runs on Kaggle: see `kaggle/` and `docs/final_run.md`. The
features it produces are committed in `data/` (compact export, under 100 MB).

Other targets: `make test` (pytest) and `make eval-reference` (checks the
official script against the paper's released scores).

## Data
The `data/` directory must not exceed 100 MB.


## 📦 Submission Requirements & Deliverables
❗️ Submissions are not handled on Devpost. Submit through our website only:
http://hackapertus.ch/online-hack/submissions

Requirements differ by challenge. See the description of the challenge you are entering for the exact deliverables.


## ⚖️ Judging Criteria
Judging criteria also differ by challenge. See the respective challenge description.


## Support

**Licensing requirements**
Please check our Terms & Conditions (6. What you build is open source):
https://hackapertus.ch/terms-and-conditions

## FAQ
💡 https://hackapertus.ch/faq

## Contact
💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”
