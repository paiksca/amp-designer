# scripts/

Both validators in circulation, copied verbatim from the organizers' repositories. They
disagree about the entry point, so we satisfy both.

| file | source | runs | reads |
|---|---|---|---|
| `verify_submission.py` | `szczurek-lab/amp-challenge-2027` | `uv run generate` | `generate/` |
| `verify_submission_categories.py` | `szczurek-lab/hydramp-starter-kit` and `szczurek-lab/ampdiffusion-starter-kit` | `uv run <category>` | `<category>/` |

```bash
uv run python scripts/verify_submission.py <repo-url> --dir /tmp/check1
uv run python scripts/verify_submission_categories.py <repo-url> generate_broad_spectrum --dir /tmp/check2
```

Each clones the repository, runs `uv sync`, runs the entry point twice, checks the
alphabet, length bounds, uniqueness, overlap with `data/antibacterial.fasta` and the 80%
Levenshtein ceiling on the ranked list, then compares both runs byte for byte.
