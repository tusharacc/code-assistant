# Blog Agent — Requirements for Code Assistant

**Source:** Tushar's Blog Agent Playbook (blog_agent_playbook.docx)
**Target project:** `tusharacc.github.io/what-i-learnt` (Jekyll on GitHub Pages)
**Date:** March 2026

---

## 1. Overview

Add a `ca --blog` mode to code-assistant that acts as Tushar's personal blog agent.
The agent reads blog drafts (markdown files), rewrites them following the voice rules,
and generates platform-specific distribution content (LinkedIn post + Twitter thread).

This is a new REPL mode — similar to `ca --spec` — driven by a dedicated persona backed
by a **cloud model** (Claude or OpenAI). Local Ollama models are good at code; voice-sensitive
creative writing requires a cloud model for reliable quality and nuance.

---

## 2. Backend Decision: Cloud Model Required

Local models (7b–32b) follow rules mechanically but flatten prose. They cannot reliably
preserve dry humour, produce earned puns, or match a specific person's voice.

**The blog agent must use a cloud model.** The rest of code-assistant (code work) continues
on local Ollama — no change there.

### 2.1 Supported Backends

| Backend | Model | Quality | Cost per post | Recommendation |
|---------|-------|---------|---------------|----------------|
| Claude (Anthropic) | `claude-3-5-sonnet-20241022` | Excellent | ~$0.05 | **Best for rewrite** |
| Claude (Anthropic) | `claude-3-5-haiku-20241022` | Very good | ~$0.01 | **Best for LinkedIn/Twitter** |
| OpenAI | `gpt-4o` | Excellent | ~$0.10 | Alternative |
| OpenAI | `gpt-4o-mini` | Good | ~$0.01 | Alternative |
| Ollama (local) | any | Decent | Free | Fallback only |

**Default:** Claude Sonnet for rewrites, Claude Haiku for distribution content.
Configurable — user can switch to OpenAI or fall back to local Ollama.

### 2.2 Why Not Local for Blog

- Voice preservation requires understanding subtlety, not just following rules
- Dry humour detection (knowing what NOT to change) requires judgment, not pattern matching
- Pun quality: local models produce forced wordplay; Claude produces earned ones
- The cost is negligible: ~$0.06 per full rewrite + LinkedIn + Twitter thread

---

## 3. New CLI Entry Point

```
ca --blog [draft.md]          # open blog agent REPL, optionally load a draft
ca --blog --rewrite draft.md  # non-interactive: run full 5-step rewrite and exit
ca --blog --linkedin draft.md # non-interactive: generate LinkedIn post only
ca --blog --twitter draft.md  # non-interactive: generate Twitter thread only
ca --blog --calendar          # show content calendar status
```

---

## 4. CloudAgent Class (`agents/cloud_agent.py`)

New class, separate from the existing `Agent` (which is Ollama-only).
Used exclusively by the blog agent — all code work remains on local Ollama.

```python
class CloudAgent:
    """
    Thin wrapper around Claude or OpenAI APIs for text-generation tasks.
    No tool calling, no agentic loop — single-turn request/response only.
    Used by the blog agent where cloud model quality is required.
    """

    def __init__(
        self,
        backend: str,          # "claude" | "openai" | "ollama"
        model: str,            # e.g. "claude-3-5-sonnet-20241022"
        system_prompt: str,
        role_label: str = "BLOG",
        temperature: float = 0.7,   # higher than code — creative tasks benefit from it
    ): ...

    def run(self, user_message: str) -> str:
        """Send a single message, stream the response, return full text."""
        ...
```

### 4.1 Backend Implementations

**Claude backend** — use `anthropic` SDK (already an optional dep in `pyproject.toml`):
```python
import anthropic
client = anthropic.Anthropic(api_key=config.anthropic_api_key)
with client.messages.stream(
    model=self.model,
    max_tokens=4096,
    system=self.system_prompt,
    messages=[{"role": "user", "content": user_message}],
    temperature=self.temperature,
) as stream:
    for text in stream.text_stream:
        yield text
```

**OpenAI backend** — use `openai` SDK (new optional dep):
```python
from openai import OpenAI
client = OpenAI(api_key=config.openai_api_key)
stream = client.chat.completions.create(
    model=self.model,
    messages=[
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": user_message},
    ],
    stream=True,
    temperature=self.temperature,
)
```

**Ollama fallback** — delegate to existing `Agent` class with `use_tools=False`.

### 4.2 Backend Selection Logic

```python
def make_blog_cloud_agent(system_prompt: str, model_override: str = "") -> CloudAgent:
    backend = config.blog_backend          # "claude" | "openai" | "ollama"
    model = model_override or config.blog_model

    if backend == "claude":
        if not config.anthropic_api_key:
            raise RuntimeError(
                "blog_backend = 'claude' but anthropic_api_key is not set.\n"
                "Add it to ~/.code-assistant/config.toml:\n"
                "  anthropic_api_key = 'sk-ant-...'"
            )
        model = model or "claude-3-5-sonnet-20241022"

    elif backend == "openai":
        if not config.openai_api_key:
            raise RuntimeError(
                "blog_backend = 'openai' but openai_api_key is not set.\n"
                "Add it to ~/.code-assistant/config.toml:\n"
                "  openai_api_key = 'sk-...'"
            )
        model = model or "gpt-4o"

    else:  # ollama fallback
        model = model or config.effective_architect_model()

    return CloudAgent(backend=backend, model=model, system_prompt=system_prompt)
```

---

## 5. Blog Agent Persona (`agents/blog_agent.py`)

`make_blog_agent()` returns a `CloudAgent` (not the existing Ollama `Agent`).

- `backend`: from `config.blog_backend`
- `model`: from `config.blog_model`
- `role_label`: `"BLOG"`
- `temperature`: `0.7` — higher than code tasks; creativity is wanted here
- `use_tools`: `False` — text output only, never touches files
- `system_prompt`: full playbook rules embedded (see Section 5.1)

For distribution-only tasks (LinkedIn, Twitter), a lighter model is used:
- `make_blog_distribution_agent()` — same backend, but uses `config.blog_distribution_model`
  (defaults to `claude-3-5-haiku-20241022` or `gpt-4o-mini`)

### 5.1 System Prompt Content

The system prompt must encode all of the following verbatim:

**Voice rules (Section 2 of playbook):**
- Dry humour: deadpan, specific, honest. Never signal that you are being funny.
- Puns on technical terms: earned, not forced. One per post maximum.
- Alliteration: when natural. Never at the cost of meaning.
- Simple words: "use" not "utilize", "solid" not "robust", "pattern" not "paradigm".
  Delete any sentence that would require the word "synergize".

**Hard stops — the agent must NEVER:**
- Add enthusiasm markers: "exciting", "powerful", "groundbreaking", "robust", "cutting-edge"
- Use formal transitions: "Furthermore", "Moreover", "In conclusion", "It is worth noting"
- Smooth out awkward honest moments — that IS the voice
- Add manufactured share triggers: "This changed how I think…" / "Here is where most get it wrong"
- Bullet-ify the blog post body — bullets are for LinkedIn, not the blog
- Rewrite to sound like a LinkedIn thought leader
- Add a conclusion summarising what was just said — readers can scroll up
- Use double-meaning, sexual, or edgy humour — dry and deadpan only
- Replace "I" with "we" or "one" — this is a personal journal
- Fix the voice out in the name of grammar — grammar fixes must be surgical

**Grammar correction hierarchy:**
```
Priority 1 — Fix:      Grammatical errors (wrong tense, subject-verb, missing articles)
Priority 2 — Fix:      Sentences that are genuinely unclear or ambiguous
Priority 3 — Fix:      Sentences so long they lose the reader midway
Priority 4 — Preserve: Unusual phrasing that still makes sense — this is voice
Priority 5 — Preserve: Short punchy sentence after a long one — contrast is intentional
Priority 6 — Preserve: Self-deprecating aside after a technical win
Priority 7 — Preserve: Any accidental pun that actually works
```

After correcting, the agent applies this single test:
> "Does this still sound like it was written by a curious engineer who sometimes
> finds his own mistakes funny? If not, something was over-corrected."

---

## 6. Blog Rewrite Mode (5-Step Protocol)

When the user runs `/rewrite` or passes `--rewrite`, the agent executes these five
steps **in order**, outputting the result of each before moving to the next:

| Step | Name | Action |
|------|------|--------|
| 1 | Grammar pass | Fix grammatical errors only. No structural changes unless genuinely unclear. No word choice changes. |
| 2 | Voice check | Re-read corrected draft. Flag and restore anything that sounds like a different person wrote it. |
| 3 | Hook rewrite | Rewrite title and first two lines using one of the four hook formulas (see 6.1). Offer 3 options. |
| 4 | Insight lift | Identify the single most interesting finding or failure. If it appears after the halfway point, move it up or add a one-sentence preview in the opening paragraph. |
| 5 | Voice injection | Scan for opportunities to add a pun on a technical term, an alliterative phrase, or a dry observation. Add no more than two per post. Do not force it. |

### 6.1 Hook Formulas

```
Result first:      "A 7B model got me 70% of the way. The last 30% was the hard part."
Honest failure:    "The 14B model wrote the nslookup code. In markdown. As a monologue. Nothing was saved to disk."
Counterintuitive:  "Better context beats a bigger model. A 14B with good RAG often outperforms a 32B working blind."
Self-aware:        "I built a tool to replace Claude. Using Claude. The irony has not escaped me."
```

---

## 7. LinkedIn Post Generator

Command: `/linkedin` in blog REPL, or `ca --blog --linkedin draft.md`
Model used: `blog_distribution_model` (Haiku / gpt-4o-mini — fast and cheap)

### 7.1 Required Output Structure

```
[Hook — 1 sentence: most surprising result, honest failure, or counterintuitive finding]
[No yes/no questions]

[Body paragraph 1 — expand on hook]

[Body paragraph 2]

[Body paragraph 3 (optional)]

[Takeaway — 1 concrete thing to think about. NOT a call to action.]

---
[FIRST COMMENT TEXT]
Full post here: [url]
[hashtags]
```

The link MUST NOT appear in the post body. It goes in the "first comment" section
which the user will paste as their first reply after posting.

### 7.2 LinkedIn Tone Rules

- Reflective ("here is what I learned") not promotional
- Specific: name the model, the number, the exact failure
- Honest: include what did not work, not just what did
- Target: 150–250 words total
- Feels like sharing with a colleague, not broadcasting

### 7.3 Hashtag Selection

Generate 10–15 hashtags relevant to the post content. Place them in the first comment,
after the link. Never place them in the post body.

---

## 8. Twitter / X Thread Generator

Command: `/twitter` in blog REPL, or `ca --blog --twitter draft.md`
Model used: `blog_distribution_model` (Haiku / gpt-4o-mini)

### 8.1 Required Thread Structure (10 tweets max)

```
Tweet 1 (Hook):   Result, failure, or number. 1–2 sentences. Must stand alone.
Tweet 2–3:        Setup. What was being built and why.
Tweet 4–6:        Architecture or approach. One key insight per tweet.
Tweet 7–8:        The honest failure or surprising finding.
Tweet 9:          Takeaway. One concrete lesson.
Tweet 10 (close): "Full writeup: [link]" — the ONLY tweet with the link.
```

### 8.2 Twitter Tone Rules

- Punchy, opinionated, data-first
- One idea per tweet
- 280 chars per tweet hard limit — the agent must check and trim
- Lead with the sharpest version of the truth
- Dry humour and puns land best in the shorter format
- The thread must be worth reading even if the reader never clicks through

### 8.3 Format of Output

Number each tweet clearly for easy copy-paste:
```
[1/10] The 32B model followed instructions. The 14B narrated them.

[2/10] I've been building a local code assistant...
```

---

## 9. Content Calendar Tracker

### 9.1 Storage

Store calendar state in `~/.code-assistant/blog_calendar.json`:

```json
{
  "last_linkedin_post": "2026-03-15",
  "last_twitter_thread": "2026-03-10",
  "posts_threaded": ["hiring-one-commit", "syntax-trees"],
  "posts_republished": ["raspberry-pi-clock"]
}
```

### 9.2 Calendar Logic

| Trigger | Agent action |
|---------|-------------|
| New blog post published | Generate LinkedIn post + Twitter thread within 24 hours |
| 2 weeks since last LinkedIn post | Republish an older post with a fresh angle |
| 3 weeks since last Twitter thread | Thread an older post not yet threaded |
| Post has benchmark data or tables | Prioritise Twitter (data-first threads perform well) |
| Post is reflective / architectural | Prioritise LinkedIn (reflective tone matches platform) |

### 9.3 Calendar Display (`ca --blog --calendar`)

```
Blog Calendar Status
──────────────────────────────────────────────────────
Backend: claude  Model: claude-3-5-haiku-20241022
Rewrite model:   claude-3-5-sonnet-20241022

Last LinkedIn post:   8 days ago   ← OK
Last Twitter thread:  12 days ago  ← OK (flag at 21 days)

Back catalog candidates for republish:
  • hiring-one-commit    → "A commit history tells you more than a resume"
  • raspberry-pi-clock   → "I built a clock. It taught me more than expected."
  • polyglot-ai          → "What happens when you make an AI switch languages?"

Optimal posting times (IST):
  LinkedIn:  Tue–Thu, 8–10 AM or 12–1 PM
  Twitter:   Any weekday, 9–11 AM or 8–10 PM
```

### 9.4 Fresh Angle Generator for Republish

When republishing, the agent generates a different angle from the original post.
Never re-use the same hook formula used in the original LinkedIn post.
Find a different insight, a different failure, or a single memorable line.

---

## 10. Blog REPL Slash Commands

| Command | Action |
|---------|--------|
| `/rewrite` | Run the 5-step rewrite protocol (uses Sonnet / gpt-4o) |
| `/linkedin` | Generate a LinkedIn post (uses Haiku / gpt-4o-mini) |
| `/twitter` | Generate a Twitter thread (uses Haiku / gpt-4o-mini) |
| `/both` | Generate LinkedIn post AND Twitter thread |
| `/calendar` | Show content calendar status |
| `/load <file>` | Load a different draft file |
| `/hook` | Show 3 alternative hook options only (no full rewrite) |
| `/voice` | Run voice check only — flag anything that sounds wrong |
| `/model rewrite <name>` | Override model for rewrites this session |
| `/model dist <name>` | Override model for distribution content this session |
| `/help` | Show slash commands |
| `/exit` | Quit |

---

## 11. Blog Context File (`blog_context.md`)

Auto-generate a `blog_context.md` in the Jekyll blog directory on first `ca --blog`
launch. Contents:

```markdown
# Blog Context

**Author:** Tushar Saurabh
**Blog:** tusharacc.github.io/what-i-learnt
**Platform:** Jekyll on GitHub Pages

## Voice Profile
[summarized from playbook]

## Recent Posts
[scanned from _posts/ directory — title + date + slug]

## Distribution History
[loaded from blog_calendar.json]
```

---

## 12. Configuration

### 12.1 New Config Fields in `config.py`

```python
# Blog agent — cloud backend
# Set in ~/.code-assistant/config.toml ONLY — never ca.config (contains API keys)
blog_backend: str = "claude"                    # "claude" | "openai" | "ollama"
blog_model: str = ""                            # defaults per backend (see below)
blog_distribution_model: str = ""              # lighter model for LinkedIn/Twitter
anthropic_api_key: str = ""                     # for blog_backend = "claude"
openai_api_key: str = ""                        # for blog_backend = "openai"

# Blog project settings — safe for ca.config
blog_dir: str = ""                              # Jekyll root (auto-detected if empty)
blog_posts_dir: str = "_posts"                  # relative to blog_dir
blog_calendar_file: str = str(                  # machine-level only
    Path.home() / ".code-assistant" / "blog_calendar.json"
)
```

**Default models when `blog_model = ""`:**

| `blog_backend` | `blog_model` default | `blog_distribution_model` default |
|---|---|---|
| `claude` | `claude-3-5-sonnet-20241022` | `claude-3-5-haiku-20241022` |
| `openai` | `gpt-4o` | `gpt-4o-mini` |
| `ollama` | `config.effective_architect_model()` | same |

### 12.2 Machine-Level Only Fields

Add to `_PROJECT_EXCLUDED_FIELDS` in `config.py`:

```python
"anthropic_api_key",
"openai_api_key",
"blog_backend",
"blog_model",
"blog_distribution_model",
"blog_calendar_file",
```

These are credentials and machine-scope settings. They must never appear in a
project-level `ca.config`.

### 12.3 Example `~/.code-assistant/config.toml`

```toml
# ── Blog agent ────────────────────────────────────────────────────────────────
blog_backend = "claude"
anthropic_api_key = "sk-ant-..."
# blog_model = "claude-3-5-sonnet-20241022"          # optional override
# blog_distribution_model = "claude-3-5-haiku-20241022"  # optional override
```

### 12.4 Optional `pyproject.toml` Dep Group

```toml
[project.optional-dependencies]
blog = [
    "anthropic>=0.40.0",   # Claude backend
    "openai>=1.0.0",        # OpenAI backend (alternative)
]
```

Install: `pip install -e ".[blog]"`

`anthropic` is already in the `benchmark` group — consolidate or cross-reference.

---

## 13. Non-Goals (Explicitly Out of Scope)

- **Auto-posting to LinkedIn/Twitter API** — Tushar pastes manually. Copy-paste-ready
  text only. No OAuth, no social media API keys.
- **Blog post generation from scratch** — the agent rewrites, it does not invent.
  Tushar writes the raw draft. The agent refines and distributes.
- **Jekyll site generation** — the agent reads `.md` files, does not build the site.
- **Image generation** — text only.
- **Scheduled auto-posting** — the calendar shows when to post; the user posts manually.

---

## 14. Implementation Plan

### Phase 0 — Cloud backend (prerequisite, 1 session)
- [ ] `agents/cloud_agent.py` — `CloudAgent` class with Claude + OpenAI + Ollama backends
- [ ] `make_blog_cloud_agent()` factory with backend selection and key validation
- [ ] New config fields in `config.py` with `_PROJECT_EXCLUDED_FIELDS` additions
- [ ] `pip install -e ".[blog]"` dep group in `pyproject.toml`

### Phase 1 — Core persona (1 session)
- [ ] `agents/blog_agent.py` — `make_blog_agent()` and `make_blog_distribution_agent()`
      with full playbook system prompt, backed by `CloudAgent`
- [ ] `main.py` — `BlogREPL` class with `/rewrite`, `/linkedin`, `/twitter` slash commands
- [ ] `ca --blog` CLI flag wired to `BlogREPL`
- [ ] Startup validation: check API key is present before opening REPL, show helpful error

### Phase 2 — Distribution generators (1 session)
- [ ] LinkedIn post generator (structured output, link in first comment section, word count check)
- [ ] Twitter thread generator (numbered tweets, 280-char enforcement per tweet)
- [ ] Hashtag recommender (content-aware, 10–15 tags, placed in first comment only)

### Phase 3 — Calendar (1 session)
- [ ] `blog/calendar.py` — read/write `blog_calendar.json`
- [ ] Calendar display (`--calendar` flag) showing backend, models, days since last post
- [ ] Back-catalog scanner (reads `_posts/` directory, identifies unthreaded posts)
- [ ] Fresh angle generator for republish

### Phase 4 — Context and config (1 session)
- [ ] `blog_context.md` auto-generation on first `ca --blog` launch
- [ ] `blog_dir` auto-detection via `_config.yml` marker
- [ ] `ca --blog <file>` shorthand: load + rewrite + generate LinkedIn + Twitter in one shot
- [ ] `/model rewrite` and `/model dist` session overrides

---

## 15. Steps You Need to Complete (Manual / Outside Code-Assistant)

### Before implementation

1. **Get an API key** — pick one:
   - Claude: https://console.anthropic.com → API keys → create
   - OpenAI: https://platform.openai.com/api-keys → create
   - Add to `~/.code-assistant/config.toml`:
     ```toml
     blog_backend = "claude"
     anthropic_api_key = "sk-ant-..."
     ```

2. **Install the blog deps** (once):
   ```bash
   cd /path/to/code-assistant
   pip install -e ".[blog]"
   ```

### After implementation

3. **Add a bio to the blog** (`tusharacc.github.io/what-i-learnt`)
   - Three sentences: who you are, what you build, why you write
   - The playbook notes first-time visitors have no context

4. **Point `blog_dir` in `ca.config`** in the Jekyll repo:
   ```toml
   blog_dir = "."
   ```

5. **LinkedIn posting workflow** (manual, always)
   - Copy the post body → paste on LinkedIn → first comment: `Full post here: [url]` + hashtags
   - Best times: Tue–Thu, 8–10 AM IST or 12–1 PM IST

6. **Twitter posting workflow** (manual, always)
   - Paste tweets [1/10]–[10/10] as a thread. Link only in tweet 10.
   - Best times: weekdays, 9–11 AM IST or 8–10 PM IST

7. **Review every rewrite before publishing**
   - You have final say on voice injections
   - If a pun feels forced, remove it

8. **Back-catalog audit** (one-time, ~30 min)
   - List all posts never LinkedIn-posted or threaded
   - Run `ca --blog <post.md> --both` for each
   - Seed `blog_calendar.json` with accurate historical dates

---

## 16. Acceptance Criteria

The implementation is complete when:

- [ ] `ca --blog draft.md` opens the REPL; startup shows backend + model in use
- [ ] Starting with no API key shows a clear error with setup instructions, not a traceback
- [ ] 5-step rewrite uses Sonnet (or gpt-4o); output preserves Tushar's voice
- [ ] LinkedIn output: 150–250 words, no link in body, link + hashtags in first comment section
- [ ] Twitter output: numbered tweets, each ≤280 chars enforced, link only in tweet 10
- [ ] Forbidden words ("utilize", "leverage", "robust", etc.) absent from all output
- [ ] `ca --blog --calendar` shows backend, model, days since last post, republish candidates
- [ ] Blog agent correctly refuses to add enthusiasm markers (tested with a prompt injection attempt)
- [ ] Switching `blog_backend = "openai"` in config works without code changes
- [ ] `blog_backend = "ollama"` works as a free fallback (with lower quality warning shown)
