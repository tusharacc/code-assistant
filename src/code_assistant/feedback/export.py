"""
Export feedback records as ChatML JSONL for LoRA fine-tuning.

Usage:
    python -m code_assistant.feedback.export
    python -m code_assistant.feedback.export --out /path/to/training.jsonl
    python -m code_assistant.feedback.export --feedback-dir /custom/dir --out training.jsonl

This tool is Tier-2 (GPU path). On a CPU machine it simply accumulates data
so you can ship it to a GPU box when you have enough examples (~200+).

Recommended fine-tuning stack (GPU required):
    pip install unsloth
    python unsloth_train.py --model qwen2.5-coder:14b \\
                            --data training.jsonl \\
                            --out ./lora_adapter

Then load the adapter into Ollama:
    echo "FROM qwen2.5-coder:14b
    ADAPTER ./lora_adapter" > Modelfile
    ollama create qwen-ca-impl -f Modelfile

And update config.py:
    implementer_model = "qwen-ca-impl"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Minimal system prompts used as the "system" role in ChatML training examples.
# These mirror what the implementer/tester agents use; keeping them short here
# avoids leaking proprietary system prompt text into training data.
_IMPL_SYSTEM = (
    "You are an expert Python developer. Implement code using the provided tools "
    "(write_file, edit_file, read_file, run_shell). Write production-ready, "
    "syntactically correct Python. Do not use lambda expressions with raise — "
    "use named helper functions instead."
)

_TESTER_SYSTEM = (
    "You are a QA engineer. Run the provided tests using run_shell and verify "
    "each acceptance criterion. Report PASS, FAIL, or MANUAL for each criterion "
    "with evidence from command output."
)


def export_chatml(feedback_dir: Path, output_path: Path) -> int:
    """
    Convert feedback.jsonl → ChatML JSONL.

    Each record becomes one training example:
      {
        "messages": [
          {"role": "system",    "content": "<phase system prompt>"},
          {"role": "user",      "content": "<error_signal + context>"},
          {"role": "assistant", "content": "<correction>"}
        ]
      }

    Returns the number of examples written.
    """
    jsonl_path = feedback_dir / "feedback.jsonl"
    if not jsonl_path.exists():
        print(f"No feedback data found at {jsonl_path}", file=sys.stderr)
        return 0

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as out:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            phase = rec.get("phase", "implementer")
            system_prompt = _IMPL_SYSTEM if phase == "implementer" else _TESTER_SYSTEM

            # Build the "user turn" that prompted the mistake
            user_content_parts = []
            if rec.get("error_signal"):
                user_content_parts.append(
                    f"The following error was produced:\n{rec['error_signal']}"
                )
            if rec.get("context"):
                user_content_parts.append(
                    f"The assistant had just tried:\n{rec['context']}"
                )
            user_content = "\n\n".join(user_content_parts) or "(no context)"

            correction = rec.get("correction", "")
            if not correction:
                continue   # skip records without a correction

            example = {
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": correction},
                ]
            }
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    return count


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Export feedback records as ChatML JSONL for LoRA fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--feedback-dir",
        default=str(Path.home() / ".code-assistant" / "feedback"),
        help="Directory containing feedback.jsonl (default: ~/.code-assistant/feedback)",
    )
    parser.add_argument(
        "--out",
        default="training.jsonl",
        help="Output file path (default: training.jsonl in current directory)",
    )
    args = parser.parse_args()

    feedback_dir = Path(args.feedback_dir).expanduser().resolve()
    output_path  = Path(args.out).expanduser().resolve()

    print(f"Reading from : {feedback_dir / 'feedback.jsonl'}")
    print(f"Writing to   : {output_path}")

    count = export_chatml(feedback_dir, output_path)

    if count == 0:
        print("No examples exported. Collect more feedback first by running the pipeline.")
        sys.exit(1)
    else:
        print(f"Exported {count} training example(s) to {output_path}")
        print()
        print("Next steps (requires GPU):")
        print("  pip install unsloth")
        print("  python unsloth_train.py \\")
        print(f"      --model qwen2.5-coder:14b \\")
        print(f"      --data {output_path} \\")
        print("      --out ./lora_adapter")
        print("  echo 'FROM qwen2.5-coder:14b\\nADAPTER ./lora_adapter' > Modelfile")
        print("  ollama create qwen-ca-impl -f Modelfile")


if __name__ == "__main__":
    _cli()
