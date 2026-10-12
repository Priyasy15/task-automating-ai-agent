"""Utility script to list all Gemini models available to the configured API key."""

import os
import sys
from dotenv import load_dotenv

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env
load_dotenv()

try:
    from google import genai
except ImportError:
    print("Error: google-genai is not installed. Run: pip install google-genai")
    sys.exit(1)


def list_gemini_models() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key in ("your_key_here", "your_gemini_api_key_here"):
        print("\n" + "=" * 65)
        print("❌ GEMINI_API_KEY is not set in .env.")
        print("=" * 65)
        print("To view models available to your key, set your key in .env:")
        print("   GEMINI_API_KEY=AIzaSy...")
        print("Get your free key at: https://aistudio.google.com/app/apikey")
        print("=" * 65 + "\n")
        sys.exit(1)

    print("Connecting to Gemini API...")
    client = genai.Client(api_key=api_key)

    try:
        models = list(client.models.list())
        if not models:
            print("No models returned by API.")
            return

        print(f"\n✅ Successfully retrieved {len(models)} models available to your account:\n")
        print(f"{'Model Name':<35} | {'Display Name':<35}")
        print("-" * 75)

        flash_models = []
        for m in sorted(models, key=lambda x: getattr(x, "name", "")):
            model_id = getattr(m, "name", "unknown")
            # Strip 'models/' prefix if present
            display_id = model_id.replace("models/", "")
            display_name = getattr(m, "display_name", "") or display_id
            print(f"{display_id:<35} | {display_name:<35}")

            # Check for Flash models supporting text generation
            actions = getattr(m, "supported_actions", []) or []
            if "flash" in display_id.lower() and (not actions or "generateContent" in actions):
                flash_models.append(display_id)

        if flash_models:
            print("\nAvailable Flash models for task automation:")
            for fm in flash_models:
                print(f"  - {fm}")

    except Exception as e:
        print(f"\n❌ Failed to query models: {type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    list_gemini_models()
