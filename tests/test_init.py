import traceback
from pathlib import Path

from core.json_engine import JsonEngine


def main():
    engines_dir = Path(__file__).parent.parent / "engines"
    chatgpt_json = engines_dir / "chatgpt.json"
    engine = JsonEngine(chatgpt_json)
    print("Attempting initialization...")
    try:
        engine.is_user_logged_in()
        print("Driver started SUCCESS!")
    except Exception:
        print("Driver started FAILED!")
        traceback.print_exc()


def test_json_engine_loads_accept_button_selectors():
    engines_dir = Path(__file__).parent.parent / "engines"
    engine = JsonEngine(engines_dir / "copilot.json")

    assert engine.accept_button_selectors == [
        "#app > div.flex.h-full.overflow-hidden.bg-sidebar-light.dark\\:bg-sidebar-dark > main > div.relative.size-full.overflow-hidden > div.relative.size-full.overflow-hidden.md\\:rounded-container > div.pointer-events-none.absolute.bottom-0.w-full.sm\\:bottom-1\\/2.sm\\:translate-y-1\\/2.sm\\:pt-36 > div > div > div > div.relative.max-h-full.min-h-composer.min-w-16.max-w-chat.rounded-5xl > div.relative.shadow-tinted-xl.backdrop-blur-2xl.backdrop-saturate-200.bg-accent-100\\/60.dark\\:bg-muted-200\\/50 > div > div.w-prompt-sign-in > div > div > dialog > div > div.flex.size-full.flex-col.gap-4.p-0.text-left > div > button:nth-child(1)"
    ]


if __name__ == "__main__":
    main()
