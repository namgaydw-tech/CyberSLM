from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True,
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    page = browser.new_page(viewport={"width": 1440, "height": 960})
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:3000", wait_until="networkidle")
    assert page.get_by_text("CyberSLM Console").is_visible()
    assert page.get_by_text("Remote models: unavailable").is_visible()

    page.locator('input[type="file"]').set_input_files(ROOT / "tests/fixtures/three.csv")
    page.get_by_text("3 records").wait_for()
    assert "EVENT_READ" in page.locator(".evidence").inner_text()

    page.locator("#mode").select_option("bridge")
    page.locator("#question").fill("count unique event types")
    page.get_by_role("button", name="Send").click()
    page.locator("article.assistant").get_by_text(
        "Scratch and Qwen inference are unavailable", exact=False
    ).wait_for()

    page.locator("#mode").select_option("chat")
    page.locator("#question").fill("Hello CyberSLM")
    page.get_by_role("button", name="Send").click()
    page.get_by_text("no answer was fabricated", exact=False).wait_for()
    assert not errors, errors
    page.screenshot(path=ROOT / "tests/vercel-smoke.png", full_page=True)
    browser.close()
