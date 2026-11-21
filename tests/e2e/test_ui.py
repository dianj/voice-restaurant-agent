"""End-to-end browser tests (Playwright) for the conversation pane and the staff view.

The LLM is the scripted fake from tests/fake_lmstudio.py and the clock is frozen, so each test is deterministic.
"""
import pytest
from playwright.sync_api import Page, expect

from restaurant import tools

pytestmark = pytest.mark.e2e


def open_app(page: Page, app_url: str):
    page.goto(app_url)
    expect(page.locator(".msg.bot")).to_have_count(1)  # greeting


def send(page: Page, text: str):
    page.locator("#input").fill(text)
    page.locator("#input").press("Enter")


def last_bot(page: Page):
    return page.locator(".msg.bot").last


def restaurant_10_to_22():
    with tools.db() as conn:
        return conn.execute("SELECT restaurant_id, name FROM restaurants WHERE opening_hours = '10:00-22:00' "
                            "ORDER BY restaurant_id LIMIT 1").fetchone()


def test_reply_shows_tool_calls_with_their_arguments(page: Page, app_url, fake_lm):
    fake_lm.script(("search_restaurants", {"city": "Sofia", "cuisine": "Italian"}), reply="Marble Hearth is a great choice.")
    open_app(page, app_url)
    send(page, "Italian in Sofia")
    expect(last_bot(page)).to_contain_text("Marble Hearth is a great choice.")
    chip = last_bot(page).locator(".tool")
    expect(chip).to_have_text("search_restaurants")
    expect(chip).to_have_class("tool success")
    chip.click()
    expect(last_bot(page).locator(".tool-args")).to_contain_text('"cuisine":"Italian"')


def test_model_output_is_rendered_as_text_not_html(page: Page, app_url, fake_lm):
    fake_lm.script(reply='<img src=x onerror="window.pwned=1"><b>bold</b>')
    open_app(page, app_url)
    send(page, "hi")
    expect(last_bot(page)).to_contain_text('<img src=x onerror="window.pwned=1">')
    assert last_bot(page).locator("img, b").count() == 0
    assert page.evaluate("window.pwned") is None


def test_booking_appears_in_the_staff_view(page: Page, app_url, fake_lm):
    rid, name = restaurant_10_to_22()
    fake_lm.script(("book_table", {"restaurant_id": rid, "customer_name": "Maria", "date": "2026-11-21",
                                   "time": "19:30", "guests": 2}), reply="Booked.")
    open_app(page, app_url)
    expect(page.locator("#reservations")).to_contain_text("No reservations yet")
    send(page, "Yes, book it")
    row = page.locator("#reservations tr").first
    expect(row).to_contain_text("Maria")
    expect(row).to_contain_text(name)
    expect(row).to_contain_text("2026-11-21 19:30")
    expect(row).to_have_class("fresh")
    expect(row.locator(".code")).to_have_text(tools.list_reservations()[0]["code"])


def test_cancelled_reservation_is_shown_as_cancelled(page: Page, app_url, fake_lm):
    rid, _ = restaurant_10_to_22()
    code = __import__("json").loads(tools.book_table(rid, "Ivan", "2026-11-21", "12:00", 4))["reservation_code"]
    fake_lm.script(("cancel_reservation", {"code": code, "customer_name": "Ivan"}), reply="Cancelled.")
    open_app(page, app_url)
    send(page, "Cancel it")
    expect(page.locator("#reservations tr").first).to_have_class("cancelled")
    expect(page.locator("#reservations .badge")).to_have_text("cancelled")


def test_enter_sends_and_shift_enter_adds_a_line(page: Page, app_url, fake_lm):
    open_app(page, app_url)
    box = page.locator("#input")
    box.fill("first line")
    box.press("Shift+Enter")
    box.type("second line")
    expect(box).to_have_value("first line\nsecond line")
    assert fake_lm.requests == []


def test_example_chip_sends_its_text(page: Page, app_url, fake_lm):
    fake_lm.script(reply="Here are some options.")
    open_app(page, app_url)
    page.locator(".example").first.click()
    expect(page.locator(".msg.user").last).to_have_text("Find a highly rated Japanese restaurant in Sofia")
    expect(last_bot(page)).to_contain_text("Here are some options.")


def test_new_conversation_clears_the_chat_and_the_server_history(page: Page, app_url, fake_lm):
    fake_lm.script(reply="Hi Maria")
    open_app(page, app_url)
    send(page, "My name is Maria")
    expect(last_bot(page)).to_contain_text("Hi Maria")
    page.get_by_role("button", name="New conversation").click()
    expect(page.locator(".msg")).to_have_count(1)
    fake_lm.script(reply="Fresh start")
    send(page, "Who am I?")
    expect(last_bot(page)).to_contain_text("Fresh start")
    assert "Maria" not in str(fake_lm.requests[-1]["messages"])


def test_llm_outage_shows_a_friendly_error(page: Page, app_url, fake_lm):
    from restaurant import agent
    fake_lm.fail_count = agent.llm_client.max_retries + 1
    open_app(page, app_url)
    send(page, "Book a table")
    expect(last_bot(page)).to_have_class("msg bot error")
    expect(last_bot(page)).to_contain_text("colleague")


def test_voice_replies_request_tts_in_the_reply_language(page: Page, app_url, fake_lm):
    requests = []

    def fake_tts(route):
        requests.append(route.request.post_data)
        route.fulfill(status=200, content_type="audio/mpeg", body=b"")

    fake_lm.script(reply="Вашата маса е запазена за 19:30.")
    page.route("**/api/tts", fake_tts)
    open_app(page, app_url)
    page.locator("#voiceReplies").check()
    send(page, "Запази маса")
    expect(last_bot(page)).to_contain_text("19:30")
    page.wait_for_timeout(500)
    assert requests and 'name="language"\r\n\r\nbg' in requests[0]


def test_header_shows_the_model_status(page: Page, app_url):
    open_app(page, app_url)
    expect(page.locator("#llmStatus .dot")).to_have_class("dot ok")
    expect(page.locator("#llmStatus")).to_contain_text("gemma-4-26b-a4b-it")
