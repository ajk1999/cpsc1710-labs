import os, time, datetime, pathlib
import requests

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-6-luna"
MAX_STEPS = 8
DESK = pathlib.Path("desk")

SYSTEM = """You are an assistant working at a student's desk. You cannot see the desk directly.
You can use tools. To use a tool, reply with exactly one line and nothing else:
TOOL: tool_name | argument

The tools:
- list_files | (leave the argument empty)   lists the files on the desk
- read_file | filename                      returns the text of one file
- days_until | YYYY-MM-DD                   counts the days from today to that date

Use one tool per reply, then wait for the result.
When you have everything you need, reply with:
ANSWER: your answer for the student"""


def make_desk():
    today = datetime.date.today()
    draft_due = today + datetime.timedelta(days=25)
    DESK.mkdir(exist_ok=True)
    for old in DESK.glob("*"):
        old.unlink()
    (DESK / "deadlines.txt").write_text(
        "Sample Seminar: deadlines (made-up course, for practice)\n"
        f"- Reading response 3: {today + datetime.timedelta(days=4)}\n"
        f"- Lab report, first draft: {draft_due}\n"
        f"- Lab report, final version: {draft_due + datetime.timedelta(days=20)}\n")
    (DESK / "report-instructions.txt").write_text(
        "Lab report: instructions (made-up course, for practice)\n"
        "Write for a reader who has ten minutes.\n"
        "Word limit: 1,200 words, not counting references.\n"
        "Structure: question, method, result, one limitation.\n"
        "Submit as a PDF.\n")
    (DESK / "reading-list.txt").write_text(
        "Reading list (made-up course, for practice)\n"
        "Week 7: two short articles on measurement.\n"
        "Week 8: one chapter, on writing up results.\n")


def list_files(_):
    return "\n".join(sorted(f.name for f in DESK.iterdir()))


def read_file(filename):
    f = DESK / filename.strip()
    return f.read_text() if f.exists() else "ERROR: there is no file called " + filename.strip()


def days_until(date):
    try:
        days = (datetime.date.fromisoformat(date.strip()) - datetime.date.today()).days
        return f"{days} days from today"
    except ValueError:
        return "ERROR: write the date as YYYY-MM-DD"


TOOLS = {"list_files": list_files, "read_file": read_file, "days_until": days_until}


def call_model(messages):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY first.")
    r = requests.post(f"{BASE_URL}/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": MODEL, "messages": messages, "max_tokens": 1500},
                      timeout=120).json()
    if "error" in r:
        raise RuntimeError("The model service returned an error: " + str(r["error"])[:300])
    return (r["choices"][0]["message"].get("content") or "").strip()


def first_action(reply):
    """Keep the first TOOL line, or the ANSWER and what follows it."""
    lines = reply.strip().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("TOOL:"):
            return line.strip()
        if line.strip().startswith("ANSWER:"):
            return "\n".join(lines[i:]).strip()
    return reply.strip()


def split_tool_line(line):
    if not line.startswith("TOOL:"):
        return "(nothing: start your reply with TOOL: or ANSWER:)", ""
    name, _, arg = line[5:].partition("|")
    return name.strip(), arg.strip()


def run_agent(question, max_steps=MAX_STEPS, model_fn=call_model):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for step in range(max_steps):
        reply = first_action(model_fn(messages))
        print(f"\n--- call {step + 1}: the model wrote ---\n{reply}")
        messages.append({"role": "assistant", "content": reply})
        if reply.startswith("ANSWER:"):
            return reply
        name, arg = split_tool_line(reply)
        if name not in TOOLS:
            result = "ERROR: there is no tool called " + name
        else:
            result = TOOLS[name](arg)
        print(f"\n--- the program ran the tool; the result is ---\n{result}")
        messages.append({"role": "user", "content": "RESULT:\n" + result})
    return "Stopped: out of steps."


if __name__ == "__main__":
    make_desk()
    run_agent("When is the first draft of my lab report due, how many days do I have, and what is the word limit?")
