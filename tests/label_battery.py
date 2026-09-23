"""Live check of the label question types (noul, choice, score) against a
running server, on either engine.

    python3 tests/label_battery.py [--base http://127.0.0.1:8011]
"""
import argparse
import json
import sys
import time
import urllib.request

# (state, questions, expected answers). A noul expects "yes"/"no", a choice the
# option name, a score the level name.
CASES = [
    ("The ball is red and the box is large.",
     {"red": ("noul", "Is the ball red?"), "blue": ("noul", "Is the ball blue?"), "large": ("noul", "Is the box large?"),
      "small": ("noul", "Is the box small?"), "cat": ("noul", "Is there a cat?")},
     {"red": "yes", "blue": "no", "large": "yes", "small": "no", "cat": "no"}),
    ("Everything is down and we have a demo at noon. Please call me back right now.",
     {"urgent": ("noul", "Does the customer need a reply within the hour?"),
      "team": ("choice", "Which team owns this?", ["billing", "outage", "feature"]),
      "tone": ("score", "How upset is the writer?", ["calm", "annoyed", "furious"])},
     {"urgent": "yes", "team": "outage", "tone": "annoyed"}),
    ("Hi, I was charged twice for my March invoice. No rush, whenever you get a chance.",
     {"urgent": ("noul", "Does the customer need a reply within the hour?"),
      "team": ("choice", "Which team owns this?", ["billing", "outage", "feature"]),
      "tone": ("score", "How upset is the writer?", ["calm", "annoyed", "furious"])},
     {"urgent": "no", "team": "billing", "tone": "calm"}),
    ("Could you add a dark mode to the dashboard? Would be nice to have someday.",
     {"team": ("choice", "Which team owns this?", ["billing", "outage", "feature"]),
      "urgent": ("noul", "Does the customer need a reply within the hour?")},
     {"team": "feature", "urgent": "no"}),
    ("Der Zug nach Berlin fährt um neun Uhr ab.",
     {"lang": ("choice", "Which language is the text in?", ["English", "German", "French", "Spanish"])},
     {"lang": "German"}),
    ("Le train pour Paris part à neuf heures.",
     {"lang": ("choice", "Which language is the text in?", ["English", "German", "French", "Spanish"])},
     {"lang": "French"}),
    ("The Eiffel Tower is in Paris. Paris is the capital of France.",
     {"capital": ("choice", "What is the capital of France?", ["Lyon", "Paris", "Marseille", "Nice"]),
      "tower": ("noul", "Is the Eiffel Tower in Lyon?")},
     {"capital": "Paris", "tower": "no"}),
    ("Water boils at 100 degrees Celsius at sea level.",
     {"boil": ("choice", "At what temperature does water boil at sea level?", ["0 C", "50 C", "100 C", "212 C"]),
      "unit": ("choice", "Which unit does the text use?", ["Celsius", "Fahrenheit", "Kelvin"])},
     {"boil": "100 C", "unit": "Celsius"}),
    ("I absolutely love this product, it changed my mornings completely!",
     {"sentiment": ("score", "How positive is the review?", ["negative", "neutral", "positive"])},
     {"sentiment": "positive"}),
    ("It arrived broken and support never answered. Never again.",
     {"sentiment": ("score", "How positive is the review?", ["negative", "neutral", "positive"]),
      "refund": ("noul", "Does the writer sound like they want a refund or to stop buying?")},
     {"sentiment": "negative", "refund": "yes"}),
]


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    args = ap.parse_args()
    ok = n = 0
    t0 = time.time()
    reads = 0
    for text, qs, want in CASES:
        questions = {}
        for k, spec in qs.items():
            kind = spec[0]
            if kind == "noul":
                questions[k] = {"type": "noul", "instructions": spec[1]}
            elif kind == "choice":
                questions[k] = {"type": "choice", "instructions": spec[1], "criteria": {o: None for o in spec[2]}}
            else:
                questions[k] = {"type": "score", "instructions": spec[1], "criteria": spec[2]}
        r = post(args.base, "/v1/systemone", {"model": "jev-latest", "state": text, "questions": questions})
        reads += r["diagnostics"]["timing"]["reads"]
        for k, truth in want.items():
            n += 1
            a = r["answers"][k]
            if a["type"] == "noul":
                got = "yes" if a["noul"] >= 0.5 else "no"
                conf = a["noul"] if got == "yes" else 1 - a["noul"]
            elif a["type"] == "choice":
                got, conf = a["choice"], a["confidence"]
            else:
                levels = list(a["legend"].values())
                got = max(a["probabilities"], key=a["probabilities"].get)
                got = a["legend"][got]
                conf = a["confidence"]
            hit = got == truth
            ok += hit
            if not hit:
                print(f"  BAD {k:10s} got {got!r:12s} want {truth!r:12s} conf {conf:.2f} | {text[:50]}")
    print(f"labels: {ok}/{n} in {time.time() - t0:.1f} s, {reads} reads, engine {r['diagnostics'].get('engine')}")
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
