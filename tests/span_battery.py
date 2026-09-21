"""Live check of the span and spans question types against a running server.

    python3 tests/span_battery.py [--base http://127.0.0.1:8011] [--seeds 1 2]

Every expected value is a substring of its text; a field whose expected value
is "none" must come back not found. Also runs a mixed schema (labels beside
spans), a list question, a long text that needs windows, and the chat form.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

CASES = [
    ("Invoice #A-1042 from Northwind Traders. Total due: $1,234.56 by 2024-03-15. "
     "Contact billing@northwind.example or +1 (415) 555-0142.",
     {"invoice_id": ("the invoice id", "A-1042"), "vendor": ("the vendor name", "Northwind Traders"),
      "amount": ("the total amount due", "$1,234.56"), "due_date": ("the due date", "2024-03-15"),
      "email": ("the billing email address", "billing@northwind.example"), "phone": ("the phone number", "+1 (415) 555-0142"),
      "tracking": ("the shipment tracking number", "none")}),
    ("The gateway at 10.0.4.17 dropped 3.2% of packets; failover to 192.168.100.254 completed at 02:14 UTC, "
     "and CPU peaked at 91%. Server https://status.example.com/api/v1/health returned 503 after 30s; "
     "admin on call is Dana Whitfield (dana.w@example.com).",
     {"gateway_ip": ("the IP address of the gateway", "10.0.4.17"), "failover_ip": ("the IP address failover went to", "192.168.100.254"),
      "packet_loss": ("the packet loss percentage", "3.2%"), "failover_time": ("the time the failover completed", "02:14 UTC"),
      "cpu_peak": ("the CPU peak percentage", "91%"), "status": ("the HTTP status code the server returned", "503"),
      "timeout": ("how long the server took before returning the status", "30s"),
      "admin_name": ("the name of the admin on call, without the email", "Dana Whitfield"),
      "admin_email": ("the admin's email address", "dana.w@example.com"), "ticket": ("the incident ticket number", "none")}),
    ("Order 88213 shipped 2 items weighing 4.8 kg to 221B Baker Street, London NW1 6XE; tracking 1Z999AA10123456784.",
     {"order_id": ("the order number", "88213"), "weight": ("the total weight", "4.8 kg"), "street": ("the street and house number, without the city or postcode", "221B Baker Street"),
      "postcode": ("the postcode", "NW1 6XE"), "tracking": ("the tracking number", "1Z999AA10123456784")}),
    ("Alex Smith sent me your way. I'm Maya Chen, a software engineer at Fern Labs. My email is maya.old@example.com, actually use maya.chen@example.com.",
     {"name": ("The speaker's full name, not somebody else's.", "Maya Chen"), "role": ("The speaker's job title, without a leading article.", "software engineer"),
      "email": ("The speaker's current email. Respect explicit corrections.", "maya.chen@example.com"), "phone": ("The speaker's phone number.", "none"),
      "referrer": ("the name of the person who referred the speaker", "Alex Smith"), "employer": ("the company the speaker works at", "Fern Labs")}),
    ("Table for six, under Dana Whitfield.", {"name": ("full name", "Dana Whitfield"), "size": ("party size", "six"), "phone": ("phone number", "none")}),
    ("Alex Smith referred me. I'm Maya Chen, born 09/07/1990. Email maya.chen@example.com or call +1 (415) 555-0123. My doctor is treating my asthma. The meeting is next Tuesday.",
     {"speaker": ("the speaker's full name", "Maya Chen"), "dob": ("the speaker's date of birth", "09/07/1990"), "email": ("the email address", "maya.chen@example.com"),
      "phone": ("the phone number", "+1 (415) 555-0123"), "condition": ("the medical condition mentioned", "asthma"), "meeting": ("when the meeting is", "next Tuesday")}),
    ("Please send the recieved invoices to accounting before Friday.",
     {"typo": ("the misspelled word", "recieved"), "deadline": ("the deadline day, the day only", "Friday"), "recipient": ("who the invoices go to", "accounting")}),
    ("Alex referred me; I am José van  der Berg.", {"name": ("full name of speaker", "José van  der Berg"), "referrer": ("who referred the speaker", "Alex")}),
    ("Call Dana Whitfield on 07700 900123.", {"name": ("the person to call", "Dana Whitfield"), "phone": ("the phone number", "07700 900123")}),
    ("Dr. Ada Lovelace paid $3.50 to ada@example.com. Was it late? No!",
     {"payer": ("who paid", "Dr. Ada Lovelace"), "amount": ("the amount paid", "$3.50"), "payee": ("the email the payment went to", "ada@example.com"),
      "late": ("whether it was late, as answered in the text", "No")}),
    ("👋 my name is JustinKessler", {"name": ("the person's name", "JustinKessler")}),
]
LISTS = [
    (CASES[1][0], "IP address", ["10.0.4.17", "192.168.100.254"]),
    (CASES[3][0], "person's name", ["Alex Smith", "Maya Chen"]),
    ("Invoice total $1,234.56, shipping $18.00, tax $99.10, paid $500 so far.", "dollar amount", ["$1,234.56", "$18.00", "$99.10", "$500"]),
    ("Matt said hi to Dana. Then Matt left, and Dana waved at Matt.", "occurrence of the name Matt", ["Matt", "Matt", "Matt"]),
    ("The meeting is at 3pm on Thursday in the usual room.", "IP address", []),
]


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1])
    args = ap.parse_args()
    failures = 0
    for seed in args.seeds:
        ok = n = 0
        t0 = time.time()
        for text, fields in CASES:
            qs = {k: {"type": "span", "instructions": q} for k, (q, _) in fields.items()}
            r = post(args.base, "/v1/systemone", {"model": "jev-latest", "state": text, "questions": qs, "seed": seed})
            for k, (q, truth) in fields.items():
                n += 1
                a = r["answers"][k]
                got = a["text"] if a["found"] else "none"
                sliced = (not a["found"]) or text[a["start"]:a["end"]] == a["text"]
                hit = norm(got) == norm(truth) and sliced
                ok += hit
                if not hit:
                    print(f"  BAD {k:12s} got {got!r:28s} want {truth!r:24s} conf {a['confidence']} reads {a['reads']}{'' if sliced else '  OFFSETS WRONG'}")
        print(f"spans seed {seed}: {ok}/{n} in {time.time() - t0:.1f} s")
        failures += n - ok

    ok = 0
    for text, kind, want in LISTS:
        r = post(args.base, "/v1/systemone", {"model": "jev-latest", "state": text, "questions": {"all": {"type": "spans", "instructions": kind}}})
        items = r["answers"]["all"]["items"]
        got = [i["text"] for i in items]
        mono = all(items[i]["end"] <= items[i + 1]["start"] for i in range(len(items) - 1)) and all(text[i["start"]:i["end"]] == i["text"] for i in items)
        hit = [norm(g) for g in got] == [norm(w) for w in want] and mono
        ok += hit
        if not hit:
            print(f"  BAD list {kind!r}: got {got} want {want}{'' if mono else '  NOT MONOTONIC'}")
    print(f"lists: {ok}/{len(LISTS)}")
    failures += len(LISTS) - ok

    # a mixed schema: labels beside spans, through the Jev shape and the chat form
    text = CASES[1][0]
    qs = {"outage": {"type": "noul", "instructions": "Is a service down or degraded?"},
          "team": {"type": "choice", "instructions": "Which team owns this?", "criteria": {"network": None, "database": None, "frontend": None}},
          "admin": {"type": "span", "instructions": "the name of the admin on call, without the email"},
          "ips": {"type": "spans", "instructions": "IP address"}}
    r = post(args.base, "/v1/systemone", {"model": "jev-latest", "state": text, "questions": qs})
    a = r["answers"]
    mixed_ok = a["outage"]["noul"] > 0.5 and a["team"]["choice"] == "network" and a["admin"]["text"] == "Dana Whitfield" and [i["text"] for i in a["ips"]["items"]] == ["10.0.4.17", "192.168.100.254"]
    print(f"mixed schema: {'ok' if mixed_ok else 'BAD ' + json.dumps({k: (v.get('choice') or v.get('text') or v.get('noul') or [i['text'] for i in v.get('items', [])]) for k, v in a.items()})}  reads {r['diagnostics']['timing']['reads']} {r['diagnostics']['timing']['total_ms']:.0f} ms")
    failures += not mixed_ok
    schema = {"questions": [{"id": "admin", "type": "span", "instructions": "the name of the admin on call, without the email", "max_tokens": 16},
                            {"id": "team", "type": "choice", "instructions": "Which team owns this?", "options": ["network", "database", "frontend"]}]}
    r = post(args.base, "/v1/chat/completions", {"messages": [{"role": "system", "content": json.dumps(schema)}, {"role": "user", "content": json.dumps({"text": text})}]})
    c = json.loads(r["choices"][0]["message"]["content"])
    chat_ok = c["answers"]["admin"]["text"] == "Dana Whitfield" and c["answers"]["team"]["choice"] == "network"
    print(f"chat form, JSON state: {'ok' if chat_ok else 'BAD ' + json.dumps(c['answers'])}")
    failures += not chat_ok

    # a long text: several windows, answers from different parts
    doc = "\n\n".join(t for t, _ in CASES)
    qs = {"gateway_ip": {"type": "span", "instructions": "the IP address of the gateway"}, "postcode": {"type": "span", "instructions": "the postcode"},
          "payer": {"type": "span", "instructions": "who paid $3.50"}, "typo": {"type": "span", "instructions": "the misspelled word"}}
    want = {"gateway_ip": "10.0.4.17", "postcode": "NW1 6XE", "payer": "Dr. Ada Lovelace", "typo": "recieved"}
    r = post(args.base, "/v1/systemone", {"model": "jev-latest", "state": doc, "questions": qs})
    ok = 0
    for k, w in want.items():
        a = r["answers"][k]
        hit = a["found"] and norm(a["text"]) == norm(w) and doc[a["start"]:a["end"]] == a["text"]
        ok += hit
        if not hit:
            print(f"  BAD long {k}: got {a.get('text')!r} want {w!r} conf {a['confidence']}")
    print(f"long text ({len(doc)} chars, windows {r['answers']['gateway_ip'].get('windows')}): {ok}/{len(want)} in {r['diagnostics']['timing']['total_ms']:.0f} ms")
    failures += len(want) - ok
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
