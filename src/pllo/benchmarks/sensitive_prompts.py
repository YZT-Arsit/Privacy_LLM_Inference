"""Synthetic sensitive-prompt stress set (NO real PII -- all fabricated).

Generates privacy-stress prompts in three length buckets (128 / 512 / 1024
whitespace tokens) across document types (email, resume, contract, finance memo,
medical-style note, code snippet, API-key redaction, internal meeting notes). Each
prompt embeds clearly-fabricated ``sensitive_spans`` (fake names, fake emails, fake
account/phone numbers, fake API keys) that a transcript scan must NEVER find on a
GPU-visible channel.

Everything is synthetic and deterministic (seeded), so it is safe to commit and
reproducible. stdlib only (random with an explicit seed).
"""

from __future__ import annotations

import random
from typing import Any

__all__ = [
    "DOC_TYPES",
    "TASKS",
    "LENGTH_BUCKETS",
    "build_sensitive_prompt_set",
]

DOC_TYPES = ("email", "resume", "contract", "finance_memo", "medical_note",
             "code_snippet", "api_key_redaction", "meeting_notes")
TASKS = ("summarize", "extract", "rewrite", "qa")
LENGTH_BUCKETS = (128, 512, 1024)

_FIRST = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Jamie"]
_LAST = ["Quill", "Vance", "Ashford", "Brightwater", "Calloway", "Dunmore"]
_COMPANY = ["Aenova Labs", "Brightforge Inc", "Corewave Systems", "Delphi Holdings"]
_FILLER = (
    "The following document is provided for internal processing only and should "
    "be handled according to the applicable data-handling policy. Please review "
    "the contents carefully and follow the requested task. Additional context is "
    "included to make the request realistic and sufficiently long for stress "
    "testing of the protected inference pipeline. ")


def _fake_email(rng, first, last):
    return "%s.%s%d@%s.example" % (first.lower(), last.lower(),
                                   rng.randint(10, 99),
                                   rng.choice(["mail", "corp", "inbox"]))


def _fake_phone(rng):
    return "+1-555-%03d-%04d" % (rng.randint(100, 999), rng.randint(0, 9999))


def _fake_account(rng):
    return "ACCT-%08d" % rng.randint(0, 99999999)


def _fake_api_key(rng):
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    return "sk-FAKE-" + "".join(rng.choice(chars) for _ in range(24))


def _fake_mrn(rng):
    return "MRN-%07d" % rng.randint(0, 9999999)


def _doc_body(rng, doc_type, first, last, company, spans):
    email = _fake_email(rng, first, last)
    phone = _fake_phone(rng)
    spans.extend([f"{first} {last}", email])
    if doc_type == "email":
        spans.append(phone)
        return (f"From: {first} {last} <{email}>\nTo: ops@{company.split()[0].lower()}.example\n"
                f"Subject: Q3 onboarding\n\nHi team, please reach {first} at {phone}. "
                f"My corporate email is {email}.")
    if doc_type == "resume":
        spans.append(phone)
        return (f"Resume of {first} {last}\nContact: {email} | {phone}\n"
                f"Experience: Senior engineer at {company}.")
    if doc_type == "contract":
        acct = _fake_account(rng)
        spans.append(acct)
        return (f"This agreement is between {company} and {first} {last}. "
                f"Settlement to account {acct}. Notices to {email}.")
    if doc_type == "finance_memo":
        acct = _fake_account(rng)
        spans.append(acct)
        return (f"Finance memo: wire from {acct} approved by {first} {last} "
                f"({email}). Amount USD {rng.randint(10000, 990000)}.")
    if doc_type == "medical_note":
        mrn = _fake_mrn(rng)
        spans.append(mrn)
        return (f"Clinical note (synthetic). Patient {first} {last}, {mrn}. "
                f"Follow-up contact {phone}. For summarization only.")
    if doc_type == "code_snippet":
        key = _fake_api_key(rng)
        spans.append(key)
        return (f"# config for {company}\nAPI_KEY = \"{key}\"\n"
                f"OWNER_EMAIL = \"{email}\"\n\ndef connect():\n    return AuthClient(API_KEY)")
    if doc_type == "api_key_redaction":
        key = _fake_api_key(rng)
        spans.append(key)
        return (f"Rotate the leaked credential {key} belonging to {first} {last} "
                f"({email}) before deployment.")
    # meeting_notes
    acct = _fake_account(rng)
    spans.append(acct)
    return (f"Internal meeting notes ({company}). Attendees: {first} {last}. "
            f"Action: update billing for {acct}; ping {email}.")


def _pad_to_tokens(text, target_tokens, rng):
    """Pad ``text`` with deterministic filler until it has >= target whitespace
    tokens (capped: never exceeds the bucket by more than one filler block)."""
    tokens = len(text.split())
    while tokens < target_tokens:
        text = text + " " + _FILLER
        tokens = len(text.split())
    # trim back to the bucket so the input stays <= the bucket size
    words = text.split()
    if len(words) > target_tokens:
        words = words[:target_tokens]
    return " ".join(words)


_TASK_INSTR = {
    "summarize": "Summarize the document in two sentences.",
    "extract": "Extract the key entities (names, contacts) as a list.",
    "rewrite": "Rewrite the document in a neutral, professional tone.",
    "qa": "Answer: who is the document about and what is the requested action?",
}


def build_sensitive_prompt_set(*, num_per_bucket: int = 4,
                               buckets: tuple[int, ...] = LENGTH_BUCKETS,
                               seed: int = 2035) -> list[dict[str, Any]]:
    """Build a deterministic synthetic sensitive-prompt set (no real PII)."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    idx = 0
    for bucket in buckets:
        for j in range(num_per_bucket):
            doc_type = DOC_TYPES[(idx) % len(DOC_TYPES)]
            task = TASKS[(idx) % len(TASKS)]
            first = rng.choice(_FIRST)
            last = rng.choice(_LAST)
            company = rng.choice(_COMPANY)
            spans: list[str] = []
            body = _doc_body(rng, doc_type, first, last, company, spans)
            instr = _TASK_INSTR[task]
            prompt = _pad_to_tokens("%s\n\nDocument:\n%s" % (instr, body),
                                    bucket, rng)
            rows.append({
                "id": "sp-%d-%04d" % (bucket, idx),
                "dataset": "sensitive_prompt_1024", "prompt": prompt,
                "sensitive_spans": sorted(set(spans)),
                "length_bucket": bucket, "task": task,
                "meta": {"doc_type": doc_type, "synthetic": True,
                         "contains_real_pii": False}})
            idx += 1
    return rows
