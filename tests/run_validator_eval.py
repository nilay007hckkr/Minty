"""Live eval of the groundedness validator in isolation.

Each case pairs a KB chunk with a hand-written answer: either faithful (often
paraphrased, to catch over-literal rejection) or deliberately corrupted (wrong
number, invented condition/term). Reports:
  - catch rate:            corrupted answers flagged ungrounded  (higher = better)
  - false-rejection rate:  faithful answers flagged ungrounded   (lower = better)

Usage: uv run python -m tests.run_validator_eval [--runs N]
Costs N x len(CASES) gpt-oss-120b calls.
"""

import argparse
from dotenv import load_dotenv

load_dotenv()

from app.ingestion import load_and_chunk
from app.nodes import validate_node

# (substring identifying the source chunk, answer, answer is faithful?)
CASES = [
    ("daily ATM withdrawal limit",
     "The standard daily ATM withdrawal limit is $1,000, while Premium and Wealth Management account holders get $2,500 per day.",
     True),
    ("daily ATM withdrawal limit",
     "The standard daily ATM withdrawal limit is $1,500, while Premium and Wealth Management account holders get $2,500 per day.",
     False),
    ("out-of-network ATM",
     "Using an out-of-network ATM costs $3.00 per transaction, and the ATM's operator may add its own surcharge.",
     True),
    ("out-of-network ATM",
     "Using an out-of-network ATM costs $3.00 per transaction, but this fee is waived for all savings account holders.",
     False),
    ("domestic wire transfer",
     "Submit and approve the wire by 4:00 PM Eastern Time on a business day for same-day processing. Outgoing domestic wires cost $25.",
     True),
    ("domestic wire transfer",
     "Submit and approve the wire by 5:00 PM Eastern Time on a business day for same-day processing. Outgoing domestic wires cost $25.",
     False),
    ("documents do I need",
     "To open a new checking account you'll need a government-issued photo ID, your Social Security number or ITIN, and at least $50 for the opening deposit. You can apply online in about 10 minutes or at any branch.",
     True),
    ("documents do I need",
     "To open a new checking account you'll need a government-issued photo ID, your Social Security number or ITIN, proof of address, and at least $50 for the opening deposit.",
     False),
    ("High-Yield Savings",
     "The High-Yield Savings account pays 4.75% APY on balances up to $50,000 and 3.00% APY above that, with a $100 minimum opening deposit. The rate is variable.",
     True),
    ("High-Yield Savings",
     "The High-Yield Savings account pays a fixed 4.75% APY on all balances, with a $100 minimum opening deposit.",
     False),
    ("credit limit increase",
     "Yes. Request it in the mobile app under Card Settings or by calling customer support. Reviews usually take 2-3 business days and may include a credit check.",
     True),
    ("credit limit increase",
     "Yes. Request it in the mobile app under Card Settings. Reviews usually take 2-3 business days and include a hard inquiry that lowers your credit score by about 5 points.",
     False),
    ("fee-free withdrawals",
     "You can make up to six fee-free withdrawals or transfers each statement cycle; every additional one costs $5.",
     True),
    # Subtle: found in the hard-negative eval, where all three systems (and
    # Minty's validator) turned "no minimum balance to start earning interest"
    # into "no minimum opening deposit".
    ("standard savings account earns",
     "The standard savings account earns 2.10% APY, compounded daily and credited monthly, and there's no minimum balance required to start earning interest.",
     True),
    ("standard savings account earns",
     "There is no minimum opening deposit for a standard savings account, so you can open it with any amount and start earning 2.10% APY.",
     False),
    ("fee-free withdrawals",
     "You can make up to 6 fee-free withdrawals per statement cycle; every additional one costs $5, and the count resets on the 1st of each month.",
     False),
]


def find_chunk(chunks, needle: str):
    matches = [c for c in chunks if needle.lower() in c.page_content.lower()]
    if not matches:
        raise ValueError(f"No chunk contains {needle!r}")
    return matches[0]


def main(runs: int):
    chunks = load_and_chunk()
    caught = missed = false_rej = accepted = 0

    for needle, answer, faithful in CASES:
        doc = find_chunk(chunks, needle)
        for _ in range(runs):
            result = validate_node({"answer": answer, "documents": [doc]})
            grounded = result["is_grounded"]
            if faithful:
                accepted += grounded
                false_rej += not grounded
            else:
                caught += not grounded
                missed += grounded
            if grounded != faithful:
                kind = "FALSE REJECT" if faithful else "MISSED"
                print(f"[{kind}] {answer}")
                print(f"    flagged: {result['unsupported_claims']}")

    n_bad, n_good = caught + missed, accepted + false_rej
    print(f"\n{'=' * 60}")
    print(f"Runs per case: {runs}")
    print(f"Catch rate (corrupted flagged):      {caught}/{n_bad}")
    print(f"False rejections (faithful flagged): {false_rej}/{n_good}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    main(parser.parse_args().runs)
