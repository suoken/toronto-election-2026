"""Voting-record chunking invariants.

One chunk states exactly one candidate's own vote in one vote-event —
the same attribution discipline as the transcript chunker. Absences are
evidence, phrased as absences, never as No votes.
"""

from comparer.chunking import chunk_voting_records

META = dict(
    source_url="local:member-voting-record-2022-2026.csv",
    fetched_date="2026-10-23",
    doc_title="Toronto Council Voting Record 2022-2026 (Toronto Open Data)",
)


def row(first, last, vote, **overrides):
    base = {
        "First Name": first,
        "Last Name": last,
        "Committee": "City Council",
        "Date/Time": "2026-02-10 10:00 AM",
        "Agenda Item #": "CC38.1",
        "Agenda Item Title": "Budget Implementation Including Property Tax Rates",
        "Motion Type": "Adoption",
        "Vote": vote,
        "Result": "Carried, 20-5",
        "Vote Description": "Majority required",
    }
    base.update(overrides)
    return base


def test_one_chunk_per_candidate_per_event():
    rows = [row("Olivia", "Chow", "Yes"), row("Brad", "Bradford", "No")]
    chunks = chunk_voting_records(rows, **META)
    assert len(chunks) == 2
    by_candidate = {c.candidate: c for c in chunks}
    assert "Olivia Chow voted Yes." in by_candidate["chow"].text
    assert "Brad Bradford voted No." in by_candidate["bradford"].text


def test_chunk_never_states_the_other_candidates_vote():
    rows = [row("Olivia", "Chow", "Yes"), row("Brad", "Bradford", "No")]
    chunks = chunk_voting_records(rows, **META)
    for chunk in chunks:
        if chunk.candidate == "chow":
            assert "Bradford" not in chunk.text
        else:
            assert "Chow" not in chunk.text


def test_absences_are_phrased_as_absences_not_no_votes():
    rows = [
        row("Olivia", "Chow", "Absent"),
        row("Brad", "Bradford", "Absent(Interest Declared)"),
    ]
    chunks = chunk_voting_records(rows, **META)
    by_candidate = {c.candidate: c for c in chunks}
    assert "was absent" in by_candidate["chow"].text
    assert "voted No" not in by_candidate["chow"].text
    assert "declared an interest" in by_candidate["bradford"].text


def test_non_candidate_rows_are_skipped_and_tier_is_record():
    rows = [row("Paul", "Ainslie", "Yes"), row("Olivia", "Chow", "Yes")]
    chunks = chunk_voting_records(rows, **META)
    assert len(chunks) == 1
    assert chunks[0].source_type == "record"
    assert chunks[0].evidence_tier == "voting record"
