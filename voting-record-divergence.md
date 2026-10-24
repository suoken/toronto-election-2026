# Voting-record divergence — Chow vs Bradford (prepared 2026-10-24, for the weekend voting brief)

Source: `corpus/records/member-voting-record-2022-2026.csv` (Toronto Open Data, "Members of Toronto City Council – Voting Record"), 46,815 rows, 2022-11-23 to 2026-08-05. Computed 2026-10-24 from the CSV in this repo; not yet chunked into the comparer (the voting-record tier is still empty in the build).

## Headline, with the caveat first

- Shared window is July 2023–July 2026: Chow's rows start 2023-07-19 (she became mayor that month); Bradford's start 2022-11-23. Only the overlap can be compared.
- Unit = one council vote-event (a motion on an agenda item, including amendments and procedural motions — NOT just final passage of an item). A split on an amendment is not the same as a split on the final item. Read the examples below as "they voted differently on that motion", not "they opposed the whole item".
- Among vote-events where both recorded Yes or No: **1,025 events, agreed 739 (72.1%), split 286 (27.9%)**.
- Absences in shared events: Chow absent ~12.1%, Bradford ~18.8% (includes "Absent (Interest Declared)"). Absence is not a No vote and is excluded from the 72.1% figure.

## Where they split most (keyword in agenda title or motion; categories overlap)

| Topic | Vote-events | Splits |
|---|---|---|
| Budget | 96 | 39 |
| Tax | 71 | 33 |
| Transit | 66 | 22 |
| Housing | 91 | 18 |
| Homelessness | 27 | 12 |
| Development charges | 13 | 3 |
| Rent | 27 | 6 |
| Water | 7 | 0 |
| Bike lanes (title keyword only) | 2 | 1 |

So the record's sharpest divergence is fiscal — budget and tax motions — not water, where the 7 title-matched vote-events show no direct split. Bike-lane and police splits are barely captured by title keywords in this CSV, so do not use this table to claim they agree on those issues; the campaign platforms and debate are the better source there.

## Illustrative direct splits (both Yes/No, most recent first)

- 2026-07-30, MM43.50 "The Mayor's Seat: Restoring Direct Mayoral Accountability on the Toronto Police Service Board" (Bradford motion, waive referral): Chow No, Bradford Yes.
- 2026-06-24, MM42.44 "Reducing Development Charges" (Chow motion): Chow Yes / Bradford No on the main motion; Chow No / Bradford Yes on two Holyday amendments — i.e. they fought over the amendments, not just the item.
- 2026-02-10, CC38.1 "Budget Implementation Including Property Tax Rates, User Fees and Related Matters": Chow Yes / Bradford No on several recommendation adoptions; Chow No / Bradford Yes on Bradford motion 8 and Holyday motion 11.
- 2025-12-17, EX28.1 "Making life more affordable for families by asking luxury-home buyers to chip in more": Chow Yes / Bradford No on adoption as amended; Chow No / Bradford Yes on Bradford motion 5b.
- 2025-12-17, MM35.10 Transform Yonge (public realm + cycling, no lane loss): Chow Yes, Bradford No.
- 2026-03-26, MM39.27 Grocery Store Pilot Project: Chow Yes / Bradford No across adoption and several amendments.

## What this does NOT establish

- It does not say who "wins" on any issue, and it must not be turned into a recommendation — the comparer boundary (compare, never recommend) applies here too.
- It does not cover the CP24 Oct 22 debate or any debate beyond TVO Oct 15; those transcripts are still blocked (see corpus/AUDIT.md).
- Per-item final-passage agreement would need collapsing amendments to one vote per agenda item; that step is not done here, so quote the 72.1% figure only with its vote-event definition.

## Suggested use in the brief

One paragraph + the fiscal table above, framed as: "On council, when both voted Yes/No, they agreed about 7 times in 10 — and disagreed most on budget/tax motions." Then point readers to the platforms for the issues the record barely captures (bike lanes, TTC fares, water privatization framing).
