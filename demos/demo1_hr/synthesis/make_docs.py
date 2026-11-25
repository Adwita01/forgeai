"""Synthesise the Demo 1 corpus: three maritime HR policy PDFs for a fictional ship manager.

Content was drafted by an LLM (Claude) from the prompts in synth_prompts.md, then reviewed
for internal consistency and frozen here as data. Rendering is deterministic.

    python demos/demo1_hr/synthesis/make_docs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from pdf_render import PAGEBREAK, render  # noqa: E402

DOCS = HERE.parent / "docs"
CO = "Bluewater Fleet Management Pte. Ltd."

# --------------------------------------------------------------------------- HRP-01
SHORE_LEAVE_POLICY = [
    ("1. Purpose and Scope", [
        f"This policy sets out the leave entitlements of all permanent shore-based employees of {CO} (the Company), "
        "including office staff in Singapore, Rotterdam and Mumbai. It does not apply to seafarers serving on Company-managed "
        "vessels, whose leave is governed by the Crew Leave, Rotation and Travel Policy (CRT-03).",
        "Contract and temporary staff are covered only where their individual contract expressly refers to this policy.",
    ]),
    ("2. Annual Leave Entitlement", [
        "Shore staff are entitled to 24 working days of paid annual leave per calendar year. Employees who have completed "
        "5 years of continuous service are entitled to 30 working days per calendar year from the start of the following year.",
        "Annual leave accrues monthly at 2 working days per completed month of service (2.5 working days for employees on the "
        "30-day entitlement). New joiners may take accrued leave after completing their 3-month probation period.",
        ("table", [["Length of continuous service", "Annual leave", "Maximum single block"],
                   ["Less than 5 years", "24 working days", "15 working days"],
                   ["5 years or more", "30 working days", "20 working days"]]),
    ]),
    ("3. Requesting Annual Leave", [
        "Leave requests must be submitted through the HR portal at least 14 calendar days in advance for absences of up to "
        "5 working days, and at least 30 calendar days in advance for longer absences. Requests are approved by the employee's "
        "line manager; leave during the annual budget close (1 to 15 December) additionally requires Head of Department approval.",
        "Leave may not be taken before it is approved. Unapproved absence is treated as unpaid leave and may lead to disciplinary action.",
    ]),
    ("4. Carry-Over and Encashment", [
        "Up to 5 working days of unused annual leave may be carried over to the next calendar year and must be used by 31 March; "
        "carried-over days not used by that date are forfeited. Leave cannot be exchanged for payment during employment.",
        "On resignation or termination, accrued but untaken leave is paid out at the employee's basic daily rate. Leave taken "
        "in excess of accrual at the leaving date is deducted from the final salary.",
    ]),
    PAGEBREAK,
    ("5. Sick Leave", [
        "Shore staff are entitled to 14 calendar days of paid outpatient sick leave per calendar year, and up to 60 calendar days "
        "of paid hospitalisation leave (inclusive of the 14 days). A medical certificate from a registered practitioner is required "
        "for any absence of 2 or more consecutive days, and for any sick day adjacent to a public holiday or annual leave.",
        "The employee must notify the line manager before 10:00 local time on the first day of absence.",
    ]),
    ("6. Parental Leave", [
        "Maternity leave: 16 weeks of paid leave for employees with at least 6 months of service, of which up to 4 weeks may be "
        "taken before the expected date of delivery. Paternity leave: 10 working days of paid leave, to be taken within 16 weeks "
        "of the birth. Adoption leave: 12 weeks of paid leave for the primary carer.",
        "Employees must give at least 8 weeks' written notice of intended maternity or adoption leave, supported by medical or agency documentation.",
    ]),
    ("7. Compassionate and Other Leave", [
        "Compassionate leave: 5 working days of paid leave on the death of an immediate family member (spouse, child, parent, "
        "sibling, parent-in-law) and 2 working days for other relatives. Marriage leave: 3 working days, once during employment.",
        "Unpaid leave of up to 30 calendar days per year may be granted at the Company's discretion after annual leave is exhausted. "
        "Study leave of up to 5 working days per year is available for examinations in Company-approved courses.",
    ]),
    ("8. Public Holidays", [
        "Shore staff observe the gazetted public holidays of their office location. Where a public holiday falls on a Saturday or "
        "Sunday, no replacement day is given unless local law requires it.",
    ]),
]

# --------------------------------------------------------------------------- CSL-02
CREW_SHORE_LEAVE = [
    ("1. Purpose", [
        "This policy implements the shore leave provisions of the Maritime Labour Convention 2006 (Regulation 2.4) for seafarers "
        f"serving on vessels managed by {CO}. Shore leave is granted to benefit the health and well-being of seafarers and is "
        "consistent with the operational requirements of the vessel.",
    ]),
    ("2. Eligibility for Shore Leave", [
        "Shore leave may be granted when the vessel is alongside a berth, or at anchor with a Company-approved launch service, "
        "for a scheduled port stay of more than 8 hours. Shore leave is always subject to the Master's approval, and seafarers "
        "on watch or on standby duty are not eligible during their duty period.",
        "At least one third of the crew, including one licensed deck officer and one licensed engineer officer, must remain on board at all times.",
        "Shore leave is not granted during cargo operations involving dangerous goods of IMO Class 1 (explosives), during bunkering, "
        "or when the vessel is under port state control detention.",
    ]),
    ("3. Return on Board", [
        "Crew members must return on board at least 2 hours before the scheduled departure time posted at the gangway, or by "
        "the time set by the Master, whichever is earlier. Missing the vessel's departure through late return is a serious "
        "breach of discipline; all costs of rejoining the vessel will be recovered from the seafarer.",
    ]),
    PAGEBREAK,
    ("4. Security Levels and High-Risk Ports", [
        "When the port facility is at ISPS Security Level 2, shore leave is restricted to daylight hours (07:00 to 19:00 local) "
        "and crew must go ashore in groups of at least 2 (buddy system). At ISPS Security Level 3, all shore leave is suspended.",
        "In ports listed as high-risk in the Company Port Advisory, a curfew of 22:00 local time applies and crew must carry the "
        "Company emergency contact card. The Master may cancel shore leave at any time if the security situation changes.",
    ]),
    ("5. Documents and Conduct Ashore", [
        "Crew going ashore must carry their shore pass or seafarer's identity document and must sign out and in at the gangway "
        "log. Seafarers must not visit areas the local agent has advised against.",
        "No seafarer may return on board with a blood alcohol concentration above 0.05%, and no alcohol may be consumed within "
        "4 hours before a scheduled watch. Breach of the drug and alcohol policy results in immediate disembarkation.",
    ]),
    ("6. Shore Leave Denied", [
        "If shore leave is refused by port authorities, the Master must record the reason in the official log book and inform "
        "the Company within 24 hours. The Company will report repeated denials to the flag State. Crew denied shore leave for "
        "more than 30 consecutive days may request an early sign-off review under CRT-03.",
    ]),
]

# --------------------------------------------------------------------------- CRT-03
CREW_ROTATION = [
    ("1. Scope", [
        f"This policy applies to all seafarers employed or managed by {CO} serving on board Company-managed vessels. It sets out "
        "tour lengths, leave accrual, sign-off and repatriation, and travel arrangements. Shore staff are covered by HRP-01.",
    ]),
    ("2. Tour Lengths and Rotation", [
        "Standard tour lengths are set by rank. A tour may be extended by up to 1 month by mutual agreement, but no seafarer may "
        "serve more than 11 months on board continuously, in line with MLC 2006 Standard A2.5.1.",
        ("table", [["Rank group", "Standard tour on board", "Standard leave ashore"],
                   ["Master, Chief Engineer", "3 months", "3 months"],
                   ["Senior officers (Chief Officer, 2nd Engineer)", "4 months", "2 months"],
                   ["Junior officers", "5 months", "2 months"],
                   ["Ratings", "6 months", "3 months"],
                   ["Cadets", "6 months", "as per training programme"]]),
    ]),
    ("3. Leave Accrual for Seafarers", [
        "Seafarers accrue paid leave at 2.5 calendar days for each month of service on board, which is the minimum required by MLC 2006. "
        "Leave pay is calculated on the seafarer's basic wage and is paid at sign-off together with the final wage account. "
        "Officers on permanent contracts accrue paid leave at 8 calendar days per month on board.",
    ]),
    PAGEBREAK,
    ("4. Relief and Sign-Off", [
        "The Crewing Department will give each seafarer at least 30 days' notice of the planned relief date. If a relief is "
        "delayed beyond the contract end date by more than 14 days, a delay allowance of USD 25 per day is paid.",
        "Early sign-off may be requested for compassionate reasons. On the death or serious illness of an immediate family "
        "member, the Company will arrange repatriation from the next practicable port, with travel to begin within 72 hours "
        "of the request where operationally possible. Compassionate repatriation is at Company cost.",
    ]),
    ("5. Repatriation and Travel", [
        "Repatriation is to the seafarer's place of engagement or home airport, at Company cost, on completion of the tour, on "
        "medical grounds, or on early termination by the Company.",
        "Travel is by economy class. Masters and Chief Engineers travel premium economy on flights longer than 8 hours, and "
        "business class on flights longer than 12 hours. A travel per diem of USD 40 per day is paid for each day of travel "
        "and for each night spent in a transit hotel.",
        "Excess baggage up to 10 kg is reimbursed against receipts; more requires prior approval from the Crewing Department.",
    ]),
    ("6. Medical Fitness on Joining", [
        "Every seafarer must hold a valid medical fitness certificate (ENG1 or flag-equivalent) with at least 3 months' remaining "
        "validity at the time of joining. Pre-joining drug and alcohol tests are mandatory.",
    ]),
]


def main() -> None:
    render(DOCS / "HRP-01_Shore_Staff_Leave_Policy.pdf", "HRP-01 Shore Staff Leave Policy",
           f"{CO} | Human Resources | Version 4.2 | Effective 1 January 2026 | Owner: Head of HR", SHORE_LEAVE_POLICY)
    render(DOCS / "CSL-02_Crew_Shore_Leave_and_Port_Conduct.pdf", "CSL-02 Crew Shore Leave and Port Conduct Policy",
           f"{CO} | Marine HR & Crewing | Version 2.0 | Effective 1 March 2026 | Owner: Director of Crewing", CREW_SHORE_LEAVE)
    render(DOCS / "CRT-03_Crew_Leave_Rotation_and_Travel.pdf", "CRT-03 Crew Leave, Rotation and Travel Policy",
           f"{CO} | Marine HR & Crewing | Version 3.1 | Effective 1 March 2026 | Owner: Director of Crewing", CREW_ROTATION)
    print(f"wrote 3 PDFs to {DOCS}")


if __name__ == "__main__":
    main()
