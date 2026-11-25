"""Synthesise the Demo 2 corpus: a spare-parts catalogue CSV, two supplier warranty/SLA PDFs and a
procurement procedure PDF for a fictional ship manager.

Catalogue items, supplier terms and the procedure were drafted by an LLM (Claude) from the prompts in
synth_prompts.md and frozen here as data. Stock quantities are drawn from a seeded RNG so the CSV is
reproducible.

    python demos/demo2_spares/synthesis/make_docs.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from pdf_render import PAGEBREAK, render  # noqa: E402

DOCS = HERE.parent / "docs"
KMS = "Kestrel Marine Supply Ltd."
HEP = "Halcyon Engine Parts Pte. Ltd."

# part_no, description, equipment, supplier, unit_price_usd, min_stock, lead_time_days, criticality
CATALOGUE = [
    ("HEP-ME-1001", "Fuel injection valve nozzle", "Main engine MAN B&W 6S50ME-C", HEP, 1840.00, 6, 21, "critical"),
    ("HEP-ME-1002", "Fuel pump plunger and barrel assembly", "Main engine MAN B&W 6S50ME-C", HEP, 6250.00, 2, 35, "critical"),
    ("HEP-ME-1003", "Exhaust valve spindle", "Main engine MAN B&W 6S50ME-C", HEP, 9400.00, 1, 42, "critical"),
    ("HEP-ME-1004", "Piston ring set (one cylinder)", "Main engine MAN B&W 6S50ME-C", HEP, 2380.00, 2, 28, "critical"),
    ("HEP-ME-1005", "Cylinder liner O-ring kit", "Main engine MAN B&W 6S50ME-C", HEP, 315.50, 6, 14, "high"),
    ("HEP-ME-1006", "Starting air valve overhaul kit", "Main engine MAN B&W 6S50ME-C", HEP, 1125.00, 2, 21, "high"),
    ("HEP-ME-1007", "Turbocharger bearing set", "Turbocharger ABB A170-L", HEP, 7820.00, 1, 49, "critical"),
    ("HEP-ME-1008", "Turbocharger nozzle ring", "Turbocharger ABB A170-L", HEP, 11900.00, 1, 56, "critical"),
    ("HEP-AE-2001", "Fuel injector complete", "Auxiliary engine Yanmar 6EY18ALW", HEP, 960.00, 4, 14, "high"),
    ("HEP-AE-2002", "Cylinder head gasket", "Auxiliary engine Yanmar 6EY18ALW", HEP, 142.75, 6, 10, "medium"),
    ("HEP-AE-2003", "Lube oil filter element", "Auxiliary engine Yanmar 6EY18ALW", HEP, 38.40, 24, 7, "medium"),
    ("HEP-AE-2004", "Fuel oil filter element", "Auxiliary engine Yanmar 6EY18ALW", HEP, 41.90, 24, 7, "medium"),
    ("HEP-AE-2005", "Cooling water pump mechanical seal", "Auxiliary engine Yanmar 6EY18ALW", HEP, 465.00, 3, 14, "high"),
    ("HEP-AE-2006", "Connecting rod bearing shell pair", "Auxiliary engine Yanmar 6EY18ALW", HEP, 288.00, 6, 21, "high"),
    ("HEP-AE-2007", "Governor actuator", "Auxiliary engine Yanmar 6EY18ALW", HEP, 4150.00, 1, 35, "critical"),
    ("HEP-PU-3001", "Fuel oil purifier bowl disc set", "Separator Alfa Laval S821", HEP, 2740.00, 1, 28, "high"),
    ("HEP-PU-3002", "Purifier major service kit", "Separator Alfa Laval S821", HEP, 1395.00, 2, 21, "high"),
    ("HEP-CO-3101", "Air compressor valve plate kit", "Main air compressor Sperre HV2/200", HEP, 690.00, 2, 21, "high"),
    ("HEP-CO-3102", "Air compressor piston ring set", "Main air compressor Sperre HV2/200", HEP, 355.00, 2, 14, "medium"),
    ("HEP-BO-3201", "Boiler burner nozzle", "Aux boiler Aalborg OL", HEP, 520.00, 3, 21, "high"),
    ("KMS-EL-4001", "Main switchboard air circuit breaker 1600A", "Main switchboard", KMS, 8650.00, 1, 42, "critical"),
    ("KMS-EL-4002", "Motor contactor 95A", "Engine room motor starters", KMS, 310.00, 4, 10, "high"),
    ("KMS-EL-4003", "LED floodlight 150W", "Deck lighting", KMS, 236.00, 6, 7, "low"),
    ("KMS-EL-4004", "Navigation light lamp 65W", "Navigation lights", KMS, 18.60, 20, 5, "high"),
    ("KMS-EL-4005", "Emergency generator battery 12V 200Ah", "Emergency generator", KMS, 545.00, 2, 10, "critical"),
    ("KMS-EL-4006", "Insulation resistance tester", "Workshop", KMS, 780.00, 1, 14, "low"),
    ("KMS-SA-5001", "Lifejacket SOLAS 150N", "Life-saving appliances", KMS, 64.00, 30, 7, "critical"),
    ("KMS-SA-5002", "Immersion suit SOLAS", "Life-saving appliances", KMS, 412.00, 6, 14, "critical"),
    ("KMS-SA-5003", "SCBA cylinder 6.8L carbon", "Fire-fighting equipment", KMS, 690.00, 4, 21, "critical"),
    ("KMS-SA-5004", "Fire hose 15m 45mm with couplings", "Fire-fighting equipment", KMS, 128.00, 10, 7, "high"),
    ("KMS-SA-5005", "CO2 extinguisher 5kg", "Fire-fighting equipment", KMS, 96.50, 8, 7, "high"),
    ("KMS-SA-5006", "Lifebuoy self-activating light", "Life-saving appliances", KMS, 74.00, 6, 7, "high"),
    ("KMS-DK-6001", "Mooring rope HMPE 64mm x 220m", "Mooring equipment", KMS, 9800.00, 1, 35, "high"),
    ("KMS-DK-6002", "Mooring winch brake lining set", "Mooring winch", KMS, 1460.00, 2, 28, "high"),
    ("KMS-DK-6003", "Hydraulic hose assembly 1/2in x 2m", "Deck cranes", KMS, 54.00, 12, 5, "medium"),
    ("KMS-DK-6004", "Crane wire rope 22mm x 120m", "Deck cranes", KMS, 3350.00, 1, 28, "critical"),
    ("KMS-DK-6005", "Pilot ladder 9m SOLAS", "Pilot transfer", KMS, 1180.00, 1, 21, "critical"),
    ("KMS-NV-7001", "Gyro compass sensitive element", "Navigation equipment", KMS, 14500.00, 1, 63, "critical"),
    ("KMS-NV-7002", "ECDIS backup UPS battery", "Navigation equipment", KMS, 395.00, 2, 14, "high"),
    ("KMS-NV-7003", "VHF handheld radio GMDSS", "Radio equipment", KMS, 610.00, 3, 10, "critical"),
]

WAREHOUSES = ["Singapore", "Rotterdam", "Fujairah"]


def write_csv() -> None:
    rng = random.Random(42)
    DOCS.mkdir(parents=True, exist_ok=True)
    with (DOCS / "spare_parts_catalogue.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part_no", "description", "equipment", "supplier", "unit_price_usd", "stock_qty", "min_stock",
                    "lead_time_days", "warehouse", "criticality"])
        for i, (pn, desc, eq, sup, price, mn, lead, crit) in enumerate(CATALOGUE):
            # ~1 in 4 items is at or below minimum stock so "what needs reordering" has real answers
            stock = rng.randint(0, mn) if i % 4 == 1 else rng.randint(mn + 1, mn * 3 + 2)
            w.writerow([pn, desc, eq, sup, f"{price:.2f}", stock, mn, lead, WAREHOUSES[i % 3], crit])


KESTREL = [
    ("1. Scope of these Terms", [
        f"These Warranty and Service Level Terms apply to all deck, safety, electrical and navigation spares supplied by {KMS} "
        "(the Supplier) to Bluewater Fleet Management Pte. Ltd. (the Buyer) under Frame Agreement KMS-FA-2026, valid from "
        "1 January 2026 to 31 December 2027. Part numbers supplied under these terms carry the prefix KMS.",
    ]),
    ("2. Warranty Period", [
        "Spares are warranted against defects in material and workmanship for 18 months from the date of delivery or 12 months "
        "from the date of installation on board, whichever expires first.",
        "Consumable and wear items - including lamps, batteries, hoses, ropes, wire ropes and brake linings - carry a reduced "
        "warranty of 6 months from delivery. Life-saving appliances carry the manufacturer's warranty, but not less than 24 months.",
        "The warranty does not cover damage caused by incorrect installation, use outside the manufacturer's specification, "
        "normal wear and tear, or repairs by persons not authorised by the Supplier.",
    ]),
    ("3. Warranty Claims", [
        "The Buyer must notify a defect in writing within 14 days of discovering it, quoting the purchase order number and part "
        "number and attaching photographs and the vessel's defect report. The Supplier will respond within 5 working days and, "
        "at its option, repair, replace or credit the defective item. Replacement parts are warranted for the remainder of the "
        "original warranty period or 6 months, whichever is longer.",
    ]),
    PAGEBREAK,
    ("4. Delivery Service Levels", [
        "Standard orders: delivery to the Buyer's warehouses in Singapore, Rotterdam or Fujairah within the catalogue lead time "
        "for the item. Urgent orders (vessel safety or class-critical items): dispatch within 24 hours and delivery to the vessel "
        "at any of those three hubs within 72 hours, subject to stock, at a surcharge of 20% on the item price.",
        ("table", [["Service level", "Commitment", "Surcharge"],
                   ["Standard", "Catalogue lead time to warehouse", "None"],
                   ["Urgent", "Dispatch 24 h, on board within 72 h at hub ports", "20% of item price"],
                   ["Out-of-hours (weekends)", "As urgent, dispatch within 36 h", "30% of item price"]]),
    ]),
    ("5. Late Delivery Credits", [
        "If a standard order is delivered after its committed date, the Supplier credits 1% of the late items' value for each "
        "day of delay, up to a maximum of 8%. If an urgent order misses the 72-hour commitment, the urgent surcharge is refunded in full.",
    ]),
    ("6. Commercial Terms", [
        "Minimum order value: USD 250 per purchase order. Payment terms: 45 days from invoice. Prices are fixed for the term "
        "of the frame agreement except for wire ropes and mooring ropes, which are re-priced quarterly. Returns of unused, "
        "undamaged items are accepted within 30 days of delivery subject to a restocking fee of 15%.",
    ]),
]

HALCYON = [
    ("1. Scope", [
        f"These terms govern supply of engine, turbocharger, separator, compressor and boiler spares by {HEP} (the Supplier) "
        "to Bluewater Fleet Management Pte. Ltd. (the Buyer) under Frame Agreement HEP-FA-2025, valid from 1 July 2025 to "
        "30 June 2027. Part numbers supplied under these terms carry the prefix HEP.",
    ]),
    ("2. Warranty", [
        "Genuine engine spares are warranted for 24 months from delivery or 12 months from installation, whichever comes first. "
        "Turbocharger parts are warranted for 12 months from installation, provided installation is supervised by an "
        "OEM-authorised service engineer.",
        "Filters, gaskets, O-rings and seal kits are consumables and are not covered by warranty beyond being free from defects on delivery.",
        "The warranty is void if the engine was operated on fuel outside ISO 8217 specification at the time of the failure.",
    ]),
    ("3. Claims Procedure", [
        "Warranty claims must be submitted within 30 days of the failure through the Supplier's claims portal, with the "
        "engine log extract, photographs and the failed part retained for inspection. The Supplier will confirm acceptance or "
        "rejection within 10 working days. Accepted claims are settled by replacement part delivered under standard service level.",
    ]),
    PAGEBREAK,
    ("4. Service Levels", [
        "Standard delivery: within the catalogue lead time to Singapore, Rotterdam or Fujairah. Emergency (vessel off-hire or "
        "main engine immobilised): the Supplier will dispatch from the nearest stock point within 48 hours and deliver to "
        "the vessel within 5 days worldwide, at a surcharge of 25% plus actual freight cost.",
        "Technical support: a duty engineer is available 24 hours a day, 7 days a week, and will respond to emergency calls within 2 hours.",
    ]),
    ("5. Late Delivery", [
        "For standard orders delivered late, the Supplier credits 0.5% of the order value per day of delay, up to a maximum of "
        "5%. No credit applies to emergency orders where the delay is caused by customs clearance.",
    ]),
    ("6. Commercial Terms", [
        "Minimum order value: USD 1,000. Free freight to the three hub warehouses for orders above USD 10,000. Payment terms: "
        "30 days from invoice. A 2% early-payment discount applies when payment is made within 10 days. Returns are accepted "
        "within 60 days only for items in original sealed packaging, with a restocking fee of 20%.",
    ]),
]

PROCEDURE = [
    ("1. Purpose", [
        "This procedure defines how spare parts for Company-managed vessels are requisitioned, approved, ordered and received. "
        "It applies to all technical superintendents, Chief Engineers and Masters.",
    ]),
    ("2. Reordering Spares", [
        "A requisition must be raised in the planned maintenance system when the stock quantity of an item falls to or below "
        "its minimum stock level. For items with criticality 'critical', the requisition must be raised within 24 hours and the "
        "superintendent informed the same day.",
        "Order quantities should restore stock to at least twice the minimum stock level, rounded up to the supplier's pack size.",
    ]),
    ("3. Approval Limits", [
        "Each purchase order must be approved according to its total value, including surcharges and freight:",
        ("table", [["Purchase order value (USD)", "Approver", "Quotations required"],
                   ["Up to 5,000", "Technical Superintendent", "1 (frame-agreement supplier)"],
                   ["5,000.01 to 25,000", "Fleet Technical Manager", "2"],
                   ["25,000.01 to 100,000", "Technical Director", "3"],
                   ["Above 100,000", "Managing Director", "3, plus budget review"]]),
        "Orders placed with a frame-agreement supplier (Kestrel Marine Supply or Halcyon Engine Parts) at frame prices need "
        "only one quotation regardless of value, but the approval limits above still apply.",
    ]),
    PAGEBREAK,
    ("4. Preferred Suppliers", [
        "Engine-room machinery spares (main and auxiliary engines, turbochargers, separators, compressors, boilers) are ordered "
        "from Halcyon Engine Parts. Deck, safety, electrical and navigation spares are ordered from Kestrel Marine Supply. "
        "Purchases from other suppliers require written justification.",
    ]),
    ("5. Urgent and Emergency Orders", [
        "Urgent or emergency service levels may only be used when the vessel's safety, class status or ability to trade is at "
        "risk. The Chief Engineer must state the reason on the requisition. If the approver for the order value is unavailable, "
        "the next higher approver may approve; outside office hours the Duty Superintendent may approve emergency orders up to USD 25,000.",
    ]),
    ("6. Receipt and Warranty Records", [
        "On delivery, the vessel or warehouse must inspect the goods within 3 working days and record the delivery date and, "
        "once fitted, the installation date in the planned maintenance system. These dates determine the warranty period, so "
        "missing records may invalidate a warranty claim.",
    ]),
]


def main() -> None:
    write_csv()
    render(DOCS / "Kestrel_Marine_Supply_Warranty_and_SLA.pdf", "Kestrel Marine Supply - Warranty and Service Level Terms",
           "Frame Agreement KMS-FA-2026 | Annex B | Issued 15 December 2025", KESTREL)
    render(DOCS / "Halcyon_Engine_Parts_Warranty_and_SLA.pdf", "Halcyon Engine Parts - Warranty and Service Terms",
           "Frame Agreement HEP-FA-2025 | Schedule 3 | Issued 20 June 2025", HALCYON)
    render(DOCS / "PROC-01_Spares_Procurement_Procedure.pdf", "PROC-01 Spares Procurement Procedure",
           "Bluewater Fleet Management Pte. Ltd. | Technical Department | Version 5.0 | Effective 1 February 2026", PROCEDURE)
    print(f"wrote catalogue ({len(CATALOGUE)} parts) and 3 PDFs to {DOCS}")


if __name__ == "__main__":
    main()
