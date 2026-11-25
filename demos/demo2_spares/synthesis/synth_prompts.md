# Demo 2 - synthesis prompts

The catalogue CSV and the three PDFs are **LLM-synthesised**. The prompts below were given to an LLM (Claude); the
output was reviewed for consistency (part-number prefixes match suppliers, the procedure's preferred-supplier rule
matches the catalogue, no figure stated twice with different values) and frozen as data in `make_docs.py`.
Stock quantities come from a seeded RNG (`random.Random(42)`). All companies, agreements and prices are fictional.

## Shared system prompt

> You produce realistic procurement documents for a fictional ship manager, "Bluewater Fleet Management", which
> buys spares under two frame agreements: Kestrel Marine Supply (deck, safety, electrical, navigation) and
> Halcyon Engine Parts (engine room machinery). Use concrete, checkable figures (months, days, hours, percentages,
> USD amounts). Supplier terms must differ from each other on every commercial point so that comparison questions
> have non-trivial answers. Do not mention freight rates, currency exchange rates, or market prices - those are
> deliberately out of scope.

## Prompt 1 - spare parts catalogue

> List 40 spare parts for a bulk carrier with a MAN B&W 6S50ME-C main engine, ABB A170-L turbocharger, Yanmar
> 6EY18ALW auxiliary engines, Alfa Laval S821 separator, Sperre compressor, Aalborg boiler, plus deck, safety,
> electrical and navigation equipment. For each: part number (prefix HEP- for engine room, KMS- for the rest),
> description, equipment, supplier, unit price in USD, minimum stock, lead time in days, criticality
> (critical/high/medium/low). Prices must be plausible relative to each other.

## Prompt 2 - Kestrel Marine Supply warranty & SLA

> Write the warranty and service-level annex of Kestrel's frame agreement (2 pages): scope and validity, warranty
> period (delivery vs installation, whichever first), reduced warranty for consumables and wear items, LSA warranty,
> exclusions, claims procedure with deadlines, delivery service levels with a table (standard / urgent / out-of-hours
> with surcharges), late-delivery credits, minimum order, payment terms, price validity, returns and restocking fee.

## Prompt 3 - Halcyon Engine Parts warranty & service terms

> Same structure for Halcyon (engine spares), but with different numbers on every point, an emergency service level
> for off-hire vessels, a turbocharger-specific warranty condition, a fuel-specification warranty exclusion, a 24/7
> duty engineer, free-freight threshold and early-payment discount.

## Prompt 4 - PROC-01 Spares Procurement Procedure

> Write the ship manager's internal procurement procedure (2 pages): reorder trigger tied to minimum stock and
> criticality, reorder quantity rule, approval limits by PO value as a table with quotation requirements, the
> frame-agreement exception, preferred supplier by equipment category, urgent/emergency order rules and out-of-hours
> approval, and receipt/installation date recording (linking to warranty).

## Design intent

- **CSV + PDFs**: answers need table lookups (`csv_lookup`) and arithmetic (`calculator`: order totals, surcharges,
  approval tier) - the factory must attach both, and must NOT attach weather.
- **Cross-document synthesis**: e.g. "who approves an urgent order for 2 turbocharger nozzle rings?" needs the
  catalogue price, Halcyon's emergency surcharge, and PROC-01's approval table.
- **A deliberate gap**: the brief asks for live freight rates, which nothing supports - the guard must flag it.
