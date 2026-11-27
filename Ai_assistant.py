"""Simple AI Business Intelligence Report Assistant Model.

Zero-dependency Python model that:
1. Ingests sales & business reports (CSV, JSON, TXT, Markdown).
2. Indexes report chunks and structured records for TF-IDF information retrieval.
3. Summarizes performance metrics and key findings.
4. Solves the Challenge: identifies the best-performing region and explains the
   reasons using quantitative drivers and retrieved qualitative evidence.
5. Generates actionable recommendations grounded in the uploaded reports.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "into", "over", "under", "than",
    "then", "so", "if", "not", "no", "can", "could", "will", "would", "should",
    "have", "has", "had", "do", "does", "did", "we", "our", "they", "their", "which",
    "what", "when", "where", "why", "how", "who", "all", "more", "most", "other",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization with stopword removal."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def parse_number(value: Any) -> float | None:
    """Parse currency, percentage, or numeric string into float."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[$€£,\s]", "", text)
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def match_column(headers: list[str], candidates: tuple[str, ...]) -> str | None:
    """Find the best matching header name from candidate aliases."""
    normalized = {h.strip().lower().replace(" ", "_"): h for h in headers}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    for cand in candidates:
        for norm_h, orig_h in normalized.items():
            if cand in norm_h:
                return orig_h
    return None


@dataclass
class DocumentChunk:
    """Indexed snippet from an uploaded report for information retrieval."""

    doc_name: str
    section: str
    text: str
    tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(f"{self.section} {self.text}")


class BusinessIntelligenceModel:
    """Simple RAG-style Business Intelligence Report Assistant."""

    REGION_ALIASES = ("region", "territory", "area", "market", "geo", "geography", "zone")
    REVENUE_ALIASES = ("revenue", "sales", "total_sales", "gross_sales", "amount", "turnover")
    PROFIT_ALIASES = ("profit", "net_profit", "operating_profit", "margin_usd", "earnings", "ebitda")
    COST_ALIASES = ("cost", "cogs", "expense", "expenses", "spend")
    UNITS_ALIASES = ("units", "units_sold", "quantity", "qty", "volume", "orders")
    GROWTH_ALIASES = ("growth", "yoy_growth", "growth_pct", "growth_rate", "qoq_growth")
    CATEGORY_ALIASES = ("category", "product", "product_line", "segment", "line", "sku")
    CHANNEL_ALIASES = ("channel", "sales_channel", "source")
    PERIOD_ALIASES = ("quarter", "period", "month", "date", "year")

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[DocumentChunk] = []
        self.records: list[dict[str, Any]] = []
        self.idf: dict[str, float] = {}

    def clear(self) -> None:
        """Reset all ingested reports and indices."""
        self.documents.clear()
        self.chunks.clear()
        self.records.clear()
        self.idf.clear()

    def ingest_file(self, file_path: str | Path) -> dict[str, Any]:
        """Load and parse a report file from disk."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(path.name, content)

    def ingest_text(self, filename: str, content: str) -> dict[str, Any]:
        """Ingest a report given its filename and text content."""
        ext = Path(filename).suffix.lower()
        doc_info = {
            "filename": filename,
            "type": ext.lstrip(".") or "txt",
            "char_count": len(content),
            "records_added": 0,
            "chunks_added": 0,
        }

        if ext in (".csv", ".tsv"):
            delimiter = "\t" if ext == ".tsv" else ","
            added_records, added_chunks = self._parse_delimited(filename, content, delimiter)
        elif ext == ".json":
            added_records, added_chunks = self._parse_json(filename, content)
        else:
            added_records, added_chunks = self._parse_narrative(filename, content)

        doc_info["records_added"] = added_records
        doc_info["chunks_added"] = added_chunks
        self.documents.append(doc_info)
        self._rebuild_index()
        return doc_info

    def _parse_delimited(self, filename: str, content: str, delimiter: str) -> tuple[int, int]:
        reader = csv.DictReader(io.StringIO(content.strip()), delimiter=delimiter)
        if not reader.fieldnames:
            return 0, 0

        headers = list(reader.fieldnames)
        col_map = {
            "region": match_column(headers, self.REGION_ALIASES),
            "revenue": match_column(headers, self.REVENUE_ALIASES),
            "profit": match_column(headers, self.PROFIT_ALIASES),
            "cost": match_column(headers, self.COST_ALIASES),
            "units": match_column(headers, self.UNITS_ALIASES),
            "growth": match_column(headers, self.GROWTH_ALIASES),
            "category": match_column(headers, self.CATEGORY_ALIASES),
            "channel": match_column(headers, self.CHANNEL_ALIASES),
            "period": match_column(headers, self.PERIOD_ALIASES),
        }

        records_added = 0
        chunks_added = 0
        for idx, row in enumerate(reader, start=1):
            if not any(str(v).strip() for v in row.values() if v is not None):
                continue

            revenue = parse_number(row.get(col_map["revenue"])) if col_map["revenue"] else None
            profit = parse_number(row.get(col_map["profit"])) if col_map["profit"] else None
            cost = parse_number(row.get(col_map["cost"])) if col_map["cost"] else None
            if profit is None and revenue is not None and cost is not None:
                profit = revenue - cost

            record = {
                "source": filename,
                "row_num": idx,
                "region": (row.get(col_map["region"]) or "Unspecified").strip() if col_map["region"] else "Unspecified",
                "revenue": revenue,
                "profit": profit,
                "cost": cost,
                "units": parse_number(row.get(col_map["units"])) if col_map["units"] else None,
                "growth": parse_number(row.get(col_map["growth"])) if col_map["growth"] else None,
                "category": (row.get(col_map["category"]) or "General").strip() if col_map["category"] else "General",
                "channel": (row.get(col_map["channel"]) or "All Channels").strip() if col_map["channel"] else "All Channels",
                "period": (row.get(col_map["period"]) or "Current").strip() if col_map["period"] else "Current",
                "raw": row,
            }
            self.records.append(record)
            records_added += 1

            summary_parts = [f"{k}: {v}" for k, v in row.items() if v is not None and str(v).strip()]
            self.chunks.append(
                DocumentChunk(
                    doc_name=filename,
                    section=f"Row {idx} ({record['region']} / {record['category']})",
                    text="; ".join(summary_parts),
                    metadata={"region": record["region"], "type": "table_row"},
                )
            )
            chunks_added += 1

        return records_added, chunks_added

    def _parse_json(self, filename: str, content: str) -> tuple[int, int]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._parse_narrative(filename, content)

        rows: list[dict[str, Any]] = []
        if isinstance(data, list):
            rows = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            for key in ("records", "sales", "data", "regions", "rows"):
                if isinstance(data.get(key), list):
                    rows = [item for item in data[key] if isinstance(item, dict)]
                    break
            if not rows:
                rows = [data]

        if not rows:
            return self._parse_narrative(filename, content)

        headers = list({k for row in rows for k in row.keys()})
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})
        return self._parse_delimited(filename, buf.getvalue(), ",")

    def _parse_narrative(self, filename: str, content: str) -> tuple[int, int]:
        lines = content.splitlines()
        sections: list[tuple[str, list[str]]] = []
        current_section = "Overview"
        buffer: list[str] = []

        for line in lines:
            stripped = line.strip()
            heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
            caps_heading = (
                len(stripped) > 3
                and len(stripped) < 70
                and stripped.endswith(":")
                and not any(ch.isdigit() for ch in stripped[:4])
            )
            if heading_match:
                if buffer:
                    sections.append((current_section, buffer))
                    buffer = []
                current_section = heading_match.group(2).strip()
            elif caps_heading:
                if buffer:
                    sections.append((current_section, buffer))
                    buffer = []
                current_section = stripped.rstrip(":")
            elif stripped == "" and len("\n".join(buffer)) > 450:
                sections.append((current_section, buffer))
                buffer = []
            elif stripped:
                buffer.append(stripped)

        if buffer:
            sections.append((current_section, buffer))

        chunks_added = 0
        for sec_title, sec_lines in sections:
            text = "\n".join(sec_lines).strip()
            if not text:
                continue
            self.chunks.append(
                DocumentChunk(
                    doc_name=filename,
                    section=sec_title,
                    text=text,
                    metadata={"type": "narrative"},
                )
            )
            chunks_added += 1

        return 0, chunks_added

    def _rebuild_index(self) -> None:
        num_docs = len(self.chunks)
        if num_docs == 0:
            self.idf = {}
            return
        df: Counter[str] = Counter()
        for chunk in self.chunks:
            df.update(set(chunk.tokens))
        self.idf = {
            term: math.log((num_docs + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant report chunks using TF-IDF cosine + phrase boost."""
        if not self.chunks:
            return []

        q_tokens = tokenize(query)
        q_lower = query.lower().strip()
        if not q_tokens:
            return [
                {
                    "doc_name": c.doc_name,
                    "section": c.section,
                    "text": c.text,
                    "score": 0.0,
                    "type": c.metadata.get("type", "text"),
                }
                for c in self.chunks[:top_k]
            ]

        q_tf = Counter(q_tokens)
        q_vec = {t: count * self.idf.get(t, 1.0) for t, count in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in self.chunks:
            if not chunk.tokens:
                continue
            c_tf = Counter(chunk.tokens)
            dot = sum(q_vec[t] * (c_tf[t] * self.idf.get(t, 1.0)) for t in q_vec if t in c_tf)
            if dot == 0:
                continue
            c_norm = math.sqrt(sum((cnt * self.idf.get(t, 1.0)) ** 2 for t, cnt in c_tf.items())) or 1.0
            score = dot / (q_norm * c_norm)

            # Small boost for narrative context or exact phrase match
            text_lower = f"{chunk.section} {chunk.text}".lower()
            if q_lower and q_lower in text_lower:
                score += 0.25
            if chunk.metadata.get("type") == "narrative":
                score *= 1.15

            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "doc_name": chunk.doc_name,
                "section": chunk.section,
                "text": chunk.text,
                "score": round(score, 4),
                "type": chunk.metadata.get("type", "text"),
            }
            for score, chunk in scored[:top_k]
        ]

    def aggregate_regions(self) -> list[dict[str, Any]]:
        """Aggregate quantitative performance metrics by region."""
        if not self.records:
            return self._extract_regions_from_narrative()

        grouped: dict[str, dict[str, Any]] = {}
        for rec in self.records:
            reg = rec["region"]
            if reg not in grouped:
                grouped[reg] = {
                    "region": reg,
                    "revenue": 0.0,
                    "profit": 0.0,
                    "cost": 0.0,
                    "units": 0.0,
                    "growth_values": [],
                    "records": 0,
                    "categories": defaultdict(float),
                    "channels": defaultdict(float),
                    "periods": defaultdict(float),
                }
            bucket = grouped[reg]
            rev = rec["revenue"] or 0.0
            prof = rec["profit"] or 0.0
            cost = rec["cost"] or 0.0
            units = rec["units"] or 0.0

            bucket["revenue"] += rev
            bucket["profit"] += prof
            bucket["cost"] += cost
            bucket["units"] += units
            bucket["records"] += 1
            if rec["growth"] is not None:
                bucket["growth_values"].append(rec["growth"])
            bucket["categories"][rec["category"]] += rev if rev > 0 else 1.0
            bucket["channels"][rec["channel"]] += rev if rev > 0 else 1.0
            bucket["periods"][rec["period"]] += rev if rev > 0 else 1.0

        total_rev = sum(b["revenue"] for b in grouped.values()) or 1.0
        results: list[dict[str, Any]] = []
        for reg, b in grouped.items():
            rev = b["revenue"]
            prof = b["profit"]
            margin_pct = (prof / rev * 100.0) if rev > 0 else 0.0
            avg_growth = (
                sum(b["growth_values"]) / len(b["growth_values"])
                if b["growth_values"]
                else None
            )
            top_category = max(b["categories"].items(), key=lambda x: x[1])[0] if b["categories"] else "N/A"
            top_channel = max(b["channels"].items(), key=lambda x: x[1])[0] if b["channels"] else "N/A"
            rev_share = (rev / total_rev) * 100.0

            # Composite performance score (weighted revenue share, profit margin, and growth)
            score = (rev_share * 0.50) + (max(margin_pct, 0.0) * 0.30) + ((avg_growth or 0.0) * 0.40)

            results.append({
                "region": reg,
                "revenue": round(rev, 2),
                "profit": round(prof, 2),
                "margin_pct": round(margin_pct, 2),
                "units": round(b["units"], 2),
                "avg_growth_pct": round(avg_growth, 2) if avg_growth is not None else None,
                "revenue_share_pct": round(rev_share, 2),
                "top_category": top_category,
                "top_channel": top_channel,
                "record_count": b["records"],
                "performance_score": round(score, 2),
                "category_breakdown": {k: round(v, 2) for k, v in sorted(b["categories"].items(), key=lambda x: -x[1])},
            })

        results.sort(key=lambda r: (r["revenue"], r["profit"], r["performance_score"]), reverse=True)
        return results

    def _extract_regions_from_narrative(self) -> list[dict[str, Any]]:
        """Fallback region extraction when only narrative text reports are uploaded."""
        region_mentions: dict[str, list[float]] = defaultdict(list)
        pattern = re.compile(
            r"\b(North America|Europe|EMEA|APAC|Asia[- ]Pacific|Latin America|LATAM|Middle East|Africa|South America)\b",
            re.IGNORECASE,
        )
        money_pattern = re.compile(r"\$([0-9]+(?:\.[0-9]+)?)\s*(M|K|million|thousand)?", re.IGNORECASE)

        for chunk in self.chunks:
            for line in chunk.text.splitlines():
                reg_match = pattern.search(line)
                if reg_match:
                    region = reg_match.group(1)
                    money_match = money_pattern.search(line)
                    val = 0.0
                    if money_match:
                        val = float(money_match.group(1))
                        unit = (money_match.group(2) or "").lower()
                        if unit in ("m", "million"):
                            val *= 1_000_000
                        elif unit in ("k", "thousand"):
                            val *= 1_000
                    region_mentions[region].append(val)

        results = []
        for reg, vals in region_mentions.items():
            est_rev = max(vals) if vals else 0.0
            results.append({
                "region": reg,
                "revenue": round(est_rev, 2),
                "profit": 0.0,
                "margin_pct": 0.0,
                "units": 0.0,
                "avg_growth_pct": None,
                "revenue_share_pct": 0.0,
                "top_category": "Extracted from narrative",
                "top_channel": "Extracted from narrative",
                "record_count": len(vals),
                "performance_score": round(est_rev, 2),
                "category_breakdown": {},
            })
        results.sort(key=lambda r: r["revenue"], reverse=True)
        return results

    def identify_best_region(self) -> dict[str, Any]:
        """Challenge: Identify the best-performing region and explain why from report content."""
        regions = self.aggregate_regions()
        if not regions:
            return {
                "best_region": None,
                "explanation": "No regional performance data found in the uploaded reports.",
                "reasons": [],
                "evidence": [],
                "ranking": [],
            }

        best = regions[0]
        runner_up = regions[1] if len(regions) > 1 else None
        reg_name = best["region"]

        reasons: list[str] = []
        if best["revenue"] > 0:
            lead_msg = (
                f"Highest revenue at ${best['revenue']:,.2f} "
                f"({best['revenue_share_pct']:.1f}% of total company revenue)"
            )
            if runner_up and runner_up["revenue"] > 0:
                lead_pct = ((best["revenue"] - runner_up["revenue"]) / runner_up["revenue"]) * 100
                lead_msg += f", outperforming runner-up {runner_up['region']} (${runner_up['revenue']:,.2f}) by {lead_pct:.1f}%."
            else:
                lead_msg += "."
            reasons.append(lead_msg)

        if best["profit"] > 0:
            reasons.append(
                f"Strong profitability generating ${best['profit']:,.2f} in profit "
                f"at a {best['margin_pct']:.1f}% profit margin."
            )

        if best["avg_growth_pct"] is not None:
            reasons.append(
                f"Consistent expansion with an average growth rate of {best['avg_growth_pct']:+.1f}%."
            )

        if best["top_category"] not in ("General", "N/A", "Extracted from narrative"):
            top_cat_rev = best["category_breakdown"].get(best["top_category"], 0.0)
            cat_share = (top_cat_rev / best["revenue"] * 100.0) if best["revenue"] > 0 else 0.0
            reasons.append(
                f"Anchored by top product/category '{best['top_category']}' "
                f"(${top_cat_rev:,.2f}, {cat_share:.1f}% of regional revenue) via {best['top_channel']}."
            )

        # Retrieve qualitative narrative reasons mentioning the winning region
        narrative_hits = [
            hit for hit in self.retrieve(f"{reg_name} growth driver reason strong performance", top_k=6)
            if hit["type"] == "narrative" and reg_name.lower() in f"{hit['section']} {hit['text']}".lower()
        ]
        if not narrative_hits:
            narrative_hits = self.retrieve(f"{reg_name} performance", top_k=3)

        qualitative_reasons: list[str] = []
        for hit in narrative_hits[:3]:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", hit["text"]):
                clean = sentence.strip(" -•\t")
                if len(clean) > 25 and reg_name.lower() in clean.lower():
                    if clean not in qualitative_reasons:
                        qualitative_reasons.append(f"[{hit['doc_name']} — {hit['section']}] {clean}")
                    if len(qualitative_reasons) >= 3:
                        break

        summary_line = (
            f"{reg_name} is the best-performing region overall"
            + (f", generating ${best['revenue']:,.2f} in revenue and ${best['profit']:,.2f} in profit." if best["revenue"] > 0 else ".")
        )

        return {
            "best_region": reg_name,
            "metrics": best,
            "explanation": summary_line,
            "reasons": reasons + qualitative_reasons,
            "evidence": narrative_hits[:3],
            "ranking": regions,
        }

    def summarize_performance(self) -> dict[str, Any]:
        """Generate an executive summary of KPIs and key findings."""
        if not self.documents:
            return {
                "has_data": False,
                "kpis": {},
                "findings": ["Upload one or more sales or business reports to generate insights."],
            }

        total_rev = sum(r["revenue"] or 0.0 for r in self.records)
        total_prof = sum(r["profit"] or 0.0 for r in self.records)
        total_units = sum(r["units"] or 0.0 for r in self.records)
        growth_vals = [r["growth"] for r in self.records if r["growth"] is not None]
        avg_growth = sum(growth_vals) / len(growth_vals) if growth_vals else None
        overall_margin = (total_prof / total_rev * 100.0) if total_rev > 0 else 0.0

        by_category: dict[str, float] = defaultdict(float)
        by_channel: dict[str, float] = defaultdict(float)
        by_period: dict[str, float] = defaultdict(float)
        for r in self.records:
            rev = r["revenue"] or 0.0
            by_category[r["category"]] += rev
            by_channel[r["channel"]] += rev
            by_period[r["period"]] += rev

        top_cat = max(by_category.items(), key=lambda x: x[1]) if by_category else ("N/A", 0.0)
        top_chan = max(by_channel.items(), key=lambda x: x[1]) if by_channel else ("N/A", 0.0)
        regions = self.aggregate_regions()

        findings: list[str] = []
        if total_rev > 0:
            findings.append(
                f"Total tracked revenue is ${total_rev:,.2f} with ${total_prof:,.2f} net profit "
                f"({overall_margin:.1f}% overall margin) across {int(total_units):,} units."
            )
        if avg_growth is not None:
            findings.append(f"Average regional/segment growth across reported periods is {avg_growth:+.1f}%.")
        if top_cat[0] not in ("General", "N/A"):
            findings.append(
                f"Top-performing category is '{top_cat[0]}', contributing ${top_cat[1]:,.2f} "
                f"({(top_cat[1] / total_rev * 100) if total_rev else 0:.1f}% of revenue)."
            )
        if top_chan[0] not in ("All Channels", "N/A"):
            findings.append(
                f"Primary revenue channel is '{top_chan[0]}' generating ${top_chan[1]:,.2f}."
            )
        if len(regions) > 1:
            top_r, low_r = regions[0], regions[-1]
            findings.append(
                f"Regional spread: {top_r['region']} leads (${top_r['revenue']:,.2f}), while "
                f"{low_r['region']} trails (${low_r['revenue']:,.2f})."
            )

        # Pull key executive bullet highlights from narrative docs
        narrative_highlights = self.retrieve("key findings executive summary performance risk", top_k=3)
        for hit in narrative_highlights:
            if hit["type"] == "narrative":
                first_line = hit["text"].splitlines()[0].strip(" -•")
                if len(first_line) > 20:
                    findings.append(f"[{hit['doc_name']}] {first_line}")

        return {
            "has_data": True,
            "kpis": {
                "documents_loaded": len(self.documents),
                "records_analyzed": len(self.records),
                "indexed_chunks": len(self.chunks),
                "total_revenue": round(total_rev, 2),
                "total_profit": round(total_prof, 2),
                "profit_margin_pct": round(overall_margin, 2),
                "total_units": round(total_units, 2),
                "avg_growth_pct": round(avg_growth, 2) if avg_growth is not None else None,
            },
            "top_category": {"name": top_cat[0], "revenue": round(top_cat[1], 2)},
            "top_channel": {"name": top_chan[0], "revenue": round(top_chan[1], 2)},
            "by_category": {k: round(v, 2) for k, v in sorted(by_category.items(), key=lambda x: -x[1])},
            "by_period": {k: round(v, 2) for k, v in by_period.items()},
            "findings": findings,
        }

    def generate_recommendations(self) -> list[dict[str, str]]:
        """Generate prioritized business recommendations grounded in report data."""
        if not self.documents:
            return []

        regions = self.aggregate_regions()
        summary = self.summarize_performance()
        recs: list[dict[str, str]] = []

        if regions:
            best = regions[0]
            recs.append({
                "priority": "High",
                "title": f"Replicate {best['region']}'s Playbook Across Other Markets",
                "rationale": (
                    f"{best['region']} achieved the highest revenue (${best['revenue']:,.2f}) "
                    f"and {best['margin_pct']:.1f}% margin led by '{best['top_category']}' on '{best['top_channel']}'."
                ),
                "action": (
                    f"Roll out {best['region']}'s {best['top_channel']} go-to-market strategy and "
                    f"{best['top_category']} bundling to under-penetrated territories."
                ),
            })

            if len(regions) > 1:
                laggard = min(regions, key=lambda r: (r["margin_pct"], r["revenue"]))
                recs.append({
                    "priority": "High",
                    "title": f"Address Margin & Growth Gap in {laggard['region']}",
                    "rationale": (
                        f"{laggard['region']} trails at ${laggard['revenue']:,.2f} revenue "
                        f"and {laggard['margin_pct']:.1f}% profit margin."
                    ),
                    "action": (
                        f"Audit discounting, fulfillment costs, and channel mix in {laggard['region']}; "
                        f"shift budget toward higher-margin offerings."
                    ),
                })

        top_cat = summary.get("top_category", {})
        if top_cat.get("name") not in (None, "General", "N/A"):
            recs.append({
                "priority": "Medium",
                "title": f"Double Down on Top Category: {top_cat['name']}",
                "rationale": (
                    f"'{top_cat['name']}' is the #1 revenue contributor at ${top_cat['revenue']:,.2f}."
                ),
                "action": (
                    f"Prioritize inventory allocation, upsell campaigns, and R&D investment for {top_cat['name']}."
                ),
            })

        # Scan narrative chunks for explicit risks/challenges/opportunities
        risk_hits = self.retrieve("recommendation risk challenge opportunity bottleneck churn cost", top_k=3)
        for hit in risk_hits:
            if hit["type"] == "narrative":
                snippet = hit["text"].splitlines()[0][:220]
                recs.append({
                    "priority": "Medium",
                    "title": f"Act on Report Insight ({hit['section']})",
                    "rationale": f"From {hit['doc_name']}: \"{snippet}\"",
                    "action": "Assign an owner to review this operational finding in the next quarterly business review.",
                })
                break

        return recs

    def answer_question(self, query: str) -> dict[str, Any]:
        """Retrieve relevant business info and synthesize a grounded answer."""
        q_lower = query.lower()
        hits = self.retrieve(query, top_k=5)

        if any(kw in q_lower for kw in ("best", "top", "highest", "leading", "winner")) and "region" in q_lower:
            best = self.identify_best_region()
            answer_lines = [best["explanation"], ""]
            answer_lines.extend(f"- {r}" for r in best["reasons"])
            return {
                "query": query,
                "answer": "\n".join(answer_lines).strip(),
                "evidence": best["evidence"] or hits,
                "best_region": best["best_region"],
            }

        if any(kw in q_lower for kw in ("recommend", "action", "improve", "strategy", "next step")):
            recs = self.generate_recommendations()
            answer_lines = ["Recommended Actions Based on Uploaded Reports:"]
            for rec in recs:
                answer_lines.append(f"- [{rec['priority']}] {rec['title']}: {rec['action']} ({rec['rationale']})")
            return {
                "query": query,
                "answer": "\n".join(answer_lines),
                "evidence": hits,
            }

        if any(kw in q_lower for kw in ("summary", "summarize", "overview", "kpi", "overall")):
            summary = self.summarize_performance()
            answer_lines = ["Performance Summary:"]
            answer_lines.extend(f"- {f}" for f in summary["findings"])
            return {
                "query": query,
                "answer": "\n".join(answer_lines),
                "evidence": hits,
            }

        if not hits:
            return {
                "query": query,
                "answer": "No matching information was found in the uploaded reports. Try uploading a report or rephrasing your question.",
                "evidence": [],
            }

        lines = ["Most relevant findings retrieved from your reports:"]
        for hit in hits[:4]:
            preview = hit["text"].replace("\n", " ")
            if len(preview) > 260:
                preview = preview[:257] + "..."
            lines.append(f"- [{hit['doc_name']} | {hit['section']}] {preview}")

        return {
            "query": query,
            "answer": "\n".join(lines),
            "evidence": hits,
        }

    def full_analysis(self) -> dict[str, Any]:
        """Run the complete BI pipeline: summary, regional challenge, and recommendations."""
        return {
            "documents": self.documents,
            "summary": self.summarize_performance(),
            "best_region_analysis": self.identify_best_region(),
            "recommendations": self.generate_recommendations(),
        }


def main() -> None:
    """Simple CLI entry point for running the BI assistant on local report files."""
    import argparse

    parser = argparse.ArgumentParser(description="AI Business Intelligence Report Assistant")
    parser.add_argument("files", nargs="*", help="Paths to sales/business report files (.csv, .json, .txt, .md)")
    parser.add_argument("-q", "--query", help="Optional business question to answer")
    args = parser.parse_args()

    model = BusinessIntelligenceModel()
    target_files = args.files
    if not target_files:
        sample_dir = Path(__file__).parent / "samples"
        if sample_dir.exists():
            target_files = [str(p) for p in sorted(sample_dir.iterdir()) if p.is_file()]

    if not target_files:
        print("No report files provided and no samples/ directory found.")
        return

    for f in target_files:
        info = model.ingest_file(f)
        print(f"Loaded {info['filename']} ({info['records_added']} records, {info['chunks_added']} chunks)")

    if args.query:
        res = model.answer_question(args.query)
        print("\n=== QUERY ANSWER ===")
        print(res["answer"])
        return

    analysis = model.full_analysis()
    print("\n=== EXECUTIVE SUMMARY ===")
    for finding in analysis["summary"]["findings"]:
        print(f"- {finding}")

    best = analysis["best_region_analysis"]
    print("\n=== CHALLENGE: BEST-PERFORMING REGION ===")
    print(best["explanation"])
    for reason in best["reasons"]:
        print(f"  * {reason}")

    print("\n=== RECOMMENDATIONS ===")
    for rec in analysis["recommendations"]:
        print(f"- [{rec['priority']}] {rec['title']}: {rec['action']}")


if __name__ == "__main__":
    main()
