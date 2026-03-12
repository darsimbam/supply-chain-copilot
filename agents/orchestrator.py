# agents/orchestrator.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import re

from services.openai_client import get_openai_client


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"


@dataclass
class Filters:
    category: Optional[str] = None
    supplier: Optional[str] = None
    sku: Optional[str] = None          # primary SKU (first mentioned)
    skus: Optional[List[str]] = None   # all SKUs when question mentions multiple
    week: Optional[str] = None
    plant: Optional[str] = None
    dc: Optional[str] = None
    region: Optional[str] = None


@dataclass
class CopilotResponse:
    question: str
    intent: str
    filters: Dict[str, Any]
    tools_used: List[str]
    evidence: Dict[str, Any]
    summary: str
    causes: List[str]
    actions: List[str]
    email: str
    risk_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Orchestrator:
    """
    Control tower for the Supply Chain Planner Copilot.

    Flow:
    1. Parse the user question into intent + filters
    2. Decide which tools to use
    3. Collect structured evidence from tools
    4. Send evidence to the LLM for business reasoning
    5. Return business-ready output
    """

    VALID_INTENTS = {
        "root_cause",
        "risk_check",
        "action_plan",
        "email_draft",
        "kpi_lookup",
        "comparison",
        "clarification_needed",
    }

    _CLARIFICATION_MESSAGE = (
        "I'm not sure I understood your question. "
        "Could you rephrase it or ask the supply chain planner directly?"
    )

    # Supply chain keywords — if none are present, the question is likely out of scope
    _SC_KEYWORDS = {
        "otif", "sku", "stock", "inventory", "supplier", "forecast", "demand",
        "sales", "order", "deliver", "fulfil", "fulfill", "delay", "risk",
        "cover", "lead time", "perform", "haircare", "skincare", "fragrance",
        "category", "product", "shipment", "po", "purchase", "reorder",
        "safety stock", "stockout", "overstock", "revenue", "bias",
    }

    def __init__(self) -> None:
        self.client = get_openai_client()
        self.system_prompt = self._load_system_prompt()

    def run(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()

        if not question:
            return self._build_empty_response("No question provided.").to_dict()

        question = self._normalize_question(question)
        filters, intent = self._parse_question(question)

        if intent == "clarification_needed":
            return self._build_empty_response(self._CLARIFICATION_MESSAGE).to_dict()

        tools_to_use = self._route_tools(intent=intent, question=question)
        evidence = self._collect_evidence(tools=tools_to_use, filters=filters)

        llm_output = self._generate_business_response(
            question=question,
            intent=intent,
            filters=filters,
            evidence=evidence,
        )

        response = CopilotResponse(
            question=question,
            intent=intent,
            filters=asdict(filters),
            tools_used=tools_to_use,
            evidence=evidence,
            summary=llm_output.get("summary", ""),
            causes=llm_output.get("causes", []),
            actions=llm_output.get("actions", []),
            email=llm_output.get("email", ""),
            risk_level=llm_output.get("risk_level", "Unknown"),
        )

        return response.to_dict()

    # Common typos and shorthand that appear in supply chain questions
    _TYPO_MAP = {
        "list selling": "least selling",
        "list sell": "least selling",
        "list performer": "worst performer",
        "list performing": "worst performing",
        "worse performing": "worst performing",
        "worse supplier": "worst supplier",
        "worse sku": "worst sku",
        "top risk": "highest risk",
        "most delay": "most delayed",
        "most performing": "best performing sku",
        "most perform": "best performing sku",
        "high performing": "best performing sku",
        "top performing": "best performing sku",
    }

    def _normalize_question(self, question: str) -> str:
        """Fix common typos before parsing and synthesis."""
        normalized = question
        lower = question.lower()
        for typo, fix in self._TYPO_MAP.items():
            if typo in lower:
                # Replace case-insensitively, preserving the rest of the string
                idx = lower.find(typo)
                normalized = normalized[:idx] + fix + normalized[idx + len(typo):]
                lower = normalized.lower()
        return normalized

    def _parse_question(self, question: str) -> tuple[Filters, str]:
        """
        Use the LLM to extract structured intent and filters.
        Falls back to rule-based parsing if needed.
        """
        parser_prompt = """
You are a supply chain intent parser.

Return valid JSON only in this structure:
{
  "intent": "root_cause" | "risk_check" | "action_plan" | "email_draft" | "kpi_lookup" | "comparison" | "clarification_needed",
  "filters": {
    "category": null,
    "supplier": null,
    "sku": null,
    "week": null,
    "plant": null,
    "dc": null,
    "region": null
  }
}

Guidance:
- "Why is OTIF down in Haircare this week?" -> root_cause
- "What is at risk next week?" -> risk_check
- "What should I do about supplier ABC?" -> action_plan
- "Draft an email to supplier ABC" -> email_draft
- "Show me OTIF for Haircare" -> kpi_lookup
- "Which SKU is performing best?" -> kpi_lookup
- "What is the best/worst/top/bottom SKU?" -> kpi_lookup
- "Show me fill rate / freight cost / cycle time" -> kpi_lookup
- "Compare Haircare vs Skincare OTIF" -> comparison
- Anything clearly unrelated to supply chain (weather, cooking, history, etc.) -> clarification_needed

Rules:
- Return JSON only
- Unknown values must be null
- Do not add explanations
- Any question mentioning SKU, OTIF, stock, inventory, supplier, forecast, order, sales, delivery, fill rate,
  freight, lead time, or any supply chain term is supply-chain related — use kpi_lookup as default for these,
  NOT clarification_needed
- Only use clarification_needed when the question has NO supply chain context whatsoever
"""

        try:
            raw_text = self.client.generate_text(
                user_input=question,
                system_prompt=parser_prompt,
                temperature=0.0,
            )
            parsed = self._safe_json_loads(raw_text)

            intent = parsed.get("intent", "root_cause")
            if intent not in self.VALID_INTENTS:
                intent = "root_cause"

            # Safety net: if LLM says clarification_needed but SC keywords are present,
            # override to kpi_lookup — the LLM is being overly conservative
            if intent == "clarification_needed":
                q_lower = question.lower()
                if any(kw in q_lower for kw in self._SC_KEYWORDS):
                    intent = "kpi_lookup"

            filters_dict = parsed.get("filters", {})
            # Normalise the primary SKU from LLM output
            if filters_dict.get("sku"):
                filters_dict["sku"] = self._normalize_sku(filters_dict["sku"])
            # Always extract the full SKU list directly from the question text
            all_skus = self._extract_skus_from_question(question)
            if len(all_skus) > 1:
                filters_dict["skus"] = all_skus
                filters_dict["sku"] = all_skus[0]
            elif all_skus and not filters_dict.get("sku"):
                filters_dict["sku"] = all_skus[0]
            filters = Filters(**filters_dict)

            return filters, intent

        except Exception:
            return self._fallback_parse_question(question)

    def _fallback_parse_question(self, question: str) -> tuple[Filters, str]:
        """
        Rule-based backup parser.
        """
        q = question.lower()

        # If the question has no supply chain keywords at all, ask for clarification
        if not any(kw in q for kw in self._SC_KEYWORDS):
            return Filters(), "clarification_needed"

        if "why" in q and "otif" in q:
            intent = "root_cause"
        elif "risk" in q or "at risk" in q or "stockout" in q:
            intent = "risk_check"
        elif "draft" in q and "email" in q:
            intent = "email_draft"
        elif "email" in q:
            intent = "email_draft"
        elif "action" in q or "what should i do" in q:
            intent = "action_plan"
        elif "compare" in q or "versus" in q or " vs " in q:
            intent = "comparison"
        else:
            intent = "kpi_lookup"

        category = self._extract_keyword(
            q,
            ["haircare", "beauty", "skincare", "fragrance", "personal care", "cosmetics"],
        )

        supplier = self._extract_after_keyword(q, "supplier")
        all_skus = self._extract_skus_from_question(question)
        week = self._extract_week(q)

        filters = Filters(
            category=category,
            supplier=supplier.upper() if supplier else None,
            sku=all_skus[0] if all_skus else None,
            skus=all_skus if len(all_skus) > 1 else None,
            week=week,
        )

        return filters, intent

    def _route_tools(self, intent: str, question: str) -> List[str]:
        """
        Decide which tools are required.
        """
        q = question.lower()
        tools: List[str] = []

        # Intent-based routing
        if intent in {"root_cause", "kpi_lookup", "action_plan", "comparison"}:
            tools.extend(["otif", "forecast", "inventory", "supplier", "segmentation"])
        elif intent in {"risk_check"}:
            tools.extend(["forecast", "inventory", "supplier", "segmentation"])
        elif intent == "email_draft":
            tools.extend(["otif", "supplier"])

        # Keyword overrides — ensure relevant tools are always included
        if "otif" in q and "otif" not in tools:
            tools.append("otif")
        if any(kw in q for kw in ["sku", "sales", "forecast", "demand", "perform"]) and "forecast" not in tools:
            tools.append("forecast")
        if any(kw in q for kw in ["stock", "inventory", "cover", "stockout", "risk",
                                    "sufficient", "healthy", "well stocked", "enough stock"]) and "inventory" not in tools:
            tools.append("inventory")
        if any(kw in q for kw in ["supplier", "delay", "lead time", "po", "order"]) and "supplier" not in tools:
            tools.append("supplier")
        if any(kw in q for kw in ["segment", "abc", "group a", "group b", "group c",
                                    "profitab", "priority", "target otif", "service target",
                                    "cost saving"]) and "segmentation" not in tools:
            tools.append("segmentation")
        if any(kw in q for kw in ["turns", "obsolete", "excess", "accuracy",
                                    "days of supply", "dos"]) and "inventory" not in tools:
            tools.append("inventory")
        if any(kw in q for kw in ["fill rate", "in-full", "in full", "shipped qty",
                                    "fulfillment"]) and "otif" not in tools:
            tools.append("otif")
        if any(kw in q for kw in ["freight", "shipping cost", "cycle time", "lead time variab",
                                    "variability"]) and "supplier" not in tools:
            tools.append("supplier")

        return list(dict.fromkeys(tools))

    def _collect_evidence(self, tools: List[str], filters: Filters) -> Dict[str, Any]:
        """
        Run the tool layer and collect structured evidence.
        When multiple SKUs are requested, evidence is collected per SKU and merged.
        """
        sku_list = filters.skus if filters.skus else ([filters.sku] if filters.sku else [None])

        if len(sku_list) > 1:
            # Collect evidence for each SKU separately and merge under a per-SKU key
            merged: Dict[str, Any] = {}
            for s in sku_list:
                sub_filters = Filters(
                    category=filters.category,
                    supplier=filters.supplier,
                    sku=s,
                    week=filters.week,
                )
                sub_evidence = self._collect_evidence_single(tools, sub_filters)
                merged[s] = sub_evidence
            return {"multi_sku": True, "skus": sku_list, "by_sku": merged}

        return self._collect_evidence_single(tools, filters)

    def _collect_evidence_single(self, tools: List[str], filters: Filters) -> Dict[str, Any]:
        """Collect evidence for a single filter context."""
        evidence: Dict[str, Any] = {}
        category = filters.category
        week = filters.week
        sku = filters.sku

        for tool_name in tools:
            try:
                if tool_name == "otif":
                    from tools.otif_tool import (
                        get_otif_trend, get_otif_drop,
                        get_sku_otif_ranking, get_otif_by_supplier,
                        get_fill_rate,
                    )
                    evidence["otif"] = {
                        "trend": get_otif_trend(category=category, sku=sku),
                        "drop": get_otif_drop(category=category, sku=sku),
                        "sku_ranking": get_sku_otif_ranking(category=category),
                        "by_supplier": get_otif_by_supplier(category=category, sku=sku),
                        "fill_rate": get_fill_rate(category=category, sku=sku),
                    }
                elif tool_name == "forecast":
                    from tools.forecast_tool import (
                        get_forecast_error, get_bias,
                        get_top_skus, get_top_skus_by_revenue,
                    )
                    evidence["forecast"] = {
                        "error": get_forecast_error(category=category, week=week, sku=sku),
                        "bias": get_bias(category=category, week=week, sku=sku),
                        "sku_performance": get_top_skus(category=category),
                        "revenue_ranking": get_top_skus_by_revenue(category=category, sku=sku),
                    }
                elif tool_name == "inventory":
                    from tools.inventory_tool import (
                        get_inventory_risk, get_stock_summary, get_healthy_skus,
                        get_inventory_turns, get_days_of_supply,
                        get_inventory_accuracy, get_excess_obsolete,
                    )
                    evidence["inventory"] = {
                        "risk": get_inventory_risk(category=category, week=week, sku=sku),
                        "summary": get_stock_summary(category=category, week=week, sku=sku),
                        "healthy_skus": get_healthy_skus(category=category, sku=sku),
                        "turns": get_inventory_turns(category=category, sku=sku),
                        "days_of_supply": get_days_of_supply(category=category, sku=sku),
                        "accuracy": get_inventory_accuracy(category=category, sku=sku),
                        "excess_obsolete": get_excess_obsolete(category=category, sku=sku),
                    }
                elif tool_name == "supplier":
                    from tools.supplier_tool import (
                        get_supplier_delays, get_supplier_performance, get_open_pos,
                        get_order_cycle_time, get_freight_cost, get_lead_time_variability,
                    )
                    evidence["supplier"] = {
                        "delays": get_supplier_delays(category=category, sku=sku),
                        "performance": get_supplier_performance(category=category, sku=sku),
                        "open_pos": get_open_pos(category=category, sku=sku),
                        "cycle_time": get_order_cycle_time(category=category, sku=sku),
                        "freight_cost": get_freight_cost(category=category, sku=sku),
                        "lead_time_variability": get_lead_time_variability(category=category),
                    }
                elif tool_name == "segmentation":
                    from tools.segmentation_tool import get_sku_segments, assess_otif_vs_target
                    evidence["segmentation"] = {
                        "abc_segments": get_sku_segments(category=category),
                        "otif_vs_target": assess_otif_vs_target(category=category),
                    }
                else:
                    evidence[tool_name] = {"error": f"Unknown tool: {tool_name}"}
            except Exception as exc:
                evidence[tool_name] = {
                    "error": f"{tool_name} tool failed",
                    "details": str(exc),
                }

        return evidence

    def _generate_business_response(
        self,
        question: str,
        intent: str,
        filters: Filters,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert structured tool output into business-ready response.
        """
        payload = {
            "question": question,
            "intent": intent,
            "filters": asdict(filters),
            "evidence": self._trim_evidence(evidence),
        }

        intent_focus = {
            "root_cause": "Identify root causes. Populate causes, actions, and risk_level.",
            "risk_check": "Assess supply chain risks. Populate causes, actions, and risk_level. Leave email empty.",
            "action_plan": "Recommend concrete actions. Populate actions and risk_level. Leave email empty unless asked.",
            "email_draft": "Write a professional email in the email field. Keep causes and actions minimal.",
            "kpi_lookup": "Summarize the KPI values directly. Populate summary and risk_level. Leave causes, actions, and email empty unless clearly supported.",
            "comparison": "Compare performance across categories or SKUs. Focus on differences in summary. Leave email empty.",
        }.get(intent, "")

        synthesis_prompt = f"""
You are a supply chain copilot.

Using the supplied evidence, return valid JSON only in this structure:
{{
  "summary": "short business-ready summary",
  "causes": ["cause 1", "cause 2"],
  "actions": ["action 1", "action 2"],
  "email": "short professional email draft",
  "risk_level": "High | Medium | Low"
}}

Rules:
- Be practical and concise
- Only mention issues supported by evidence
- If evidence is missing or weak, say that clearly
- Actions should be operationally realistic
- Email should be short and executive-friendly
- risk_level must be exactly one of: High, Medium, Low
- Set risk_level=High if OTIF < 85%, stockouts detected, or suppliers severely delayed
- Set risk_level=Medium if moderate issues but no immediate crisis
- Set risk_level=Low if performance looks healthy

Contradiction check — BEFORE stating a cause, verify it is consistent with ALL evidence:
- Do NOT cite "under-forecasting" or "poor forecast" as a cause if forecast_accuracy_pct > 90% or MAPE < 10%
- Do NOT cite "supplier delays" as a cause if days_late <= 0 or on_time_rate > 95%
- Do NOT cite "low inventory" as a cause if weeks_of_cover > 4 or inventory_status is "OK"/"Overstock"
- Do NOT cite "in_full failure" as a cause if in_full = True
- If the only evidence of a problem is one metric (e.g. in_full=False), name that metric specifically
  rather than inventing a chain of secondary causes not supported by the data
- When evidence clearly rules out a cause, explicitly exclude it from the causes list

Focus instruction: {intent_focus}

Scope guidance: If the question is unrelated to supply chain, too vague to answer from the evidence,
or the evidence is empty, set summary to exactly:
"I'm not sure I understood your question. Could you rephrase it or ask the supply chain planner directly?"
and set risk_level to "Unknown". Do not guess or invent data.

Ranking guidance: Evidence includes both top-performing and bottom-performing SKUs.
- If the question asks for "least", "lowest", "worst", "bottom", or "minimum", use the bottom/worst data.
- If it asks for "top", "best", "highest", or "most", use the top/best data.
- When the question is ambiguous, provide both.
"""

        raw_text = ""
        try:
            raw_text = self.client.generate_text(
                user_input=json.dumps(payload, default=str),
                system_prompt=self.system_prompt + "\n\n" + synthesis_prompt,
                temperature=0.2,
            )
            parsed = self._safe_json_loads(raw_text)

            return {
                "summary": parsed.get("summary", "No summary generated."),
                "causes": parsed.get("causes", []),
                "actions": parsed.get("actions", []),
                "email": parsed.get("email", ""),
                "risk_level": parsed.get("risk_level", "Unknown"),
            }

        except Exception as exc:
            # If the model returned usable text but not valid JSON, surface it as summary
            if raw_text and not raw_text.startswith("OpenAI API error"):
                return {
                    "summary": raw_text,
                    "causes": [],
                    "actions": [],
                    "email": "",
                }
            return {
                "summary": f"Could not generate business response: {str(exc)}",
                "causes": [],
                "actions": [],
                "email": "",
            }

    def _load_system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text(encoding="utf-8").strip()

        return (
            "You are a supply chain planner copilot. "
            "You help users interpret OTIF, forecast accuracy, inventory risks, "
            "and supplier performance using structured evidence."
        )

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        """
        Parse JSON safely even if wrapped in markdown fences or mixed with prose.
        """
        if not text or text.startswith("OpenAI API error"):
            raise ValueError(text or "Empty response from model")

        cleaned = text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract the first JSON object embedded anywhere in the text
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    @staticmethod
    def _normalize_sku(raw: str) -> str:
        """
        Normalise an extracted SKU token so it matches the CSV format (e.g. 'SKU35').
        - Strips whitespace and collapses internal spaces: 'SKU 35' → 'SKU35'
        - Bare numbers: '35' → 'SKU35'
        - Already correct: 'SKU35' → 'SKU35'
        """
        s = raw.strip().upper().replace(" ", "")
        if s.isdigit():
            s = "SKU" + s
        return s

    @staticmethod
    def _extract_skus_from_question(question: str) -> List[str]:
        """
        Extract and normalise ALL SKU references from free text.
        Returns a list (may be empty). Handles 'SKU35', 'SKU 35', 'sku35'.
        """
        matches = re.findall(r'\bsku\s*([0-9]+[a-zA-Z0-9_-]*)\b', question, re.IGNORECASE)
        return [Orchestrator._normalize_sku("SKU" + m) for m in matches]

    @staticmethod
    def _extract_keyword(text: str, keywords: List[str]) -> Optional[str]:
        for keyword in keywords:
            if keyword in text:
                return keyword
        return None

    @staticmethod
    def _extract_after_keyword(text: str, keyword: str) -> Optional[str]:
        pattern = rf"{keyword}\s+([a-zA-Z0-9_-]+)"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_week(text: str) -> Optional[str]:
        # Only extract explicit ISO dates or date-like strings (YYYY-MM-DD, YYYY-WNN).
        # Relative terms like "this week" / "next week" are intentionally ignored
        # because they cannot be reliably resolved to a CSV date value.
        patterns = [
            r"(\d{4}-W\d{2})",          # ISO week: 2024-W42
            r"(\d{4}-\d{2}-\d{2})",     # ISO date: 2024-10-14
            r"week\s*(\d{1,2})",        # "week 42" -> return as bare number for caller
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def _trim_evidence(evidence: Dict[str, Any], max_records: int = 10) -> Dict[str, Any]:
        """Truncate list fields recursively to prevent token overflow."""
        if not isinstance(evidence, dict):
            return evidence
        result: Dict[str, Any] = {}
        for k, v in evidence.items():
            if isinstance(v, dict):
                result[k] = Orchestrator._trim_evidence(v, max_records)
            elif isinstance(v, list) and len(v) > max_records:
                result[k] = v[:max_records]
                result[f"_{k}_note"] = f"Truncated to {max_records} of {len(v)} records"
            else:
                result[k] = v
        return result

    @staticmethod
    def _build_empty_response(message: str) -> CopilotResponse:
        return CopilotResponse(
            question="",
            intent="unknown",
            filters={},
            tools_used=[],
            evidence={},
            summary=message,
            causes=[],
            actions=[],
            email="",
            risk_level="Unknown",
        )


_orchestrator_instance: Optional[Orchestrator] = None


def run_copilot(question: str) -> Dict[str, Any]:
    """
    Simple wrapper for app.py or Streamlit.
    Reuses a single Orchestrator instance across calls to avoid reloading the system prompt.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance.run(question)