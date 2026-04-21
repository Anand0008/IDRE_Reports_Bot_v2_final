"""
SQL Writer Agent  (updated for Story 3.3 — Assumption Annotation)
1. Checks metric_cards.json for an exact/fuzzy NL trigger match (fast path).
2. Falls back to Gemini 3.1 Pro with schema context.

The LLM is instructed to append an ASSUMPTIONS: block after the SQL whenever it
makes any interpretive decision.  _parse_llm_response() splits the two parts.
"""
import json
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_settings
from state.context import GraphState

METRIC_CARDS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "metric_cards.json")

SYSTEM_PROMPT = """You are a MySQL expert writing SELECT queries for the IDRE (Independent Dispute Resolution Entity) platform — a healthcare dispute resolution system under the No Surprises Act.

=== CRITICAL DISPLAY RULES ===
1. NEVER return `case`.id as the dispute identifier — it is an internal UUID useless to users.
   ALWAYS SELECT `case`.shortId AS dispute_number. The UI displays this as "DISP-<shortId>".
   When a user searches by "DISP-XXXXXXX", filter: WHERE `case`.shortId = 'XXXXXXX' (strip the DISP- prefix).
2. When listing disputes, ALWAYS include: `case`.shortId AS dispute_number, `case`.status, `case`.createdAt.
3. When the query mentions an organization name (e.g. "UHC", "UnitedHealthcare", "HaloMD", "PacificHealth"):
   JOIN `organization` on the appropriate FK to filter by `organization`.name LIKE '%<name>%'.
   - For NIP org: JOIN `organization` nip_org ON `case`.nonInitiatingPartyOrganizationId = nip_org.id
   - For IP org: JOIN `organization` ip_org ON `case`.initiatingPartyOrganizationId = ip_org.id
   - For owner org: JOIN `organization` own_org ON `case`.ownedByOrganizationId = own_org.id
   - If unclear which party, search ALL org FKs with OR.
   Common org aliases: UHC = United Healthcare = UnitedHealthcare; HaloMD = Halo; CB = Capitol Bridge.
4. When the query asks about a person by name (arbitrator, specialist, staff):
   JOIN `user` u ON `case`.assignedToId = u.id WHERE u.name LIKE '%<name>%'.
   Only SELECT u.name and u.email — NEVER select password hashes or other sensitive user columns.
5. For NIP (non-initiating party) info: JOIN `organization` ON `case`.nonInitiatingPartyOrganizationId = `organization`.id.
   For IP (initiating party) info: JOIN `organization` ON `case`.initiatingPartyOrganizationId = `organization`.id.
   Also JOIN `case_party` if you need party contact details (email, phone, address):
   - IP contact: JOIN `case_party` ip ON `case`.initiatingPartyId = ip.id
   - NIP contact: JOIN `case_party` nip ON `case`.nonInitiatingPartyId = nip.id
   IMPORTANT: `case_party`.partyType values are 'PROVIDER' or 'HEALTH_PLAN' — NOT 'INITIATING'/'NON_INITIATING'.
   Do NOT filter by case_party.partyType to distinguish IP from NIP. The FK (initiatingPartyId vs nonInitiatingPartyId) already determines which party is which.
6. For "cases assigned to [name]" queries: active assignments = `case`.assignedToId IS NOT NULL AND status
   NOT IN terminal statuses. For "cases closed by [name]": use `case`.closed_by_user_id or
   case_action WHERE actionType='STATUS_CHANGED' AND toValue LIKE 'CLOSED%'.
7. Exclude soft-deleted data: dispute_line_items WHERE status = 'ACTIVE' (not 'REMOVED').
   case_note WHERE deletedAt IS NULL.

=== SQL RULES ===
8. Only write SELECT statements — no INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, EXEC, or CALL.
9. Always backtick table names. `case` is a MySQL reserved word — always backtick it.
10. Do NOT add a LIMIT clause unless the user explicitly asks for "top N" / "first N" / "latest N".
    The executor auto-caps in-memory results at 50,000 rows (safety), and CSV downloads go up to
    100,000 rows. The UI shows only the first 50 rows with the rest available via CSV.
    A manual LIMIT only artificially truncates the answer and should be avoided.
11. Use human-readable column aliases (e.g., AS dispute_number, AS org_name, AS total_amount).
12. Use ONLY tables and columns listed in the schema context.
    If a "VERIFIED COLUMNS" block is present, treat it as ground truth — those are the ONLY
    columns that exist. If it says a column "does NOT exist", DO NOT use that column under any
    circumstances. Using a non-existent column will cause a runtime error.

=== STATUS VALUES ===
Case statuses: INITIAL_ELIGIBILITY_REVIEW, PENDING_PAYMENTS, PENDING_SECOND_PAYMENT,
FINAL_ELIGIBILITY_REVIEW, PENDING_RFI, PENDING_INITIAL_RFI, FINAL_ELIGIBILITY_COMPLETED,
FINAL_DETERMINATION_PENDING, FINAL_DETERMINATION_RENDERED, PENDING_CLOSURE_PAYMENTS,
PENDING_ADMINISTRATIVE_CLOSURE, INELIGIBLE_PENDING_ADMIN_FEE, INELIGIBLE,
CLOSED_INITIATING_PARTY, CLOSED_NON_INITIATING_PARTY, CLOSED_SPLIT_DECISION,
CLOSED_DEFAULT, CLOSED_DEFAULT_IP, CLOSED_DEFAULT_NIP, CLOSED_ADMINISTRATIVE,
NOTICE_OF_DISMISSAL_NON_PAYMENT, PAYBACK_REQUEST, REOPENED_FOR_CORRECTION.

Terminal/closed statuses: CLOSED_DEFAULT, CLOSED_INITIATING_PARTY, CLOSED_NON_INITIATING_PARTY,
CLOSED_ADMINISTRATIVE, CLOSED_SPLIT_DECISION, NOTICE_OF_DISMISSAL_NON_PAYMENT,
CLOSED_DEFAULT_IP, CLOSED_DEFAULT_NIP, INELIGIBLE, FINAL_DETERMINATION_RENDERED.
IMPORTANT: FINAL_DETERMINATION_RENDERED is terminal even though it doesn't start with "CLOSED".
Do NOT use LIKE 'CLOSED%' for terminal status checks — always enumerate the full list above.

Ineligible statuses (for ineligibility rates): INELIGIBLE AND INELIGIBLE_PENDING_ADMIN_FEE.
Both must be counted. INELIGIBLE_PENDING_ADMIN_FEE means the case failed eligibility but
the $115 admin fee hasn't been collected yet — it's still an ineligible case.

=== TEAM PERFORMANCE / ROLE RULES ===
When counting cases per user (team stats, arbitrator performance, top closers):
- Only include users with roles: arbitrator, arbitrator-contractor, admin-support, eligibility-specialist.
- JOIN `user` u ON `case`.assignedToId = u.id (or closed_by_user_id for closures).
- Filter: u.role IN ('arbitrator', 'arbitrator-contractor', 'admin-support', 'eligibility-specialist').

=== CASE OUTCOME / DECISION RULES ===
For case outcomes (who won, win rates, decision quality, award breakdown):
- Use the `arbitration_decision` table, NOT case.status.
- JOIN `arbitration_decision` ad ON ad.caseId = `case`.id.
- Group by ad.awardRecipient: values are INITIATING_PARTY, NON_INITIATING_PARTY, SPLIT_DECISION.
- Do NOT infer outcomes from case.status (CLOSED_INITIATING_PARTY etc.) — it misses nuances.

=== CAPITOL BRIDGE / PAYOUT RULES ===
Capitol Bridge is identified by bankingSnapshot JSON field, NOT by payment.type:
- JSON_EXTRACT(p.bankingSnapshot, '$.accountHolderName') LIKE '%Capitol Bridge%'
  OR JSON_EXTRACT(p.bankingSnapshot, '$.recipientName') LIKE '%Capitol Bridge%'.
- Do NOT filter by payment.type = 'CAPITOL_BRIDGE_FEE' for identifying CB — that's the fee type, not the payee.

=== REVENUE / FEE RULES ===
For revenue and fee collection queries:
- Use payment.type = 'CASE_PAYMENT' (NOT direction = 'INCOMING').
- direction = 'INCOMING' includes other payment types and overcounts revenue.

=== PAYMENT KNOWLEDGE ===
- The payment type column is named `type` (NOT paymentType). Use p.type for filtering.
  Values: CASE_PAYMENT (incoming fees), REFUND_TO_PREVAILING_PARTY/PARTY_REFUND_IP/PARTY_REFUND_NIP (outgoing refunds),
  CAPITOL_BRIDGE_FEE/THIRD_PARTY_PAYMENT (internal payouts), CMS_INVOICE_PAYMENT/CMS_ADMIN_FEE_TRANSFER (CMS fees).
- Payment statuses: PENDING, ON_HOLD, APPROVED, COMPLETED, CANCELLED, FAILED.
- Payment directions: INCOMING (from parties), OUTGOING (to parties/internal).
- Join case to payments: `case_payment_allocation` cpa ON cpa.caseId = `case`.id, then `payment` p ON cpa.paymentId = p.id.
- cpa.partyType indicates which party the payment is for: 'INITIATING' or 'NON_INITIATING'.
- "Paid in full" for SINGLE/BUNDLED = total allocatedAmount >= 710.00 per party.
  For BATCHED = total allocatedAmount >= 910.00 per party (base, excluding surcharges).
  For paid-in-full queries: SELECT the SUM(cpa.allocatedAmount) aliased as ip_paid_amount / nip_paid_amount
  so the post-processor can add paid_in_full_ip/paid_in_full_nip flags automatically.
- P=0/P=1/P=2: count COMPLETED payments per case via case_payment_allocation JOIN payment.
- For NACHA processing dates: JOIN `nacha_batch` nb ON p.nachaBatchId = nb.id. Use nb.transmittedAt or nb.createdAt.
- Refund amounts: $595 full (winner), $297.50 split.
- case_refunds table: stores refund records. case_refunds.refundAmountCents is in CENTS (divide by 100 for dollars).
- Dispute types: SINGLE (1 line item), BUNDLED (2+ items), BATCHED (2-25+ items with surcharges).

=== CLARIFICATION & GLOSSARY ===
13. If query contains " — clarification: ", combine original + answer. Clarification takes priority.
14. If "GLOSSARY TERMS DETECTED" block exists, apply each SQL filter verbatim.
15. After SQL, list interpretive decisions under ASSUMPTIONS: heading.

Output format (when assumptions exist):
<SQL statement>

ASSUMPTIONS:
- <assumption 1>
- <assumption 2>

Output format (when no assumptions):
<SQL statement>

Schema context:
{schema_context}

{platform_context}

If prior query failed with this error, fix the query:
{error_context}
"""


_BREAKDOWN_WORDS = re.compile(
    r"\b(by|per|group|breakdown|split|each|list|show|which|who|detail|"
    r"organisation|organization|region|status|type|category|compare|"
    r"between|versus|vs|trend|over time|monthly|daily|weekly)\b",
    re.IGNORECASE,
)
_COUNT_INTENT = re.compile(
    r"^(how many|what is the (total|count|number)|count of|total number|"
    r"number of|how much|what('s| is) the)",
    re.IGNORECASE,
)


def _check_metric_cards(query: str) -> str:
    """
    Fast path: return a pre-verified SQL template when the query is clearly
    asking for a single aggregate metric with no breakdown or detail intent.

    Guards applied before matching:
    - Query must start with a count/total question word pattern.
    - Query must not contain breakdown/grouping/detail keywords.
    - Query must be short (≤ 12 words) — longer queries almost always need
      custom SQL (filters, joins, groupings).
    """
    if not os.path.exists(METRIC_CARDS_PATH):
        return None

    word_count = len(query.split())
    if word_count > 12:
        return None
    if not _COUNT_INTENT.match(query.strip()):
        return None
    if _BREAKDOWN_WORDS.search(query):
        return None

    with open(METRIC_CARDS_PATH) as f:
        cards = json.load(f)

    query_lower = query.lower()
    for metric in cards.get("metrics", []):
        for trigger in metric.get("nl_triggers", []):
            if trigger.lower() in query_lower:
                return metric["sql"]
    return None


def _parse_llm_response(raw: str) -> tuple[str, list[str]]:
    """
    Split the raw LLM output into (sql, assumptions).
    Looks for an ASSUMPTIONS: header anywhere after the first line.
    Returns (full_raw_stripped, []) if no ASSUMPTIONS block is found.
    """
    match = re.search(r"\nASSUMPTIONS\s*:\s*\n", raw, re.IGNORECASE)
    if not match:
        sql_part = raw.strip()
        # Strip trailing markdown fence if present
        sql_part = re.sub(r"\s*```\s*$", "", sql_part)
        return sql_part, []

    sql_part = raw[: match.start()].strip()
    # Strip trailing markdown fence from SQL if present
    sql_part = re.sub(r"\s*```\s*$", "", sql_part)

    assumptions_raw = raw[match.end() :].strip()

    assumptions = []
    for line in assumptions_raw.splitlines():
        cleaned = line.strip().lstrip("-").lstrip("*").strip()
        # Skip lines that are just markdown fences
        if cleaned and not cleaned.startswith("```"):
            assumptions.append(cleaned)

    return sql_part, assumptions


def _extract_token_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "input":  int(usage.get("input_tokens", 0)),
        "output": int(usage.get("output_tokens", 0)),
        "total":  int(usage.get("total_tokens", 0)),
    }


def _generate_sql_with_llm(
    query: str, schema_context: str, error_context: str = "",
    platform_context: str = ""
) -> tuple[str, list[str], dict]:
    """Returns (sql, assumptions, token_usage)."""
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0,
        google_api_key=settings.gemini_api_key,
    )
    system = SYSTEM_PROMPT.format(
        schema_context=schema_context,
        error_context=error_context or "None",
        platform_context=platform_context or "",
    )
    messages = [SystemMessage(content=system), HumanMessage(content=query)]
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = "".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
    raw = content.strip()

    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    sql, assumptions = _parse_llm_response(raw)
    return sql, assumptions, _extract_token_usage(response)


def sql_writer_node(state: GraphState) -> GraphState:
    # Use resolved_query if available, else fall back to raw query
    query = state.get("resolved_query") or state["user_query"]
    schema_context = state.get("schema_context", "")
    platform_context = state.get("platform_context", "")
    # Prefer debugger-analyzed retry_context over raw execution_error
    error_context = state.get("retry_context", "") or state.get("execution_error", "") or ""
    retry_count = state.get("retry_count", 0)

    # Fast path: metric card match (only on first attempt) — no assumptions
    if retry_count == 0:
        sql = _check_metric_cards(query)
        if sql:
            trace_entry = {
                "agent": "SQL Writer",
                "status": "ok",
                "summary": "Served from metric card (fast path — no LLM call needed)",
                "detail": [],
            }
            trace = state.get("agent_trace", []) + [trace_entry]
            return {**state, "generated_sql": sql, "assumptions": [], "agent_trace": trace}

    # LLM path
    sql, assumptions, tok = _generate_sql_with_llm(query, schema_context, error_context, platform_context)
    token_usage = dict(state.get("token_usage") or {})
    writer_key = "SQL Writer" if retry_count == 0 else f"SQL Writer (retry {retry_count})"
    token_usage[writer_key] = tok

    label = "Retry" if retry_count > 0 else "Attempt 1"
    detail = []
    if error_context:
        detail.append(f"Previous error: {error_context[:120]}")
    glossary_matches = state.get("glossary_matches", [])
    if glossary_matches:
        terms = [m["term"] for m in glossary_matches]
        detail.append(f"Glossary filters applied: {', '.join(terms)}")
    if assumptions:
        detail.append(f"{len(assumptions)} assumption(s) annotated")

    trace_entry = {
        "agent": "SQL Writer",
        "status": "ok",
        "summary": f"SQL generated via Gemini 3.1 Pro · {label}"
        + (f" · {len(assumptions)} assumption(s)" if assumptions else ""),
        "detail": detail,
    }
    trace = state.get("agent_trace", []) + [trace_entry]
    return {
        **state,
        "generated_sql":   sql,
        "assumptions":     assumptions,
        "agent_trace":     trace,
        "execution_error": None,
        "token_usage":     token_usage,
    }
