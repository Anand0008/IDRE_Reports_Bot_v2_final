"""
Schema Mapper Agent
Finds the most relevant tables from schema_catalog.json for a given user query
using ChromaDB vector search with local sentence-transformers embeddings (no API key needed).
Also appends FK join paths between the matched tables (Story 1.3).
"""
import json
import os
import re
from typing import List
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from state.context import GraphState
from utils.join_graph import get_join_context, get_graph_stats
from utils.glossary_matcher import format_glossary_context

SCHEMA_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "schema_catalog.json")
CHROMA_PERSIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "schema_tables_v4"  # v4 = updated case_party/payment descriptions
TOP_K = 6

# Local model — downloads once (~90MB), runs fully offline after that
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Module-level singleton — built once per process, reused for every query
_collection_cache = None


# Business descriptions for IDRE tables — helps the LLM understand what each table represents
TABLE_DESCRIPTIONS = {
    "case": "Core dispute case entity. Each row is one IDR dispute filing. Key fields: status (case lifecycle), disputeType (SINGLE/BUNDLED/BATCHED), disputeNumber (7-digit ID), due_date, due_date_until_decision, createdAt, closureReason, assignedToId (arbitrator). Links to organizations via initiatingPartyOrganizationId and nonInitiatingPartyOrganizationId.",
    "case_action": "Audit trail of all case events/status changes. actionType='STATUS_CHANGED' tracks lifecycle transitions (fromValue → toValue). Also tracks NOTE_ADDED, DOCUMENT_UPLOADED, ASSIGNMENT_CHANGED, etc.",
    "case_party": "Contact information for a party in a case (name, address, phone, email, fax). partyType is 'PROVIDER' or 'HEALTH_PLAN'. To get IP contact: JOIN ON case.initiatingPartyId = case_party.id. To get NIP contact: JOIN ON case.nonInitiatingPartyId = case_party.id.",
    "case_payment_allocation": "Junction table linking payments to cases. Each row allocates an amount from a payment to a specific case+party. Key: caseId, paymentId, partyType, allocatedAmount.",
    "case_refunds": "Refund records tied to case outcomes. Tracks entity fee refunds to prevailing parties after arbitration decision.",
    "payment": "All financial transactions. Key fields: amount, status (PENDING/ON_HOLD/APPROVED/COMPLETED/CANCELLED/FAILED), direction (INCOMING/OUTGOING), type (CASE_PAYMENT/REFUND_TO_PREVAILING_PARTY/CAPITOL_BRIDGE_FEE/THIRD_PARTY_PAYMENT/CMS_INVOICE_PAYMENT/CMS_ADMIN_FEE_TRANSFER/PARTY_REFUND_IP/PARTY_REFUND_NIP), nachaBatchId (FK to nacha_batch), paidAt, processedAt.",
    "arbitration_decision": "Arbitration decisions per case. decisionType (CASE_LEVEL/LINE_ITEM_LEVEL), awardRecipient (INITIATING_PARTY/NON_INITIATING_PARTY/SPLIT_DECISION), renderedAt, reasoning.",
    "line_item_decision": "Per-line-item arbitration decisions. Links to arbitration_decision and dispute_line_items.",
    "dispute_line_items": "Individual line items within a dispute. Each represents a specific claim/charge being disputed.",
    "organization": "Organizations (healthcare providers and health plans). name, type, createdAt. Parent entity for parties in cases.",
    "member": "Organization members with roles (owner, admin, member). Links users to organizations.",
    "invoice": "Billing invoices for case fees. invoiceNumber, totalAmount, dueDate, status (PENDING/SENT/PAID/OVERDUE/CANCELLED).",
    "invoice_item": "Individual line items within an invoice. Links to cases via caseId.",
    "invoice_payment": "Payments against invoices. Tracks varianceType (OVERPAYMENT/UNDERPAYMENT/EXACT) and varianceAmount.",
    "cms_invoice": "CMS (Centers for Medicare & Medicaid) fee invoices. status (RECEIVED/VALIDATED/PROCESSED/DISCREPANCY/REJECTED).",
    "cms_invoice_payment_allocation": "Allocates CMS invoice payments to cases.",
    "nacha_batch": "ACH batch files for bulk payment processing. Groups multiple payments into a single NACHA file.",
    "bank_account": "Bank accounts for organizations. achStatus (PENDING/APPROVED/REJECTED/REQUIRES_VERIFICATION).",
    "bank_account_approval": "Approval workflow for bank account verification.",
    "payment_approval": "Approval records for individual payments.",
    "payment_reminder": "Payment reminder records sent to parties about pending fees.",
    "payment_reminder_log": "Log of reminder emails sent, tracking delivery status.",
    "email_job": "Email notification queue. Tracks all platform emails: type, recipient, status (pending/sent/failed).",
    "case_note": "Internal notes attached to cases by staff.",
    "case_document": "Documents uploaded or attached to cases.",
    "case_documentation_checklist": "Tracks completion of required documentation (Notice of Offer, NPI, TIN, etc.).",
    "case_contact": "Contact information for parties involved in a case.",
    "case_ach_info": "ACH payment information specific to a case.",
    "case_party_payment_lock": "Locks preventing duplicate payment processing for a case party.",
    "global_organization_member": "Cross-organization role assignments for admin users.",
    "invoice_number_audit_log": "Audit log for invoice number generation and changes.",
}


def _build_table_document(table_name: str, table_info: dict) -> str:
    parts = [f"Table: {table_name}"]

    # Add business description if available
    desc = TABLE_DESCRIPTIONS.get(table_name)
    if desc:
        parts.append(f"Description: {desc}")

    cols = table_info.get("columns", [])
    if cols:
        col_strs = [f"{c['name']} ({c['type']})" for c in cols]
        parts.append("Columns: " + ", ".join(col_strs))

    fks = table_info.get("foreign_keys", [])
    if fks:
        fk_strs = [
            f"{fk['column']} -> {fk['references_table']}.{fk['references_column']}"
            for fk in fks
        ]
        parts.append("Foreign keys: " + ", ".join(fk_strs))

    sample = table_info.get("sample_values", {})
    if sample:
        sample_strs = []
        for col, vals in list(sample.items())[:4]:
            if vals:
                sample_strs.append(f"{col}: {', '.join(str(v) for v in vals[:3])}")
        if sample_strs:
            parts.append("Sample values — " + "; ".join(sample_strs))

    return "\n".join(parts)


def _get_collection():
    global _collection_cache
    if _collection_cache is not None:
        return _collection_cache

    os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
        if collection.count() > 0:
            _collection_cache = collection
            return _collection_cache
    except Exception:
        pass

    # Build collection from schema_catalog
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    with open(SCHEMA_CATALOG_PATH) as f:
        catalog = json.load(f)

    docs, ids, metadatas = [], [], []
    for table_name, table_info in catalog["tables"].items():
        doc = _build_table_document(table_name, table_info)
        docs.append(doc)
        ids.append(table_name)
        metadatas.append({
            "table_name": table_name,
            "row_count": table_info.get("row_count_approx", 0),
        })

    collection.upsert(documents=docs, ids=ids, metadatas=metadatas)
    _collection_cache = collection
    return _collection_cache


def get_relevant_tables(query: str) -> List[str]:
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=TOP_K)
    return results["ids"][0]


def build_schema_context(table_names: List[str]) -> str:
    with open(SCHEMA_CATALOG_PATH) as f:
        catalog = json.load(f)

    parts = []
    for name in table_names:
        info = catalog["tables"].get(name)
        if not info:
            continue
        parts.append(_build_table_document(name, info))

    schema_text = "\n\n".join(parts)

    # Story 1.3 — append FK join paths between the matched tables
    join_hints = get_join_context(table_names)
    if join_hints:
        schema_text += "\n\n" + join_hints

    return schema_text


_ORG_NAME_PATTERN = re.compile(
    r"\b(uhc|united\s*health\s*care|unitedhealth|halomd|halo\s*md|pacifichealth|"
    r"pacific\s*health|capitol\s*bridge|veratru|vera\s*tru|aetna|cigna|anthem|"
    r"humana|bcbs|blue\s*cross|kaiser|molina|centene|radix|"
    r"organization|org\s+name)\b",
    re.IGNORECASE,
)
_PERSON_NAME_PATTERN = re.compile(
    r"\b(assigned\s+to|closed\s+by|arbitrator|specialist|"
    r"[A-Z][a-z]+\s+[A-Z][a-z]+)\b",  # Matches "Tina Fields", "Siobhan Dubin" etc.
)
_DISPUTE_NUM_PATTERN = re.compile(r"\bDISP-\w+\b", re.IGNORECASE)
_PAYMENT_PATTERN = re.compile(
    r"\b(payment|paid|unpaid|refund|amount|fee|invoice|nacha|ach|"
    r"disbursement|payout|allocation|balance|fund)\b",
    re.IGNORECASE,
)
_NIP_IP_PATTERN = re.compile(
    r"\b(NIP|non.initiating|initiating\s+party|IP\s+|health\s+plan|provider|respondent|"
    r"filing\s+party|claimant)\b",
    re.IGNORECASE,
)


def _detect_intent_tables(query: str) -> List[str]:
    """Detect query intent and return tables that MUST be included."""
    forced = []

    # Always include case for any dispute-related query
    forced.append("case")

    # Organization name mentioned → force org + case_party
    if _ORG_NAME_PATTERN.search(query):
        forced.extend(["organization", "case_party"])

    # Person name or assignment query → force user
    if _PERSON_NAME_PATTERN.search(query):
        forced.append("user")

    # Dispute number mentioned → case is sufficient (shortId)
    if _DISPUTE_NUM_PATTERN.search(query):
        forced.append("case_party")
        forced.append("organization")

    # Payment-related query → force payment tables
    if _PAYMENT_PATTERN.search(query):
        forced.extend(["payment", "case_payment_allocation"])

    # NIP/IP party query → force party + org tables
    if _NIP_IP_PATTERN.search(query):
        forced.extend(["case_party", "organization"])

    return list(set(forced))


def schema_mapper_node(state: GraphState) -> GraphState:
    query = state.get("resolved_query") or state["user_query"]
    tables = get_relevant_tables(query)

    # Enforce role permissions: drop any table not in permitted_tables
    permitted = state.get("permitted_tables", [])
    if permitted:
        blocked = [t for t in tables if t not in permitted]
        tables = [t for t in tables if t in permitted]
    else:
        blocked = []

    # Force-include tables based on query intent detection
    intent_forced = []
    for tbl in _detect_intent_tables(query):
        if tbl not in tables and (not permitted or tbl in permitted):
            tables = tables + [tbl]
            intent_forced.append(tbl)

    # Force-include any tables required by matched glossary terms
    glossary_matches = state.get("glossary_matches", [])
    glossary_forced = []
    for match in glossary_matches:
        if match.get("requires_join") and match.get("join_table"):
            join_tbl = match["join_table"]
            if join_tbl not in tables and (not permitted or join_tbl in permitted):
                tables = tables + [join_tbl]
                glossary_forced.append(join_tbl)
        for tbl in match.get("applies_to_tables", []):
            if tbl not in tables and (not permitted or tbl in permitted):
                tables = tables + [tbl]
                glossary_forced.append(tbl)

    schema_ctx = build_schema_context(tables)

    # Append glossary filter context block for the SQL Writer
    glossary_block = format_glossary_context(glossary_matches)
    if glossary_block:
        schema_ctx = schema_ctx + "\n\n" + glossary_block

    # Count how many join paths were found
    join_hints = get_join_context(tables)
    join_count = join_hints.count("↔") if join_hints else 0
    stats = get_graph_stats()

    detail = list(tables)
    if blocked:
        detail.append(f"Permission-blocked tables (not shown to LLM): {', '.join(blocked)}")
    if intent_forced:
        detail.append(f"Intent-forced tables: {', '.join(set(intent_forced))}")
    if glossary_forced:
        detail.append(f"Glossary forced tables: {', '.join(set(glossary_forced))}")
    if join_count:
        detail.append(f"FK graph: {stats['fk_edges']} edges across {stats['tables']} tables")

    summary = f"Matched {len(tables)} tables via semantic search · {join_count} join path(s) resolved"
    if blocked:
        summary += f" · {len(blocked)} table(s) blocked by role"
    if intent_forced:
        summary += f" · {len(set(intent_forced))} intent table(s) forced"
    if glossary_forced:
        summary += f" · {len(set(glossary_forced))} glossary table(s) forced"

    trace_entry = {
        "agent": "Schema Mapper",
        "status": "ok",
        "summary": summary,
        "detail": detail,
    }
    trace = state.get("agent_trace", []) + [trace_entry]
    return {**state, "relevant_tables": tables, "schema_context": schema_ctx, "agent_trace": trace}
