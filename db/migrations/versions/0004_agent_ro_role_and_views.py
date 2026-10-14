"""Create the read-only `apx_agent_ro` role and the three masked, joined views it may SELECT
from -- `v_agent_invoices`, `v_agent_vendor_accounts` and `v_agent_audit` -- so `apx.agent`
(ADR-0010 D2) never has a grant on a raw IBAN column, a payment-state table, or
anything outside these views. IBANs are masked to their last 4 characters inside the view
definitions (`'****' || right(iban, 4)`), never in application code, so the role's own grants
are the enforcement, not caller discipline.

Revision ID: 0004_agent_ro_role_and_views
Revises: 0003_ingested_documents
Create Date: 2026-09-22
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_agent_ro_role_and_views"
down_revision: str | None = "0003_ingested_documents"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

# The role is created without a password, so it cannot log in until migration
# 0013_agent_ro_password sets one from the AGENT_RO_PASSWORD env var (no credential
# in source).
_CREATE_ROLE_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'apx_agent_ro') THEN
        CREATE ROLE apx_agent_ro LOGIN;
    END IF;
END
$$
"""

_GRANT_CONNECT_SQL = """
DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO apx_agent_ro', current_database());
END
$$
"""
_GRANT_USAGE_SQL = "GRANT USAGE ON SCHEMA public TO apx_agent_ro"

# ------------------------------------------------------------------------------------------
# v_agent_invoices: one row per invoice. IBANs never leave the view unmasked; the document's
# own IBAN is compared to the vendor's current verified payee entirely in SQL and only the
# boolean result is exposed. `fields_json`/`proposals_json`/`reasons_json` are pre-aggregated
# so the agent role never needs a grant on `fields`, `proposals` or `decisions` directly.
# ------------------------------------------------------------------------------------------

_CREATE_V_AGENT_INVOICES_SQL = """
CREATE VIEW v_agent_invoices AS
SELECT
    i.id AS invoice_id,
    i.status AS status,
    i.route AS route,
    i.vendor_id AS vendor_id,
    v.name AS vendor_name,
    i.number AS invoice_number_raw,
    i.issue_date AS issue_date,
    i.due_date AS due_date,
    i.po_number AS po_number_raw,
    i.currency AS currency,
    i.subtotal AS subtotal,
    i.vat_total AS vat_total,
    i.total AS total,
    i.created_at AS received_at,
    CASE WHEN payee.iban IS NOT NULL THEN '****' || right(payee.iban, 4) END AS payee_masked,
    (i.iban IS NOT NULL AND payee.iban IS NOT NULL AND i.iban = payee.iban)
        AS document_iban_matches_payee,
    fld.fields_json AS fields_json,
    prop.proposals_json AS proposals_json,
    dec.reasons_json AS reasons_json
FROM invoices i
LEFT JOIN vendors v ON v.id = i.vendor_id
LEFT JOIN LATERAL (
    SELECT bd.iban
    FROM bank_details bd
    WHERE bd.vendor_id = i.vendor_id AND bd.is_current AND bd.verified
    ORDER BY bd.changed_at DESC
    LIMIT 1
) payee ON true
LEFT JOIN LATERAL (
    SELECT jsonb_agg(
        jsonb_build_object(
            'name', f2.name, 'value', f2.value, 'normalized_value', f2.normalized_value,
            'confidence', f2.confidence, 'source', f2.source
        )
    ) AS fields_json
    FROM (
        SELECT DISTINCT ON (f.name)
            f.name, f.value, f.normalized_value, f.confidence, f.source
        FROM fields f
        WHERE f.invoice_id = i.id
        ORDER BY f.name,
            CASE f.source
                WHEN 'human' THEN 0
                WHEN 'arbiter' THEN 1
                WHEN 'extractor_b' THEN 2
                ELSE 3
            END,
            f.created_at DESC
    ) f2
) fld ON true
LEFT JOIN LATERAL (
    SELECT jsonb_agg(
        jsonb_build_object(
            'id', p.id, 'kind', p.kind, 'status', p.status,
            'created_by', p.created_by, 'created_at', p.created_at
        )
    ) AS proposals_json
    FROM proposals p
    WHERE p.invoice_id = i.id AND p.status = 'pending'
) prop ON true
LEFT JOIN LATERAL (
    SELECT d.reasons AS reasons_json
    FROM decisions d
    WHERE d.invoice_id = i.id AND d.action = 'policy_route' AND d.reasons IS NOT NULL
    ORDER BY d.created_at DESC
    LIMIT 1
) dec ON true
"""

# ------------------------------------------------------------------------------------------
# v_agent_vendor_accounts: one row per current account ('account') plus one row per historical
# change ('change'), always with the vendor's own name/trn/payee so a vendor with zero bank
# accounts still resolves (LEFT JOIN from vendors, never from bank_details).
# ------------------------------------------------------------------------------------------

_CREATE_V_AGENT_VENDOR_ACCOUNTS_SQL = """
CREATE VIEW v_agent_vendor_accounts AS
SELECT
    v.id AS vendor_id,
    v.name AS vendor_name,
    v.trn AS vendor_trn,
    payee.masked AS payee_masked,
    'account'::text AS kind,
    CASE WHEN bd.iban IS NOT NULL THEN '****' || right(bd.iban, 4) END AS masked_iban,
    NULL::text AS old_masked_iban,
    bd.verified AS verified,
    bd.source AS source,
    bd.changed_by AS changed_by,
    bd.changed_at AS changed_at,
    bv.verified_at AS verified_at
FROM vendors v
LEFT JOIN bank_details bd ON bd.vendor_id = v.id AND bd.is_current
LEFT JOIN bank_verifications bv ON bv.id = bd.verification_id
LEFT JOIN LATERAL (
    SELECT '****' || right(bd2.iban, 4) AS masked
    FROM bank_details bd2
    WHERE bd2.vendor_id = v.id AND bd2.is_current AND bd2.verified
    ORDER BY bd2.changed_at DESC
    LIMIT 1
) payee ON true

UNION ALL

SELECT
    v.id AS vendor_id,
    v.name AS vendor_name,
    v.trn AS vendor_trn,
    payee.masked AS payee_masked,
    'change'::text AS kind,
    '****' || right(h.iban, 4) AS masked_iban,
    (
        SELECT '****' || right(h2.iban, 4)
        FROM bank_details_history h2
        WHERE h2.vendor_id = h.vendor_id AND h2.recorded_at < h.recorded_at
        ORDER BY h2.recorded_at DESC
        LIMIT 1
    ) AS old_masked_iban,
    h.verified AS verified,
    h.source AS source,
    h.changed_by AS changed_by,
    h.changed_at AS changed_at,
    NULL::timestamptz AS verified_at
FROM bank_details_history h
JOIN vendors v ON v.id = h.vendor_id
LEFT JOIN LATERAL (
    SELECT '****' || right(bd2.iban, 4) AS masked
    FROM bank_details bd2
    WHERE bd2.vendor_id = v.id AND bd2.is_current AND bd2.verified
    ORDER BY bd2.changed_at DESC
    LIMIT 1
) payee ON true
WHERE h.operation IN ('I', 'U')
"""

# ------------------------------------------------------------------------------------------
# v_agent_audit: a narrow pass-through of the append-only audit log; `explain_decision`
# filters facts down to AUDIT_FACT_KEYS in Python (ADR-0010 D1), not here.
# ------------------------------------------------------------------------------------------

_CREATE_V_AGENT_AUDIT_SQL = """
CREATE VIEW v_agent_audit AS
SELECT id AS audit_id, ts, actor, action, entity, payload_json
FROM audit_log
"""

_GRANT_SELECT_SQL = (
    "GRANT SELECT ON v_agent_invoices, v_agent_vendor_accounts, v_agent_audit TO apx_agent_ro"
)


def upgrade() -> None:
    op.execute(_CREATE_ROLE_SQL)
    op.execute(_GRANT_CONNECT_SQL)
    op.execute(_GRANT_USAGE_SQL)
    op.execute(_CREATE_V_AGENT_INVOICES_SQL)
    op.execute(_CREATE_V_AGENT_VENDOR_ACCOUNTS_SQL)
    op.execute(_CREATE_V_AGENT_AUDIT_SQL)
    op.execute(_GRANT_SELECT_SQL)


def downgrade() -> None:
    op.execute(
        "REVOKE SELECT ON v_agent_invoices, v_agent_vendor_accounts, v_agent_audit "
        "FROM apx_agent_ro"
    )
    op.execute("DROP VIEW IF EXISTS v_agent_audit")
    op.execute("DROP VIEW IF EXISTS v_agent_vendor_accounts")
    op.execute("DROP VIEW IF EXISTS v_agent_invoices")
    op.execute("REVOKE USAGE ON SCHEMA public FROM apx_agent_ro")
    op.execute(
        "DO $$ BEGIN EXECUTE format("
        "'REVOKE CONNECT ON DATABASE %I FROM apx_agent_ro', current_database()); END $$"
    )
    op.execute("DROP ROLE IF EXISTS apx_agent_ro")
