"""Redefine `v_agent_invoices` so the agent's payee is the one P1 chose (ADR-0014 §7): the
masked payee, its match and any refusal come from the newest `policy_route` decision, which the
workflow writes at routing (`apx.policy.interfaces.routing_record`), not from a raw IBAN compare
against the newest verified account. Schema only, no table change; the `apx_agent_ro` grant is
re-issued because the view is dropped and recreated.

Revision ID: 0008_agent_routed_payee
Revises: 0007_invoice_kind
Create Date: 2026-09-24
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008_agent_routed_payee"
down_revision: str | None = "0007_invoice_kind"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

_FIELDS_AND_PROPOSALS = """
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
"""

# The newest policy_route decision's reasons hold the PolicyReason items plus one
# {"payee": {...}} element: {"masked_iban", "match", "verified_count"} or {"refusal"}. Only a
# masked IBAN is ever stored there; the view re-masks it anyway, so the role's grants stay the
# enforcement (0004's rule). An invoice with no policy_route row yet has no payee (NULL).
_CREATE_V_AGENT_INVOICES_SQL = f"""
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
    CASE WHEN rs.payee ->> 'masked_iban' IS NOT NULL
        THEN '****' || right(rs.payee ->> 'masked_iban', 4) END AS payee_masked,
    CASE WHEN rs.payee ->> 'masked_iban' IS NOT NULL
        THEN (rs.payee ->> 'match') = 'matched' END AS document_iban_matches_payee,
    fld.fields_json AS fields_json,
    prop.proposals_json AS proposals_json,
    rs.reasons_json AS reasons_json,
    rs.payee ->> 'match' AS payee_match,
    rs.payee ->> 'refusal' AS payee_refusal
FROM invoices i
LEFT JOIN vendors v ON v.id = i.vendor_id
{_FIELDS_AND_PROPOSALS}
LEFT JOIN LATERAL (
    SELECT d.reasons
    FROM decisions d
    WHERE d.invoice_id = i.id AND d.action = 'policy_route' AND d.reasons IS NOT NULL
    ORDER BY d.created_at DESC, d.id DESC
    LIMIT 1
) dec ON true
LEFT JOIN LATERAL (
    SELECT
        jsonb_agg(e.value) FILTER (WHERE e.value -> 'payee' IS NULL) AS reasons_json,
        (array_agg(e.value -> 'payee') FILTER (WHERE e.value -> 'payee' IS NOT NULL))[1]
            AS payee
    FROM jsonb_array_elements(
        CASE WHEN jsonb_typeof(dec.reasons) = 'array' THEN dec.reasons ELSE '[]'::jsonb END
    ) e
) rs ON true
"""

# 0004's definition, restored on downgrade.
_CREATE_V_AGENT_INVOICES_0004_SQL = f"""
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
{_FIELDS_AND_PROPOSALS}
LEFT JOIN LATERAL (
    SELECT d.reasons AS reasons_json
    FROM decisions d
    WHERE d.invoice_id = i.id AND d.action = 'policy_route' AND d.reasons IS NOT NULL
    ORDER BY d.created_at DESC
    LIMIT 1
) dec ON true
"""

_GRANT_SQL = "GRANT SELECT ON v_agent_invoices TO apx_agent_ro"


def upgrade() -> None:
    op.execute("DROP VIEW v_agent_invoices")
    op.execute(_CREATE_V_AGENT_INVOICES_SQL)
    op.execute(_GRANT_SQL)


def downgrade() -> None:
    op.execute("DROP VIEW v_agent_invoices")
    op.execute(_CREATE_V_AGENT_INVOICES_0004_SQL)
    op.execute(_GRANT_SQL)
