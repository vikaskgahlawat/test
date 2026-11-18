import json
from datetime import datetime, timezone

from GZ.SparkUtils import sparkSession
from pyspark.sql.types import (
    StringType,
    StructField,
    StructType,
)

# Connections and target
AGENT_CONNECTION = "AgentTestConn"
AGENT_ID = "1a37e97d-893f-47ea-af39-ef5ba85b6128"

S3_CONNECTION = "GZS3"
LAKEHOUSE_CONNECTION = "Home-Heath-lh"
LAKEHOUSE_SITE = "preview"
DATABASE = "home_health_rcm"
OUTPUT_TABLE = "gold_rcm_exception_investigations"

AS_OF_DATE = "2026-10-31"

spark = sparkSession(
    "RCM_Exception_Investigation",
    "WARN",
    siteNames={LAKEHOUSE_CONNECTION: LAKEHOUSE_SITE},
)


def read_markdown(path):
    """Read a Markdown object from S3 and return its text."""
    df = spark.readS3(
        connName=S3_CONNECTION,
        fileName=path,
        format="text",
        options={},
    )

    content = "\n".join(
        row["value"] for row in df.select("value").collect()
    )

    if not content.strip():
        raise RuntimeError(f"No text was read from {path}")

    print(f"Read {len(content)} characters from {path}", flush=True)
    return content


def read_query_rows(query):
    """Run a Lakehouse SQL query and return its rows as dictionaries."""
    df = spark.executeQuery(LAKEHOUSE_CONNECTION, query)
    return [
        row.asDict(recursive=True)
        for row in df.collect()
    ]


def call_agent(incident, evidence, document_path, document_text):
    """Send one grouped incident and its evidence to the RCM agent."""
    message = f"""
Investigate this grouped RCM incident for human review.

Use only the incident data, SQL-calculated evidence, and document text
provided below. Do not recalculate or invent financial values.

Separate observed facts from documented context. If evidence does not
establish a cause, say the cause is unconfirmed and identify the missing
evidence. Cite the document by its title and source filename.

Use snapshot date {AS_OF_DATE}. Return the required structured JSON and
set review_status to pending.

Incident:
{json.dumps(incident, indent=2, default=str)}

SQL-calculated evidence:
{json.dumps(evidence, indent=2, default=str)}

Document source filename: {document_path}

Document text:
{document_text}
"""

    print(f"Calling agent for {incident['incident_id']}...", flush=True)

    response = spark.invokeAgent(
        AGENT_CONNECTION,
        AGENT_ID,
        message,
    )

    # Groundzero returns the validated structured response under outcome.data.
    result = None
    if isinstance(response, dict):
        outcome = response.get("outcome") or {}
        result = outcome.get("data")

        if result is None:
            raw_response = response.get("response")
            if isinstance(raw_response, str):
                result = json.loads(raw_response)
            elif isinstance(raw_response, dict):
                result = raw_response

    if not isinstance(result, dict):
        raise TypeError(
            f"Agent did not return a JSON object for {incident['incident_id']}"
        )

    if result.get("review_status") != "pending":
        raise ValueError(
            f"Unexpected review_status for {incident['incident_id']}"
        )

    return response, result


# Read the two policy documents from S3.
northbridge_doc_path = "casiva/rcm/docs/payer_policy_northbridge.md"
summit_audit_path = (
    "casiva/rcm/docs/audit_recoupment_notice_summit_advantage.md"
)
summit_close_path = "casiva/rcm/docs/month_end_close_note_aug2026.md"

northbridge_policy = read_markdown(northbridge_doc_path)
summit_audit = read_markdown(summit_audit_path)
summit_close = read_markdown(summit_close_path)


# Northbridge: invoice-to-first-valid-payment summary.
# This measures time to first posted payment, not full claim settlement.
northbridge_query = f"""
WITH payer AS (
    SELECT payer_id, typical_days_to_pay
    FROM {DATABASE}.ref_dim_payer
    WHERE LOWER(payer_name) LIKE '%northbridge%'
),
claim_level AS (
    SELECT
        t.claim_id,
        t.invoice_id,
        MIN(
            CASE WHEN t.transaction_type = 'Invoice'
                 THEN t.transaction_date
            END
        ) AS invoice_date,
        MIN(
            CASE
                WHEN t.transaction_type = 'Remittance'
                 AND COALESCE(t.is_duplicate_posting, 0) = 0
                 AND COALESCE(t.is_unapplied, 0) = 0
                THEN t.transaction_date
            END
        ) AS first_payment_date,
        SUM(
            CASE
                WHEN COALESCE(t.is_duplicate_posting, 0) = 0
                 AND COALESCE(t.is_unapplied, 0) = 0
                THEN t.ar_impact
                ELSE 0
            END
        ) AS open_balance
    FROM {DATABASE}.gold_fact_payment_transactions t
    INNER JOIN payer p
        ON t.payer_id = p.payer_id
    WHERE t.transaction_date <= DATE '{AS_OF_DATE}'
      AND t.claim_id IS NOT NULL
    GROUP BY t.claim_id, t.invoice_id
)
SELECT
    p.payer_id,
    p.typical_days_to_pay,
    COUNT(*) AS invoice_count,
    SUM(
        CASE WHEN c.first_payment_date IS NOT NULL THEN 1 ELSE 0 END
    ) AS invoices_with_payment,
    AVG(
        CASE WHEN c.first_payment_date IS NOT NULL
             THEN DATEDIFF(c.first_payment_date, c.invoice_date)
        END
    ) AS avg_days_to_first_payment,
    SUM(
        CASE
            WHEN c.first_payment_date IS NOT NULL
             AND DATEDIFF(c.first_payment_date, c.invoice_date)
                 > p.typical_days_to_pay
            THEN 1 ELSE 0
        END
    ) AS invoices_paid_after_typical_days,
    SUM(
        CASE
            WHEN c.first_payment_date IS NULL
             AND DATEDIFF(DATE '{AS_OF_DATE}', c.invoice_date)
                 > p.typical_days_to_pay
            THEN 1 ELSE 0
        END
    ) AS unpaid_over_typical_days,
    SUM(
        CASE
            WHEN c.first_payment_date IS NULL
             AND DATEDIFF(DATE '{AS_OF_DATE}', c.invoice_date)
                 > p.typical_days_to_pay
             AND c.open_balance > 0
            THEN c.open_balance ELSE 0
        END
    ) AS open_ar_over_typical_days
FROM claim_level c
CROSS JOIN payer p
WHERE c.invoice_date IS NOT NULL
GROUP BY p.payer_id, p.typical_days_to_pay
"""

northbridge_rows = read_query_rows(northbridge_query)

if not northbridge_rows:
    northbridge_evidence = {
        "result": "No Northbridge invoice rows matched the query.",
        "query_reference": northbridge_query,
        "source_table": f"{DATABASE}.gold_fact_payment_transactions",
    }
else:
    northbridge_evidence = {
        "summary_rows": northbridge_rows,
        "query_reference": northbridge_query,
        "source_table": f"{DATABASE}.gold_fact_payment_transactions",
        "measure_note": (
            "Days are measured from invoice date to first valid posted "
            "remittance. This does not measure full claim settlement."
        ),
    }

northbridge_incident = {
    "incident_id": f"northbridge-payment-delay-{AS_OF_DATE}",
    "incident_type": "payer_payment_delay",
    "snapshot_date": AS_OF_DATE,
    "payer_name": "Northbridge Health Plan",
    "analysis_scope": "Grouped payer-level summary; no patient identifiers",
}

northbridge_response, northbridge_result = call_agent(
    northbridge_incident,
    northbridge_evidence,
    northbridge_doc_path,
    northbridge_policy,
)


# Summit: summarize posted reversals by CARC code.
summit_query = f"""
WITH payer AS (
    SELECT payer_id, payer_name
    FROM {DATABASE}.ref_dim_payer
    WHERE LOWER(payer_name) LIKE '%summit%'
)
SELECT
    p.payer_id,
    p.payer_name,
    t.carc_code,
    COUNT(*) AS reversal_row_count,
    COUNT(DISTINCT t.claim_id) AS affected_claim_count,
    SUM(t.payment_amount) AS recorded_reversal_amount,
    MIN(t.transaction_date) AS first_reversal_date,
    MAX(t.transaction_date) AS last_reversal_date
FROM {DATABASE}.gold_fact_payment_transactions t
INNER JOIN payer p
    ON t.payer_id = p.payer_id
WHERE t.transaction_type = 'Reversal'
  AND t.transaction_date <= DATE '{AS_OF_DATE}'
  AND COALESCE(t.is_duplicate_posting, 0) = 0
  AND COALESCE(t.is_unapplied, 0) = 0
GROUP BY p.payer_id, p.payer_name, t.carc_code
"""

summit_rows = read_query_rows(summit_query)

summit_evidence = {
    "reversal_summary_by_carc": summit_rows,
    "query_reference": summit_query,
    "source_table": f"{DATABASE}.gold_fact_payment_transactions",
    "limitation": (
        "This Gold table provides reversal and CARC fields. It does not "
        "provide the service code or an N115 remark code, so those details "
        "are not confirmed by this query."
    ),
}

summit_incident = {
    "incident_id": f"summit-recoupment-{AS_OF_DATE}",
    "incident_type": "payer_recoupment",
    "snapshot_date": AS_OF_DATE,
    "payer_name": "Summit Advantage Medicare",
    "analysis_scope": "Grouped reversal summary; no patient identifiers",
}

summit_documents = (
    f"Document: {summit_audit_path}\n"
    f"{summit_audit}\n\n"
    f"Document: {summit_close_path}\n"
    f"{summit_close}"
)

summit_response, summit_result = call_agent(
    summit_incident,
    summit_evidence,
    f"{summit_audit_path}; {summit_close_path}",
    summit_documents,
)


# Save both structured investigations. Append preserves earlier job runs.
output_schema = StructType([
    StructField("incident_id", StringType(), False),
    StructField("incident_type", StringType(), False),
    StructField("snapshot_date", StringType(), False),
    StructField("payer_id", StringType(), True),
    StructField("agent_run_id", StringType(), True),
    StructField("generated_at_utc", StringType(), False),
    StructField("evidence_json", StringType(), False),
    StructField("document_sources", StringType(), False),
    StructField("investigation_json", StringType(), False),
    StructField("review_status", StringType(), False),
])

generated_at = datetime.now(timezone.utc).isoformat()

output_rows = [
    (
        northbridge_result.get("incident_id"),
        northbridge_result.get("incident_type"),
        northbridge_result.get("snapshot_date"),
        northbridge_result.get("payer_id"),
        northbridge_response.get("runId"),
        generated_at,
        json.dumps(northbridge_evidence, default=str),
        northbridge_doc_path,
        json.dumps(northbridge_result, default=str),
        northbridge_result.get("review_status"),
    ),
    (
        summit_result.get("incident_id"),
        summit_result.get("incident_type"),
        summit_result.get("snapshot_date"),
        summit_result.get("payer_id"),
        summit_response.get("runId"),
        generated_at,
        json.dumps(summit_evidence, default=str),
        f"{summit_audit_path}; {summit_close_path}",
        json.dumps(summit_result, default=str),
        summit_result.get("review_status"),
    ),
]

output_df = spark.createDataFrame(output_rows, schema=output_schema)

spark.writeGzLakehouse(
    df=output_df,
    connName=LAKEHOUSE_CONNECTION,
    database=DATABASE,
    tableName=OUTPUT_TABLE,
    mode="overwrite",
    streaming=False,
)

print(
    f"Saved {len(output_rows)} investigations to "
    f"{DATABASE}.{OUTPUT_TABLE}",
    flush=True,
)