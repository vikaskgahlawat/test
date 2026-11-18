"""Casiva RCM: Lakehouse Bronze -> Silver -> validation -> Gold.

Reads Bronze and loaded dimensions from the Groundzero lakehouse. 
"""

from GZ.SparkUtils import sparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ------------------------------- Settings -------------------------------
S3_CONNECTION_NAME = "GZS3"
LAKEHOUSE_CONNECTION_NAME = "Home-Heath-lh"
DATABASE = "home_health_rcm"
SITE_NAME = "preview"
CLEAN_PREFIX = "casiva/rcm/clean"
AS_OF_DATE = "2026-10-31"

spark = sparkSession(
    S3_CONNECTION_NAME,
    "WARN",
    siteNames={LAKEHOUSE_CONNECTION_NAME: SITE_NAME},
)
# ------------------------------- IO -------------------------------------
def read_lh(table_name):
    query = f"SELECT * FROM {DATABASE}.{table_name}"
    # This runtime accepts the SQL query as the second positional argument.
    return spark.executeQuery(LAKEHOUSE_CONNECTION_NAME, query)


def read_s3_csv(file_name, infer_schema=True, add_lineage=False):
    """Read an S3 CSV, used for reference files not loaded to LH and validation."""
    df = spark.readS3(
        connName=S3_CONNECTION_NAME,
        fileName=file_name,
        format="csv",
        options={"header": True, "inferSchema": infer_schema},
    )
    if add_lineage:
        original_columns = df.columns
        df = (df
            .withColumn("_source_file", F.lit(file_name))
            .withColumn("_row_hash", F.sha2(
                F.to_json(F.struct(*[F.col(c) for c in original_columns])), 256
            ))
            .withColumn("_ingested_at", F.current_timestamp())
        )
    return df


def ensure_lineage(df, source_table):
    """Use Bronze lineage when present; create stable source-only hash if absent."""
    if "_source_file" not in df.columns:
        df = df.withColumn("_source_file", F.lit(source_table))
    if "_row_hash" not in df.columns:
        source_columns = [c for c in df.columns if c not in ["_source_file", "_ingested_at"]]
        df = df.withColumn(
            "_row_hash",
            F.sha2(F.to_json(F.struct(*[F.col(c) for c in source_columns])), 256),
        )
    if "_ingested_at" not in df.columns:
        df = df.withColumn("_ingested_at", F.current_timestamp())
    return df


def write_lh(df, table_name):
    # Avoid an extra full Spark action before the Lakehouse write. The prior
    # count() was the first point where Spark evaluated the whole lazy plan and
    # obscured the actual destination write context in the job log.
    print(f"Writing {DATABASE}.{table_name}; output schema:")
    df.printSchema()
    try:
        spark.writeGzLakehouse(
            df=df,
            connName=LAKEHOUSE_CONNECTION_NAME,
            tableName=table_name,
            database=DATABASE,
            mode="overwrite",
            streaming=False,
        )
    except Exception as exc:
        print(f"Lakehouse write failed for {DATABASE}.{table_name}")
        print(f"Python/Py4J error: {repr(exc)}")
        java_exc = getattr(exc, "java_exception", None)
        level = 0
        while java_exc is not None and level < 8:
            print(f"Java cause {level + 1}: {java_exc.toString()}")
            java_exc = java_exc.getCause()
            level += 1
        raise
    print(f"Write completed for {DATABASE}.{table_name}")


def require_columns(df, name, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


# ------------------------------- Helpers --------------------------------
def norm(name):
    return F.upper(F.trim(F.col(name).cast("string")))


def money(name):
    raw = F.trim(F.col(name).cast("string"))
    cleaned = F.regexp_replace(raw, r"[$,]", "")
    return F.when(raw.isNull() | (raw == ""), F.lit(None).cast("decimal(18,2)")) \
        .otherwise(cleaned.cast("decimal(18,2)"))


def us_date(name):
    raw = F.trim(F.col(name).cast("string"))
    # Mixed visit values may contain a time after the date.
    date_part = F.when(raw.contains(" "), F.substring(raw, 1, 10)).otherwise(raw)
    # Expand US two-digit years (e.g. 01/01/25) explicitly to 20xx, then use
    # Spark 3.5 try_to_timestamp so nonmatching patterns return null, not errors.
    two_digit = date_part.rlike(r"^\d{1,2}/\d{1,2}/\d{2}$")
    month = F.regexp_extract(date_part, r"^(\d{1,2})/(\d{1,2})/(\d{2})$", 1)
    day = F.regexp_extract(date_part, r"^(\d{1,2})/(\d{1,2})/(\d{2})$", 2)
    year = F.regexp_extract(date_part, r"^(\d{1,2})/(\d{1,2})/(\d{2})$", 3)
    expanded = F.when(
        two_digit,
        F.concat(F.lpad(month, 2, "0"), F.lit("/"), F.lpad(day, 2, "0"), F.lit("/20"), year),
    ).otherwise(date_part)

    parsed = [
        F.to_date(F.try_to_timestamp(expanded, F.lit(pattern)))
        for pattern in [
            "yyyy-MM-dd", "M/d/yyyy", "MM/dd/yyyy", "yyyyMMdd",
            "dd-MMM-yyyy", "MM-dd-yyyy",
        ]
    ]
    serial_days = F.regexp_extract(date_part, r"^(\d{5})(?:\.0)?$", 1)
    excel_date = F.when(
        date_part.rlike(r"^\d{5}(?:\.0)?$"),
        F.date_add(F.lit("1899-12-30").cast("date"), serial_days.cast("int")),
    )
    return F.coalesce(*parsed, excel_date)


def claim_id_expr(claim_col, invoice_col):
    c = F.upper(F.trim(F.col(claim_col).cast("string")))
    i = F.upper(F.trim(F.col(invoice_col).cast("string")))
    c_num = F.regexp_extract(c, r"^C(\d+)$", 1)
    c_inv = F.regexp_extract(c, r"^INV-SYN-(\d+)$", 1)
    i_num = F.regexp_extract(i, r"^INV-SYN-(\d+)$", 1)
    return (
        F.when(c.startswith("CLM-SYN-"), c)
        .when(c_num != "", F.concat(F.lit("CLM-SYN-"), F.lpad(c_num, 7, "0")))
        .when(c_inv != "", F.concat(F.lit("CLM-SYN-"), F.lpad(c_inv, 7, "0")))
        .when(i_num != "", F.concat(F.lit("CLM-SYN-"), F.lpad(i_num, 7, "0")))
        .otherwise(c)
    )


def make_map(mapping):
    args = []
    for k, v in mapping.items():
        args.extend([F.lit(k), F.lit(v)])
    return F.create_map(*args)


def with_dq_reason(df, conditions):
    return df.withColumn(
        "_dq_reason",
        F.concat_ws("; ", *[F.when(cond, F.lit(msg)) for cond, msg in conditions]),
    )


def exception_rows(df, source):
    return (
        df.filter(F.col("_dq_reason").isNotNull() & (F.col("_dq_reason") != ""))
        .select(
            F.lit(source).alias("source_table"),
            F.col("_source_file").cast("string").alias("source_file"),
            F.col("_row_hash").cast("string").alias("row_hash"),
            F.col("_dq_reason").alias("reason"),
            F.to_json(F.struct(*[F.col(c) for c in df.columns])).alias("record_json"),
        )
    )


def reconcile(actual, expected, label, keys, amount_col):
    a = actual.groupBy(*keys).agg(
        F.count(F.lit(1)).cast("long").alias("actual_count"),
        F.sum(F.col(amount_col).cast("decimal(18,2)")).alias("actual_amount"),
    ).alias("a")
    e = expected.groupBy(*keys).agg(
        F.count(F.lit(1)).cast("long").alias("expected_count"),
        F.sum(F.col(amount_col).cast("decimal(18,2)")).alias("expected_amount"),
    ).alias("e")
    cond = None
    for k in keys:
        eq = F.col(f"a.{k}").eqNullSafe(F.col(f"e.{k}"))
        cond = eq if cond is None else cond & eq
    joined = a.join(e, cond, "full_outer")
    out = joined.select(
        F.lit(label).alias("dataset"),
        F.to_json(F.struct(*[
            F.coalesce(F.col(f"a.{k}"), F.col(f"e.{k}")).alias(k) for k in keys
        ])).alias("grain_key"),
        F.coalesce(F.col("a.actual_count"), F.lit(0)).alias("actual_count"),
        F.coalesce(F.col("e.expected_count"), F.lit(0)).alias("expected_count"),
        F.coalesce(F.col("a.actual_amount"), F.lit(0).cast("decimal(18,2)")).alias("actual_amount"),
        F.coalesce(F.col("e.expected_amount"), F.lit(0).cast("decimal(18,2)")).alias("expected_amount"),
    )
    return (
        out.withColumn("count_difference", F.col("actual_count") - F.col("expected_count"))
        .withColumn("amount_difference", F.round(F.col("actual_amount") - F.col("expected_amount"), 2))
        .withColumn("status", F.when(
            (F.col("count_difference") == 0) & (F.abs(F.col("amount_difference")) <= 0.01),
            "MATCH",
        ).otherwise("DIFFERENCE"))
    )


# ----------------------- Read Bronze and references ----------------------
print("Reading Bronze and loaded lookup tables from the lakehouse...")
billing_raw = ensure_lineage(read_lh("bronze_billing_claims"), "bronze_billing_claims")
remit_raw = ensure_lineage(read_lh("bronze_remit_835"), "bronze_remit_835")
schedule_raw = ensure_lineage(read_lh("bronze_scheduling_visits"), "bronze_scheduling_visits")
payer_ref = read_lh("ref_dim_payer")
location_ref = read_lh("ref_dim_location")
caregiver_ref = read_lh("ref_dim_caregiver")
# These lookup dimension tables are also read from the lakehouse.
patient_ref = read_lh("ref_dim_patient")
tenant_ref = read_lh("ref_dim_tenant")
franchisor_ref = read_lh("ref_dim_franchisor")
date_ref = read_lh("ref_dim_date")
# Bridge/code references and clean fact validation files are read from S3;
# no lakehouse tables for these inputs were shown in the current setup.
bridge = read_s3_csv(f"{CLEAN_PREFIX}/bridge_caregiver_location.csv")
carc_ref = read_s3_csv(f"{CLEAN_PREFIX}/ref_carc_codes.csv")
service_ref = read_s3_csv(f"{CLEAN_PREFIX}/ref_service_codes.csv")

require_columns(billing_raw, "bronze_billing_claims", [
    "loc_billing_id", "claim_no", "invoice_no", "patient_id", "payer_name",
    "invoice_date", "svc_from", "svc_to", "total_charges",
])
require_columns(remit_raw, "bronze_remit_835", [
    "source_system", "payer_name", "payer_id_raw", "location_code", "patient_id",
    "member_id", "claim_id", "invoice_id", "txn_type", "service_date",
    "payment_date", "paid", "adjustment", "adj_reason", "check_or_eft",
])
require_columns(schedule_raw, "bronze_scheduling_visits", [
    "sched_location_id", "employee_id", "patient_ref", "visit_date", "start_time",
    "end_time", "hours", "hours_uom", "status",
])

# Explicit payer alias crosswalk. Canonical names are populated from ref_dim_payer.
payer_aliases = {}
canonical_payer_names = set()
valid_payer_ids = set()
for r in payer_ref.select("payer_id", "payer_name").collect():
    payer_aliases[r["payer_name"].strip().upper()] = r["payer_id"]
    canonical_payer_names.add(r["payer_name"].strip().upper())
    valid_payer_ids.add(r["payer_id"])
payer_aliases.update({
    "MEDICARE": "PAY-001", "MEDICARE FFS": "PAY-001", "CMS MEDICARE": "PAY-001",
    "MEDICARE TRADITIONAL": "PAY-001", "MCR": "PAY-001",
    "FL MEDICAID": "PAY-003", "AHCA FLORIDA": "PAY-003", "FL-MCD": "PAY-003",
    "VA-CCN": "PAY-005", "VETERANS COMMUNITY CARE": "PAY-005",
    "VA CCN": "PAY-005", "VHA COMMUNITY CARE": "PAY-005",
    "NORTHBRIDGE HEALTH": "PAY-006", "NORTHBRIDGE HP": "PAY-006",
    "NBHP": "PAY-006", "NORTHBRIDGE HEALTHPLAN": "PAY-006",
    "SUMMIT ADV MA": "PAY-007", "SELF PAY": "PAY-013",
    "PRIVATEPAY": "PAY-013", "SELF-PAY": "PAY-013",
    "PATIENT PRIVATE PAY": "PAY-013",
})
# Complete aliases from the documented synthetic source contract.
payer_alias_groups = {
 "PAY-002": ["IL MEDICAID", "HFS Illinois", "IL-MCD", "Illinois Dept of Healthcare"],
 "PAY-004": ["TMHP", "TX MEDICAID", "TX-MCD", "Texas Medicaid & Healthcare"],
 "PAY-007": ["SUMMIT ADVANTAGE", "Summit Advantage", "SAM-MA"],
 "PAY-008": ["BEACON SENIOR ADV", "Beacon MA", "Beacon Senior"],
 "PAY-009": ["HEARTLAND BS", "Heartland BlueShield", "HBS"],
 "PAY-010": ["PACIFIC CARE+", "PacificCare Plus", "PCP"],
 "PAY-011": ["OAKMONT", "Oakmont Health Plan", "OHP"],
 "PAY-012": ["EVERGREEN LTC", "Evergreen LTCI", "EGLTC"],
 "PAY-014": ["PATRIOT CC", "Patriot Community", "PCC-MCD"],
 "PAY-015": ["WA APPLE HEALTH", "Apple Health", "WA-MCD"],
}
for r in read_lh("ref_payer_alias").collect():
    key, value = r["raw_payer_alias"].strip().upper(), r["payer_id"]
    if key in payer_aliases and payer_aliases[key] != value:
        raise ValueError(f"Conflicting payer mapping: {key}")
    payer_aliases[key] = value
for payer_id, aliases in payer_alias_groups.items():
    for alias in aliases:
        payer_aliases[alias.strip().upper()] = payer_id
payer_lookup = make_map(payer_aliases)
write_lh(spark.createDataFrame(
    [(alias, pid, "reviewed_source_contract") for alias, pid in sorted(payer_aliases.items())],
    "raw_payer_alias string, payer_id string, alias_source string"), "ref_payer_alias")
carc_aliases = {}
for r in carc_ref.select("carc_code", "description").collect():
    code, description = r["carc_code"].upper(), r["description"].upper()
    for alias in [code, code.replace("-", ""), code.split("-")[-1], description, description[:40]]:
        if alias in carc_aliases and carc_aliases[alias] != code:
            raise ValueError(f"Ambiguous CARC alias: {alias}")
        carc_aliases[alias] = code
carc_lookup = make_map(carc_aliases)

# Explicit location-code crosswalk built from dimensions plus common raw aliases.
location_aliases = {}
location_alias_rows = []
for r in location_ref.select(
    "location_id", "location_name", "city", "state",
    "billing_location_code", "scheduling_location_code",
).collect():
    lid = r["location_id"]
    n = lid.split("-")[-1]
    city = (r["city"] or "").strip().upper()
    state = (r["state"] or "").strip().upper()
    typed_aliases = [
        (lid, "canonical_id"), (r["location_name"], "location_name"),
        (r["city"], "city_name"), (r["billing_location_code"], "billing_primary"),
        (r["scheduling_location_code"], "scheduling_primary"),
        (n, "numeric_billing_alias"), (str(int(n)), "numeric_billing_alias"),
        (f"LOC{n}", "billing_alias"), (f"BILL-L-{n}", "billing_alias"),
        (f"SCH-L-{n}", "scheduling_alias"), (f"BS-{city}", "scheduling_alias"),
        (f"{city}-{state}", "city_state_alias"),
        (f"{state}-{city[:3]}-{int(n):02d}", "billing_alias"),
        (f"BS-{city.replace(' ', '')}", "scheduling_alias"),
        (f"{city.replace(' ', '-')}-{state}", "scheduling_alias"),
    ]
    for alias, code_type in typed_aliases:
        if alias:
            key = str(alias).strip().upper()
            location_aliases[key] = lid
            location_alias_rows.append((key, lid, code_type))
location_aliases["IL-NAP-01"] = "L-001"
location_alias_rows.append(("IL-NAP-01", "L-001", "billing_alias"))
for r in read_lh("ref_location_code").collect():
    key, value = r["raw_location_code"].strip().upper(), r["location_id"]
    if key in location_aliases and location_aliases[key] != value:
        raise ValueError(f"Conflicting location mapping: {key}")
    location_aliases[key] = value
    location_alias_rows.append((key, value, r["code_type"]))
location_lookup = make_map(location_aliases)
write_lh(spark.createDataFrame(list(set(location_alias_rows)),
    "raw_location_code string, location_id string, code_type string"), "ref_location_code")
valid_caregivers = [r["caregiver_id"] for r in caregiver_ref.select("caregiver_id").collect()]

# Crosswalks are held in memory for this run; only Silver/Gold outputs are written.


# ----------------------- Billing Bronze -> Silver ------------------------
print("Cleaning billing claims...")
billing_raw = billing_raw.dropDuplicates(["_row_hash"])
billing = (
    billing_raw
    .withColumn("claim_id", claim_id_expr("claim_no", "invoice_no"))
    .withColumn("invoice_id", F.upper(F.trim(F.col("invoice_no"))))
    .withColumn("patient_id", F.upper(F.trim(F.col("patient_id"))))
    .withColumn("location_id", F.coalesce(
        F.element_at(location_lookup, norm("loc_billing_id")),
        F.element_at(location_lookup, norm("location_name")),
    ))
    .withColumn("payer_id", F.element_at(payer_lookup, norm("payer_name")))
    .withColumn("invoice_date_clean", us_date("invoice_date"))
    .withColumn("service_start", us_date("svc_from"))
    .withColumn("service_end", us_date("svc_to"))
    .withColumn("charge_amount", money("total_charges"))
    .withColumn("units_visits_clean", F.col("units_visits").cast("int"))
)
billing = with_dq_reason(billing, [
    (F.col("claim_id").isNull() | (F.col("claim_id") == ""), "missing claim ID"),
    (F.col("invoice_date_clean").isNull(), "invalid invoice date"),
    (F.col("charge_amount").isNull(), "invalid charge amount"),
    (F.col("payer_id").isNull(), "unmapped payer"),
    (F.col("location_id").isNull(), "unmapped billing location"),
])
billing_dq = exception_rows(billing, "billing")
billing_silver = billing.filter(F.col("_dq_reason") == "").drop("_dq_reason")
write_lh(billing_silver, "silver_billing_claims")


# ---------------------- Remittance Bronze -> Silver ----------------------
print("Cleaning remittances...")
# Hash contains source fields only. Exact source duplicates are removed in Silver;
# Bronze remains intact. Repeat ingestion must separately be idempotent upstream.
if "_row_hash" in remit_raw.columns:
    remit_dedup = remit_raw.dropDuplicates(["_row_hash"])
else:
    remit_dedup = remit_raw.dropDuplicates([
        c for c in remit_raw.columns if c not in ["_source_file", "_ingested_at"]
    ])

remit = (
    remit_dedup
    .withColumn("transaction_type", F.initcap(F.trim(F.col("txn_type"))))
    .withColumn("claim_id", claim_id_expr("claim_id", "invoice_id"))
    .withColumn("invoice_id", F.upper(F.trim(F.col("invoice_id"))))
    .withColumn(
        "_claim_was_supplied",
        F.when(F.col("claim_id").isNotNull() & (F.col("claim_id") != ""), 1).otherwise(0),
    )
    .withColumn("payer_id_from_raw", F.col("payer_id_raw").cast("string"))
    .withColumn("payer_id", F.when(
        F.col("payer_id_from_raw").isin(list(valid_payer_ids)), F.col("payer_id_from_raw")
    ).otherwise(F.element_at(payer_lookup, norm("payer_name"))))
    .withColumn("location_id", F.element_at(location_lookup, norm("location_code")))
    .withColumn("service_start", us_date("service_date"))
    .withColumn("transaction_date", us_date("payment_date"))
    .withColumn("paid_amount", F.abs(money("paid")))
    .withColumn("adjustment_amount", F.abs(money("adjustment")))
    .withColumn("carc_code", F.element_at(carc_lookup, norm("adj_reason")))
    # The source generator serializes zero-dollar denial adjustments as blanks.
    .withColumn("adjustment_amount", F.when(
        (F.col("transaction_type") == "Adjustment")
        & (F.col("adjustment").isNull() | (F.trim(F.col("adjustment")) == ""))
        & F.col("carc_code").isin("CO-16", "CO-29", "CO-50", "CO-97", "CO-151", "CO-18", "CO-4"),
        F.lit(0).cast("decimal(18,2)")
    ).otherwise(F.col("adjustment_amount")))
)

# Sign normalization considers source system and transaction type. MANUAL-POST
# reversals may arrive as negative paid values; normalized event amounts are positive.
remit = remit.withColumn(
    "ar_impact",
    # MANUAL-POST reversals may arrive with negative paid values; LOCKBOX-CSV
    # adjustments may arrive negative. Amounts are normalized to magnitudes,
    # then source system and transaction type determine the accounting sign.
    F.when((F.col("transaction_type") == "Reversal") & (F.col("source_system") == "MANUAL-POST"), F.abs(F.col("paid_amount")))
    .when(F.col("transaction_type") == "Reversal", F.abs(F.col("paid_amount")))
    .when(F.col("transaction_type") == "Remittance", -F.abs(F.col("paid_amount")))
    .when((F.col("transaction_type") == "Adjustment") & (F.col("source_system") == "LOCKBOX-CSV"), -F.abs(F.col("adjustment_amount")))
    .when(F.col("transaction_type") == "Adjustment", -F.abs(F.col("adjustment_amount")))
    .otherwise(F.lit(0).cast("decimal(18,2)")),
)

# Recover patient/location when a supplied claim ID or invoice ID matches billing.
bill_by_claim = billing_silver.filter(F.col("claim_id").isNotNull()).select(
    "claim_id",
    F.lit(1).alias("_claim_matched"),
    F.col("patient_id").alias("_claim_patient_id"),
    F.col("location_id").alias("_claim_location_id"),
).dropDuplicates(["claim_id"])
bill_by_invoice = billing_silver.filter(F.col("invoice_id").isNotNull()).select(
    "invoice_id",
    F.lit(1).alias("_invoice_matched"),
    F.col("patient_id").alias("_invoice_patient_id"),
    F.col("location_id").alias("_invoice_location_id"),
).dropDuplicates(["invoice_id"])
remit = remit.join(bill_by_claim, "claim_id", "left").join(bill_by_invoice, "invoice_id", "left")
remit = remit.withColumn(
    "patient_id", F.coalesce(F.col("patient_id"), F.col("_claim_patient_id"), F.col("_invoice_patient_id"))
).withColumn(
    "location_id", F.coalesce(F.col("location_id"), F.col("_claim_location_id"), F.col("_invoice_location_id"))
).withColumn(
    "billing_match_found",
    F.when(F.col("_claim_matched").isNotNull() | F.col("_invoice_matched").isNotNull(), 1).otherwise(0),
).withColumn(
    "claim_match_failed",
    F.when(
        ((F.col("_claim_was_supplied") == 1) & F.col("_claim_matched").isNull())
        | ((F.col("_claim_was_supplied") == 0)
           & F.col("invoice_id").isNotNull() & (F.col("invoice_id") != "")
           & F.col("_invoice_matched").isNull()),
        1,
    ).otherwise(0),
).drop(
    "_claim_matched", "_invoice_matched", "_claim_patient_id", "_invoice_patient_id", "_claim_location_id",
    "_invoice_location_id", "_claim_was_supplied", "payer_id_from_raw",
)

# A missing claim can be recovered through invoice ID before classifying cash.
remit = remit.withColumn(
    "claim_id",
    F.when(
        (F.col("claim_id").isNull() | (F.col("claim_id") == ""))
        & F.col("invoice_id").isNotNull() & (F.col("invoice_id") != ""),
        F.concat(F.lit("CLM-SYN-"), F.regexp_extract(F.col("invoice_id"), r"INV-SYN-(\d+)", 1)),
    ).otherwise(F.col("claim_id")),
).withColumn(
    "is_unapplied",
    F.when(
        (F.col("claim_id").isNull() | (F.col("claim_id") == ""))
        & (F.col("invoice_id").isNull() | (F.col("invoice_id") == "")), 1
    ).otherwise(0),
).withColumn(
    "reconciliation_exception",
    F.when((F.col("is_unapplied") == 0) & (F.col("claim_match_failed") == 1), 1).otherwise(0),
)

dupe_window = Window.partitionBy("check_or_eft", "claim_id", "paid_amount").orderBy(
    F.col("transaction_date").asc_nulls_last(), F.col("_row_hash")
)
remit = remit.withColumn(
    "is_duplicate_posting",
    F.when(
        (F.col("transaction_type") == "Remittance")
        & (F.col("source_system") == "MANUAL-POST")
        & F.col("check_or_eft").isNotNull()
        & F.col("claim_id").isNotNull()
        & (F.row_number().over(dupe_window) > 1), 1
    ).otherwise(0),
)
remit = with_dq_reason(remit, [
    (F.col("transaction_date").isNull(), "invalid payment date"),
    (F.col("payer_id").isNull(), "unmapped payer"),
    (~F.col("transaction_type").isin("Remittance", "Reversal", "Adjustment"), "unknown transaction type"),
    (F.col("transaction_type").isin("Remittance", "Reversal") & F.col("paid_amount").isNull(), "missing/invalid paid amount"),
    ( (F.col("transaction_type") == "Adjustment") & F.col("adjustment_amount").isNull(), "missing/invalid adjustment amount"),
])
remit_dq = exception_rows(remit, "remittance")
remit_silver = remit.filter(F.col("_dq_reason") == "").drop("_dq_reason")
write_lh(remit_silver, "silver_remit_835")


# ---------------------- Scheduling Bronze -> Silver ----------------------
print("Cleaning scheduled visits...")
schedule_raw = schedule_raw.dropDuplicates(["_row_hash"])
emp_digits = F.regexp_extract(F.col("employee_id"), r"EMP(\d+)", 1)
hrs = money("hours")
schedule = (
    schedule_raw
    # Preserve the original combined date/time before parsing the date column.
    .withColumn("visit_date_raw", F.trim(F.col("visit_date").cast("string")))
    .withColumn("location_id", F.element_at(location_lookup, norm("sched_location_id")))
    .withColumn("caregiver_id", F.when(
        emp_digits != "",
        F.concat(F.lit("CG-"), F.lpad((emp_digits.cast("int") - 4000).cast("string"), 4, "0")),
    ))
    .withColumn("patient_id", F.when(
        F.trim(F.col("patient_ref")) == "", F.lit(None).cast("string")
    ).otherwise(F.upper(F.trim(F.col("patient_ref")))))
    .withColumn("start_time", F.when(
        F.col("start_time").isNull() | (F.trim(F.col("start_time")) == ""),
        F.when(
            F.col("visit_date_raw").contains(" "),
            F.regexp_extract(F.col("visit_date_raw"), r"\s(\d{1,2}:\d{2})$", 1),
        ),
    ).otherwise(F.trim(F.col("start_time"))))
    .withColumn("visit_date", us_date("visit_date_raw"))
    .withColumn("end_time", F.trim(F.col("end_time")))
    .withColumn("hours", F.when(
        F.lower(F.trim(F.col("hours_uom"))) == "min", hrs / 60
    ).otherwise(hrs).cast("decimal(18,2)"))
    .withColumn("service_code", F.upper(F.trim(F.col("paycode"))))
    .withColumn("status", F.when(F.upper(F.trim(F.col("status"))) == "NO-SHOW", "No-show")
        .when(F.upper(F.trim(F.col("status"))) == "CANCELLED", "Cancelled")
        .when(F.upper(F.trim(F.col("status"))) == "COMPLETED", "Completed")
        .otherwise(F.initcap(F.trim(F.col("status")))))
)
schedule = with_dq_reason(schedule, [
    (F.col("visit_date").isNull(), "invalid visit date"),
    (F.col("location_id").isNull(), "unmapped scheduling location"),
    (~F.col("caregiver_id").isin(valid_caregivers), "caregiver ID not found in reference dimension"),
    (F.col("patient_id").isNull(), "patient ID missing; no reliable recovery key in raw scheduling row"),
    (F.col("hours").isNull(), "invalid hours"),
])
schedule_dq = exception_rows(schedule, "scheduling")
schedule_silver = schedule.filter(F.col("_dq_reason") == "").drop("_dq_reason")

# Validate caregiver-to-location membership on the visit date. Keep and flag rows
# that violate the assignment dates; do not quarantine them.
require_columns(bridge, "bridge_caregiver_location.csv", [
    "caregiver_id", "location_id", "assignment_start", "assignment_end"
])
bridge_dates = bridge.select(
    F.col("caregiver_id").cast("string").alias("_b_caregiver_id"),
    F.col("location_id").cast("string").alias("_b_location_id"),
    F.to_date("assignment_start").alias("_assignment_start"),
    F.to_date("assignment_end").alias("_assignment_end"),
).dropDuplicates()
valid_visit_assignments = schedule_silver.alias("v").join(
    bridge_dates.alias("b"),
    (F.col("v.caregiver_id") == F.col("b._b_caregiver_id"))
    & (F.col("v.location_id") == F.col("b._b_location_id"))
    & (F.col("v.visit_date") >= F.col("b._assignment_start"))
    & (F.col("b._assignment_end").isNull() | (F.col("v.visit_date") <= F.col("b._assignment_end"))),
    "inner",
).select(F.col("v._row_hash").alias("_assignment_match_hash")).dropDuplicates()
schedule_silver = schedule_silver.join(
    valid_visit_assignments,
    schedule_silver._row_hash == valid_visit_assignments._assignment_match_hash,
    "left",
).withColumn(
    "assignment_violation",
    F.when(F.col("_assignment_match_hash").isNull(), 1).otherwise(0),
).drop("_assignment_match_hash")
write_lh(schedule_silver, "silver_scheduling_visits")


# ---------------------------- DQ exceptions ------------------------------
dq = billing_dq.unionByName(remit_dq).unionByName(schedule_dq)
# Present claim IDs that fail to match billing are not unapplied cash.
unmatched_claim_exceptions = remit_silver.filter(
    (F.col("is_unapplied") == 0) & (F.col("reconciliation_exception") == 1)
).select(
    F.lit("remittance").alias("source_table"),
    F.col("_source_file").cast("string").alias("source_file"),
    F.col("_row_hash").cast("string").alias("row_hash"),
    F.lit("claim/invoice present but no matching billing record was found").alias("reason"),
    F.to_json(F.struct(*[F.col(c) for c in remit_silver.columns])).alias("record_json"),
)
dq = dq.unionByName(unmatched_claim_exceptions)
write_lh(dq, "dq_exceptions")


# ---------------- Reconcile to supplied clean facts (S3 only) -------------
print("Reconciling billing + remittance to clean payment fact (validation only)...")
expected_payment = read_s3_csv(f"{CLEAN_PREFIX}/fact_payment_transactions.csv")
require_columns(expected_payment, "fact_payment_transactions", [
    "transaction_type", "claim_id", "invoice_id", "transaction_date",
    "charge_amount", "payment_amount", "adjustment_amount",
])

actual_invoices = billing_silver.select(
    F.lit("Invoice").alias("transaction_type"), "claim_id", "invoice_id",
    F.col("invoice_date_clean").alias("transaction_date"),
    F.col("charge_amount").alias("amount"),
)
actual_remits = remit_silver.select(
    "transaction_type", "claim_id", "invoice_id", "transaction_date",
    F.when(F.col("transaction_type") == "Adjustment", F.col("adjustment_amount"))
    .otherwise(F.col("paid_amount")).cast("decimal(18,2)").alias("amount"),
)
actual_payment = actual_invoices.unionByName(actual_remits)
expected_payment_events = expected_payment.select(
    F.initcap(F.trim(F.col("transaction_type"))).alias("transaction_type"),
    F.upper(F.trim(F.col("claim_id"))).alias("claim_id"),
    F.upper(F.trim(F.col("invoice_id"))).alias("invoice_id"),
    F.to_date("transaction_date").alias("transaction_date"),
    F.when(F.col("transaction_type") == "Invoice", F.col("charge_amount"))
    .when(F.col("transaction_type") == "Adjustment", F.col("adjustment_amount"))
    .otherwise(F.col("payment_amount")).cast("decimal(18,2)").alias("amount"),
)
payment_recon = reconcile(
    actual_payment, expected_payment_events, "payment_transactions",
    ["transaction_type", "claim_id", "invoice_id", "transaction_date"], "amount",
)

print("Reconciling scheduling to clean visit fact (validation only)...")
expected_visits = read_s3_csv(f"{CLEAN_PREFIX}/fact_visits.csv")
require_columns(expected_visits, "fact_visits", ["location_id", "visit_date", "status", "hours"])
actual_visits_for_recon = schedule_silver.select(
    "location_id", F.col("visit_date").alias("visit_date"), "status", "hours"
)
expected_visits_for_recon = expected_visits.select(
    F.col("location_id").cast("string").alias("location_id"),
    F.to_date("visit_date").alias("visit_date"),
    F.when(F.upper(F.trim(F.col("status"))) == "NO-SHOW", "No-show")
    .when(F.upper(F.trim(F.col("status"))) == "CANCELLED", "Cancelled")
    .when(F.upper(F.trim(F.col("status"))) == "COMPLETED", "Completed")
    .otherwise(F.initcap(F.trim(F.col("status")))).alias("status"),
    F.col("hours").cast("decimal(18,2)").alias("hours"),
)
visit_recon = reconcile(
    actual_visits_for_recon, expected_visits_for_recon, "visits",
    ["location_id", "visit_date", "status"], "hours",
)
reconciliation_report = payment_recon.unionByName(visit_recon)
write_lh(reconciliation_report, "reconciliation_report")
reconciliation_report.filter(F.col("status") == "DIFFERENCE").show(100, truncate=False)


# ------------------------------ Gold facts -------------------------------
print("Building Gold payment fact...")
location_tenant = location_ref.select(
    F.col("location_id").cast("string").alias("_gold_location_id"),
    F.col("tenant_id").cast("string").alias("tenant_id"),
)

invoice_gold = billing_silver.select(
    F.sha2(F.concat_ws("|", F.lit("Invoice"), F.col("_row_hash")), 256).alias("transaction_id"),
    F.lit("Invoice").alias("transaction_type"), "claim_id", "invoice_id", "patient_id",
    "payer_id", "location_id", F.col("invoice_date_clean").alias("transaction_date"),
    F.col("charge_amount").cast("decimal(18,2)").alias("charge_amount"),
    F.lit(0).cast("decimal(18,2)").alias("payment_amount"),
    F.lit(0).cast("decimal(18,2)").alias("adjustment_amount"),
    F.col("charge_amount").cast("decimal(18,2)").alias("ar_impact"),
    F.lit(None).cast("string").alias("carc_code"),
    F.lit(None).cast("string").alias("check_eft_number"),
    F.lit(0).alias("is_duplicate_posting"), F.lit(0).alias("is_unapplied"),
    F.col("_source_file").alias("source_file"), F.col("_row_hash").alias("source_row_hash"),
)
remit_gold = remit_silver.select(
    F.sha2(F.concat_ws("|", F.col("transaction_type"), F.col("_row_hash")), 256).alias("transaction_id"),
    "transaction_type", "claim_id", "invoice_id", "patient_id", "payer_id", "location_id",
    F.col("transaction_date"),
    F.lit(0).cast("decimal(18,2)").alias("charge_amount"),
    F.when(F.col("transaction_type").isin("Remittance", "Reversal"), F.col("paid_amount"))
    .otherwise(F.lit(0)).cast("decimal(18,2)").alias("payment_amount"),
    F.when(F.col("transaction_type") == "Adjustment", F.col("adjustment_amount"))
    .otherwise(F.lit(0)).cast("decimal(18,2)").alias("adjustment_amount"),
    F.col("ar_impact").cast("decimal(18,2)").alias("ar_impact"),
    F.when(F.col("carc_code") == "", F.lit(None).cast("string")).otherwise(F.col("carc_code")).alias("carc_code"),
    F.col("check_or_eft").alias("check_eft_number"), "is_duplicate_posting", "is_unapplied",
    F.col("_source_file").alias("source_file"), F.col("_row_hash").alias("source_row_hash"),
)
payment_union = invoice_gold.unionByName(remit_gold).alias("p")
gold_payment = payment_union.join(
    location_tenant.alias("l"),
    F.col("p.location_id") == F.col("l._gold_location_id"),
    "left",
).drop("_gold_location_id")
carc_descriptions = carc_ref.select(
    F.upper(F.trim(F.col("carc_code"))).alias("_carc_lookup_code"),
    F.col("description").alias("carc_description"),
)
gold_payment = gold_payment.join(
    carc_descriptions,
    F.upper(F.trim(gold_payment.carc_code)) == carc_descriptions._carc_lookup_code,
    "left",
).drop("_carc_lookup_code")
write_lh(gold_payment, "gold_fact_payment_transactions")

print("Building Gold visit fact...")
gold_visits = schedule_silver.join(
    location_tenant,
    schedule_silver.location_id == location_tenant._gold_location_id,
    "left",
).select(
    F.concat(F.lit("VIS-"), F.substring(F.col("_row_hash"), 1, 12)).alias("visit_id"),
    "patient_id", "caregiver_id", "location_id", "tenant_id", "visit_date",
    "start_time", "end_time", "hours", "status", "service_code",
    "assignment_violation", F.col("_source_file").alias("source_file"),
    F.col("_row_hash").alias("source_row_hash"),
)
service_descriptions = service_ref.select(
    F.upper(F.trim(F.col("service_code"))).alias("_service_lookup_code"),
    F.col("description").alias("service_description"),
)
gold_visits = gold_visits.join(
    service_descriptions,
    F.upper(F.trim(gold_visits.service_code)) == service_descriptions._service_lookup_code,
    "left",
).drop("_service_lookup_code")
write_lh(gold_visits, "gold_fact_visits")


# ---------------------- Gold AR and supported KPIs -----------------------
print("Calculating AR aging and supported KPI snapshot...")
as_of = F.lit(AS_OF_DATE).cast("date")
claim_balances = (
    gold_payment.filter(F.col("claim_id").isNotNull() & (F.col("transaction_date") <= as_of))
    .groupBy("claim_id", "invoice_id")
    .agg(
        F.sum("ar_impact").cast("decimal(18,2)").alias("open_balance"),
        F.min(F.when(F.col("transaction_type") == "Invoice", F.col("transaction_date"))).alias("original_invoice_date"),
        F.first("payer_id", ignorenulls=True).alias("payer_id"),
        F.first("location_id", ignorenulls=True).alias("location_id"),
        F.first("tenant_id", ignorenulls=True).alias("tenant_id"),
    )
    .filter(F.col("open_balance") > 0)
    .withColumn("age_days", F.datediff(as_of, F.col("original_invoice_date")))
    .withColumn("aging_bucket", F.when(F.col("age_days") <= 30, "0-30")
        .when(F.col("age_days") <= 60, "31-60")
        .when(F.col("age_days") <= 90, "61-90")
        .otherwise("91+"))
)
ar_aging = claim_balances.groupBy("aging_bucket").agg(
    F.countDistinct("claim_id").alias("claim_count"),
    F.sum("open_balance").cast("decimal(18,2)").alias("open_ar"),
)
write_lh(ar_aging, "gold_ar_aging")

as_of_py = F.lit(AS_OF_DATE).cast("date")
start_90 = F.date_sub(as_of_py, 89)
eligible = gold_payment.filter(F.col("transaction_date") <= as_of_py)
totals = eligible.agg(
    F.sum(F.when(F.col("transaction_type") == "Invoice", F.col("charge_amount")).otherwise(0)).alias("gross_charges"),
    F.sum(F.when(
        (F.col("transaction_type") == "Adjustment") & (F.col("carc_code") == "CO-45"),
        F.col("adjustment_amount"),
    ).otherwise(0)).alias("contractual_co45"),
    F.sum(F.when(
        (F.col("transaction_type") == "Remittance")
        & (F.col("is_duplicate_posting") == 0) & (F.col("is_unapplied") == 0),
        F.col("payment_amount"),
    ).otherwise(0)).alias("posted_payments"),
    F.sum(F.when(
        (F.col("transaction_type") == "Reversal")
        & (F.col("is_duplicate_posting") == 0) & (F.col("is_unapplied") == 0),
        F.col("payment_amount"),
    ).otherwise(0)).alias("reversals"),
    F.sum(F.when(
        (F.col("transaction_type") == "Invoice")
        & (F.col("transaction_date") >= start_90), F.col("charge_amount"),
    ).otherwise(0)).alias("charges_last_90"),
    F.sum(F.when(
        (F.col("transaction_type") == "Adjustment")
        & (F.col("carc_code") == "CO-45")
        & (F.col("transaction_date") >= start_90), F.col("adjustment_amount"),
    ).otherwise(0)).alias("co45_last_90"),
).first()

open_debit = claim_balances.agg(F.sum("open_balance").alias("v")).first()["v"] or 0
gross = totals["gross_charges"] or 0
co45 = totals["contractual_co45"] or 0
payments = totals["posted_payments"] or 0
reversals = totals["reversals"] or 0
net_rev_90 = (totals["charges_last_90"] or 0) - (totals["co45_last_90"] or 0)
dso = float(open_debit) / (float(net_rev_90) / 90.0) if net_rev_90 else None
net_collection = (float(payments) - float(reversals)) / (float(gross) - float(co45)) if (gross - co45) else None

kpi_rows = [
    (AS_OF_DATE, "open_debit_ar", float(open_debit), "SUPPORTED", "Sum of positive per-claim balances as of snapshot"),
    (AS_OF_DATE, "gross_charges", float(gross), "SUPPORTED", "Invoice charges through snapshot date"),
    (AS_OF_DATE, "contractual_co45", float(co45), "SUPPORTED_IF_CARC_MAPPED", "CO-45 adjustments through snapshot date"),
    (AS_OF_DATE, "net_payments_excluding_duplicates_unapplied_and_reversals", float(payments) - float(reversals), "SUPPORTED", "Eligible remittances less reversals"),
    (AS_OF_DATE, "net_collection_rate", net_collection, "SUPPORTED_IF_CARC_MAPPED", "(eligible payments - reversals) / (charges - CO-45)"),
    (AS_OF_DATE, "dso", dso, "SUPPORTED_IF_CARC_MAPPED", "Open debit AR / (trailing 90-day net revenue / 90)"),
    (AS_OF_DATE, "first_pass_denial_rate", None, "UNSUPPORTED_FROM_CURRENT_RAW_FIELDS", "Submission status alone does not establish reliable first-pass invoice outcome"),
]
kpi_schema = "snapshot_date string, metric_name string, metric_value double, support_status string, definition string"
gold_kpis = spark.createDataFrame(kpi_rows, schema=kpi_schema)
write_lh(gold_kpis, "gold_kpi_snapshot")


# --------------------- Caregiver overlap investigation --------------------
print("Finding overlapping caregiver visit times...")
visits_for_overlap = gold_visits.withColumn(
    "start_ts", F.to_timestamp(F.concat_ws(" ", F.col("visit_date"), F.col("start_time")))
).withColumn(
    "end_ts", F.to_timestamp(F.concat_ws(" ", F.col("visit_date"), F.col("end_time")))
).alias("a")
v2 = gold_visits.withColumn(
    "start_ts", F.to_timestamp(F.concat_ws(" ", F.col("visit_date"), F.col("start_time")))
).withColumn(
    "end_ts", F.to_timestamp(F.concat_ws(" ", F.col("visit_date"), F.col("end_time")))
).alias("b")
overlaps = visits_for_overlap.join(
    v2,
    (F.col("a.caregiver_id") == F.col("b.caregiver_id"))
    & (F.col("a.visit_date") == F.col("b.visit_date"))
    & (F.col("a.visit_id") < F.col("b.visit_id"))
    & (F.col("a.location_id") != F.col("b.location_id"))
    & F.col("a.start_ts").isNotNull() & F.col("a.end_ts").isNotNull()
    & F.col("b.start_ts").isNotNull() & F.col("b.end_ts").isNotNull()
    & (F.col("a.start_ts") < F.col("b.end_ts"))
    & (F.col("b.start_ts") < F.col("a.end_ts")),
    "inner",
).select(
    F.col("a.caregiver_id").alias("caregiver_id"),
    F.col("a.visit_date").alias("visit_date"),
    F.col("a.visit_id").alias("visit_id_a"),
    F.col("a.location_id").alias("location_id_a"),
    F.col("a.start_time").alias("start_a"), F.col("a.end_time").alias("end_a"),
    F.col("b.visit_id").alias("visit_id_b"),
    F.col("b.location_id").alias("location_id_b"),
    F.col("b.start_time").alias("start_b"), F.col("b.end_time").alias("end_b"),
)
write_lh(overlaps, "gold_caregiver_overlaps")

print("Silver, reconciliation, and Gold pipeline finished.")
print("Review dq_exceptions and reconciliation_report before using Gold outputs.")
