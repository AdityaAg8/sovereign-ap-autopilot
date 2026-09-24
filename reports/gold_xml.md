# Gold PINT AE XML (M2.5)

ERP snapshot: seed `2026`. Documents validated: **5000**. Schematron pass rate: **5000/5000 (100.00%)**.

## Counts per split

| Split | Documents | Passed | Pass rate |
|---|---:|---:|---:|
| train | 3500 | 3500 | 100.00% |
| cal | 750 | 750 | 100.00% |
| test | 750 | 750 | 100.00% |

## Counts per tax case

| Tax case | Documents | Passed | Pass rate |
|---|---:|---:|---:|
| designated_zone | 964 | 964 | 100.00% |
| exempt | 1039 | 1039 | 100.00% |
| reverse_charge | 966 | 966 | 100.00% |
| standard | 1059 | 1059 | 100.00% |
| zero_rated | 972 | 972 | 100.00% |

## Credit notes

0 of 5000 documents are credit notes. The M2 factory (`factory/build.py`, `factory/layouts.py`'s `TaxCase`) generates invoices only -- every gold document here is a PINT AE `Invoice` (`DocumentTypeCode.INVOICE`, type code "380").

## Failures by rule ID

None -- every document passed the runtime PINT AE schematron validator.
