# Factory label self-check (M2.4)

pdfplumber text at each DOM-labeled box vs. the label's ground-truth value, on every digital PDF in the M2.3 UAE invoice dataset.

**Overall exact match: 628408/629210 (99.87%) -- PASS** (threshold 99.5%)

## By language

| script | exact match |
|---|---|
| latin | 558675/558715 (99.99%) |
| arabic | 69733/70495 (98.92%) |

## By split

| split | exact match |
|---|---|
| cal | 93024/93157 (99.86%) |
| test | 96363/96444 (99.92%) |
| train | 439021/439609 (99.87%) |

## By layout

| layout | exact match |
|---|---|
| ar_bold_corporate | 38950/38950 (100.00%) |
| ar_compact_utility | 40405/40405 (100.00%) |
| ar_ledger_classic | 35418/35418 (100.00%) |
| ar_modern_minimal | 33465/33465 (100.00%) |
| bilingual_bold_corporate | 65434/65438 (99.99%) |
| bilingual_compact_utility | 62630/62630 (100.00%) |
| bilingual_ledger_classic | 67186/67186 (100.00%) |
| bilingual_modern_minimal | 64674/65472 (98.78%) |
| en_bold_corporate | 55297/55297 (100.00%) |
| en_compact_utility | 54947/54947 (100.00%) |
| en_ledger_classic | 54643/54643 (100.00%) |
| en_modern_minimal | 55359/55359 (100.00%) |

## By field

| field | exact match |
|---|---|
| buyer_name | 7454/7454 (100.00%) |
| buyer_trn | 5000/5000 (100.00%) |
| currency | 5000/5000 (100.00%) |
| description | 140772/141534 (99.46%) |
| due_date | 5000/5000 (100.00%) |
| iban | 4960/5000 (99.20%) |
| invoice_number | 5000/5000 (100.00%) |
| issue_date | 5000/5000 (100.00%) |
| net_amount | 103029/103029 (100.00%) |
| po_number | 5000/5000 (100.00%) |
| quantity | 103029/103029 (100.00%) |
| subtotal | 5000/5000 (100.00%) |
| supplier_address | 5000/5000 (100.00%) |
| supplier_name | 8106/8106 (100.00%) |
| supplier_trn | 5000/5000 (100.00%) |
| total | 5000/5000 (100.00%) |
| unit_price | 103029/103029 (100.00%) |
| vat_rate | 103029/103029 (100.00%) |
| vat_total | 5000/5000 (100.00%) |

## Sample failures

| sha256 | split | layout | field | line | page | expected | extracted |
|---|---|---|---|---|---|---|---|
| 56665a404730 | cal | bilingual_modern_minimal | description | 0 | 2 | `محول شبكة 24 منفذ مُدار جيجابت` | `` |
| 56665a404730 | cal | bilingual_modern_minimal | description | 1 | 2 | `خدمة تموين، غداء بوفيه، للفرد` | `البيان` |
| 56665a404730 | cal | bilingual_modern_minimal | description | 2 | 2 | `ورق تصوير A4 80 جم، كرتون 5 رزم` | `محول شبكةم 24 نفذ مُدار جيجاب` |
| 9beddc76d607 | cal | bilingual_modern_minimal | description | 0 | 2 | `عزل مجاري تكييف 25 مم، لكل متر طولي` | `البيان` |
| 9beddc76d607 | cal | bilingual_modern_minimal | description | 1 | 2 | `مكتب مكتبي 160x80 سم، تشطيب لامينيت` | `عزل مجاري تكييف 52 مم، لكل متر طولي` |
| 9beddc76d607 | cal | bilingual_modern_minimal | description | 2 | 2 | `خدمات نظافة، عقد مكتبي شهري` | `مكتب مكتبي 08x061 سم، تشطي` |
| d2cae7c2138c | cal | bilingual_modern_minimal | description | 0 | 2 | `محول شبكة 24 منفذ مُدار جيجابت` | `` |
| d2cae7c2138c | cal | bilingual_modern_minimal | description | 1 | 2 | `دعم تقني، لكل يوم فني` | `البيان` |
| d2cae7c2138c | cal | bilingual_modern_minimal | description | 2 | 2 | `خدمات نظافة، عقد مكتبي شهري` | `محول شبكة 42 منفذ مُدار جيجابت` |
| e3fe8fae7ca9 | cal | bilingual_modern_minimal | description | 0 | 1 | `ورق تصوير A4 80 جم، كرتون 5 رزم` | `البيان` |
| e3fe8fae7ca9 | cal | bilingual_modern_minimal | iban |  | 1 | `AE140145945028316632337` | `` |
| ec2ecd032a76 | cal | bilingual_modern_minimal | description | 0 | 2 | `خدمة تموين، غداء بوفيه، للفرد` | `` |
| ec2ecd032a76 | cal | bilingual_modern_minimal | description | 1 | 2 | `عزل مجاري تكييف 25 مم، لكل متر طولي` | `البيان` |
| ec2ecd032a76 | cal | bilingual_modern_minimal | description | 2 | 2 | `طفاية حريق 6 كجم بودرة جافة ABC` | `خدمة تموين، غداء بوفيه، للفرد` |
| ff33d9110e76 | cal | bilingual_modern_minimal | description | 0 | 2 | `خرسانة جاهزة درجة C30، لكل متر مكعب` | `` |
| ff33d9110e76 | cal | bilingual_modern_minimal | description | 1 | 2 | `عارضة صلب إنشائية 6 متر، مجلفنة` | `البيان` |
| ff33d9110e76 | cal | bilingual_modern_minimal | description | 2 | 2 | `دعم تقني، لكل يوم فني` | `خرسانة جاهزة درجة 03C، لك` |
| 01c6cd2203a2 | test | bilingual_bold_corporate | iban |  | 3 | `AE370446815188447294055` | `` |
| 0d5bb4492cb8 | test | bilingual_modern_minimal | description | 0 | 2 | `مكتب مكتبي 160x80 سم، تشطيب لامينيت` | `البيان` |
| 0d5bb4492cb8 | test | bilingual_modern_minimal | description | 1 | 2 | `محول شبكة 24 منفذ مُدار جيجابت` | `مكتب مكتبي 08x061 سم، تشطيب` |
