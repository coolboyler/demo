# 假期分流模型评估

- 数据集：`/Users/cayron/work/demo/pipeline/tianlang/new/baseline_d6_dataset_2024fill.csv`
- 普通日走规则 D-6。
- 节假日前后和节日本身按节日家族路由，只有训练回测优于基线的家族才激活。
- 纯调休工作日 `makeup_workday_w*` 单独走调休专模。

## 家族激活结果
| holiday_family | train_rows | special_daily_accuracy_percent | base_daily_accuracy_percent | accuracy_lift_percent | is_active |
| --- | --- | --- | --- | --- | --- |
| Dragon Boat Festival | 9 | 90.62511276382747 | 83.53697512611427 | 7.088137637713203 | 1 |
| Labour Day | 11 | 97.13074204796956 | 76.57163113932029 | 20.559110908649274 | 1 |
| Mid-autumn Festival | 2 | 100.0 | 41.30528204957006 | 58.69471795042994 | 1 |
| National Day | 12 | 84.63152657558307 | 65.92049547898542 | 18.71103109659765 | 1 |
| Spring Festival | 14 | 84.16813418425903 | -52.022792817747444 | 136.19092700200648 | 1 |
| Tomb-sweeping Day | 9 | 96.98559917628008 | 80.45214907465638 | 16.5334501016237 | 1 |
| New Year's Day | 3 | 96.99512147731592 | 96.97497334052866 | 0.02014813678725602 | 0 |
| __generic_makeup__ | 3 | 62.00125637061856 | 63.11094514776158 | -1.1096887771430204 | 0 |
| __ordinary_weekend__ | 162 | 93.70445759545366 | 89.15167044030167 | 4.552787155151989 | 1 |
| __ordinary_workday__ | 406 | 92.89294150388457 | 90.2901223125176 | 2.602819191366976 | 1 |

## 总体分段评估
| model_variant | segment | days | hourly_accuracy_percent | daily_accuracy_percent | daily_bias_percent |
| --- | --- | --- | --- | --- | --- |
| base_rule_d6 | validation | 31 | 88.86166237241203 | 89.84867152277101 | 1.1479424140811327 |
| base_rule_d6 | test | 37 | 23.76666855135501 | 23.79303216203529 | 19.80190859397659 |
| holiday_router | validation | 31 | 84.39183656694746 | 85.33323036709263 | -5.502532187335088 |
| holiday_router | test | 37 | 72.25226728109509 | 73.33438893815269 | 7.3178523694893 |

## 月度评估
| model_variant | split | year_month | days | hourly_accuracy_percent | daily_accuracy_percent | daily_bias_percent |
| --- | --- | --- | --- | --- | --- | --- |
| base_rule_d6 | validation | 2026-01 | 31 | 88.86166237241203 | 89.84867152277101 | 1.1479424140811327 |
| base_rule_d6 | test | 2026-02 | 28 | 11.820694660126762 | 11.863689429213892 | 68.43858057602662 |
| base_rule_d6 | test | 2026-03 | 9 | 42.70335336678356 | 42.70335336678356 | -57.29664663321644 |
| holiday_router | validation | 2026-01 | 31 | 84.39183656694746 | 85.33323036709263 | -5.502532187335088 |
| holiday_router | test | 2026-02 | 28 | 63.54861923273427 | 64.46906362777003 | 19.663285800732098 |
| holiday_router | test | 2026-03 | 9 | 86.04923698293348 | 87.38764844780471 | -12.252053267836988 |

## 被路由的日期
| target_date | split | holiday_family | holiday_segment | route_name | actual_daily_total | pred_daily_total | similar_reference_dates | similar_reference_tags | similar_reference_scores | similar_reference_weights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-07 00:00:00 | validation |  | other | ordinary_similar:workday | 199.49413 | 196.93899012772994 | 2025-01-08|2024-12-25|2025-01-15 | after:New Year's Day:w1|before:New Year's Day:w1|before:Spring Festival:w2 | 48.666667|24.916667|24.583333 | 0.495756|0.253820|0.250424 |
| 2026-01-08 00:00:00 | validation |  | other | ordinary_similar:workday | 199.73390999999998 | 178.68415548806456 | 2025-01-09|2024-12-26|2025-01-16 | after:New Year's Day:w2|before:New Year's Day:w1|before:Spring Festival:w2 | 37.166667|24.916667|24.583333 | 0.428846|0.287500|0.283654 |
| 2026-01-09 00:00:00 | validation |  | other | ordinary_similar:workday | 200.35656 | 181.02531310610527 | 2025-01-10|2024-12-27|2025-01-17 | after:New Year's Day:w2|before:New Year's Day:w1|before:Spring Festival:w2 | 37.166667|24.916667|24.583333 | 0.428846|0.287500|0.283654 |
| 2026-01-10 00:00:00 | validation |  | other | ordinary_similar:weekend | 192.95391999999998 | 179.04661503073598 | 2025-01-11|2024-12-28|2025-01-18 | after:New Year's Day:w2|before:New Year's Day:w1|before:Spring Festival:w2 | 37.166667|24.916667|24.583333 | 0.428846|0.287500|0.283654 |
| 2026-01-11 00:00:00 | validation |  | other | ordinary_similar:weekend | 151.03354 | 164.51100058671807 | 2025-01-12|2025-01-05|2025-01-19 | after:New Year's Day:w2|after:New Year's Day:w1|before:Spring Festival:w2 | 48.666667|31.250000|27.083333 | 0.454829|0.292056|0.253115 |
| 2026-01-12 00:00:00 | validation |  | other | ordinary_similar:workday | 185.93215 | 173.10214746854356 | 2025-01-13|2025-01-06|2025-01-20 | after:New Year's Day:w2|after:New Year's Day:w1|before:Spring Festival:w2 | 48.666667|31.250000|27.083333 | 0.454829|0.292056|0.253115 |
| 2026-01-13 00:00:00 | validation |  | other | ordinary_similar:workday | 197.21783 | 170.44264968142386 | 2025-01-14|2025-01-07|2025-01-21 | after:New Year's Day:w2|after:New Year's Day:w1|before:Spring Festival:w1 | 48.666667|31.250000|24.583333 | 0.465710|0.299043|0.235247 |
| 2026-01-14 00:00:00 | validation |  | other | ordinary_similar:workday | 199.45696999999998 | 194.0180472389736 | 2025-01-15|2025-01-08|2026-01-07 | before:Spring Festival:w2|after:New Year's Day:w1|after:New Year's Day:w1 | 33.666667|31.250000|25.494444 | 0.372373|0.345643|0.281984 |
| 2026-01-15 00:00:00 | validation |  | other | ordinary_similar:workday | 198.95517999999998 | 180.9091350840829 | 2025-01-09|2025-01-16|2026-01-08 | after:New Year's Day:w2|before:Spring Festival:w2|after:New Year's Day:w1 | 42.750000|33.666667|25.494444 | 0.419483|0.330353|0.250164 |
| 2026-01-16 00:00:00 | validation |  | other | ordinary_similar:workday | 196.24609 | 182.6407234912685 | 2025-01-10|2025-01-17|2026-01-09 | after:New Year's Day:w2|before:Spring Festival:w2|after:New Year's Day:w1 | 42.750000|33.666667|25.494444 | 0.419483|0.330353|0.250164 |
| 2026-01-17 00:00:00 | validation |  | other | ordinary_similar:weekend | 188.37523 | 179.35630558133997 | 2025-01-11|2025-01-18|2026-01-10 | after:New Year's Day:w2|before:Spring Festival:w2|after:New Year's Day:w1 | 42.750000|33.666667|25.494444 | 0.419483|0.330353|0.250164 |
| 2026-01-18 00:00:00 | validation |  | other | ordinary_similar:weekend | 147.66226 | 164.77084696728326 | 2025-01-12|2025-01-19|2025-01-05 | after:New Year's Day:w2|before:Spring Festival:w2|after:New Year's Day:w1 | 31.250000|31.166667|27.166667 | 0.348837|0.347907|0.303256 |
| 2026-01-19 00:00:00 | validation |  | other | ordinary_similar:workday | 188.67225000000002 | 169.96861941958153 | 2025-01-13|2025-01-20|2025-01-06 | after:New Year's Day:w2|before:Spring Festival:w2|after:New Year's Day:w1 | 31.250000|31.166667|27.166667 | 0.348837|0.347907|0.303256 |
| 2026-01-20 00:00:00 | validation |  | other | ordinary_similar:workday | 207.37941 | 165.7980059181209 | 2025-01-14|2025-01-21|2025-01-07 | after:New Year's Day:w2|before:Spring Festival:w1|after:New Year's Day:w1 | 31.250000|31.166667|27.166667 | 0.348837|0.347907|0.303256 |
| 2026-01-21 00:00:00 | validation |  | other | ordinary_similar:workday | 206.6461 | 167.7602022110226 | 2025-01-22|2025-01-08|2026-01-14 | before:Spring Festival:w1|after:New Year's Day:w1|after:New Year's Day:w2 | 31.166667|27.166667|25.494444 | 0.371794|0.324077|0.304129 |
| 2026-01-22 00:00:00 | validation |  | other | ordinary_similar:workday | 211.32305 | 155.52709965493196 | 2025-01-23|2025-01-09|2026-01-15 | before:Spring Festival:w1|after:New Year's Day:w2|after:New Year's Day:w2 | 31.166667|27.166667|25.494444 | 0.371794|0.324077|0.304129 |
| 2026-01-23 00:00:00 | validation |  | other | ordinary_similar:workday | 213.8347 | 154.85801613640055 | 2025-01-24|2025-01-10|2026-01-16 | before:Spring Festival:w1|after:New Year's Day:w2|after:New Year's Day:w2 | 31.166667|27.166667|25.494444 | 0.371794|0.324077|0.304129 |
| 2026-01-24 00:00:00 | validation |  | other | ordinary_similar:weekend | 195.19712 | 190.35274540265488 | 2025-01-11|2026-01-17|2026-01-10 | after:New Year's Day:w2|after:New Year's Day:w2|after:New Year's Day:w1 | 27.166667|25.494444|25.455556 | 0.347770|0.326364|0.325866 |
| 2026-01-25 00:00:00 | validation |  | other | ordinary_similar:weekend | 157.77241 | 156.02724656882478 | 2025-01-19|2024-01-28|2025-02-23 | before:Spring Festival:w2|before:Spring Festival:w2|after:Spring Festival:w3 | 31.250000|25.500000|23.750000 | 0.388199|0.316770|0.295031 |
| 2026-01-26 00:00:00 | validation |  | other | ordinary_similar:workday | 187.22114 | 159.56696455280917 | 2024-01-22|2025-01-20|2024-01-29 | before:Spring Festival:w3|before:Spring Festival:w2|before:Spring Festival:w2 | 36.500000|31.250000|25.500000 | 0.391421|0.335121|0.273458 |
| 2026-01-27 00:00:00 | validation |  | other | ordinary_similar:workday | 190.73505999999998 | 157.50242299664922 | 2024-01-23|2025-01-21|2024-01-30 | before:Spring Festival:w3|before:Spring Festival:w1|before:Spring Festival:w2 | 36.500000|31.250000|25.500000 | 0.391421|0.335121|0.273458 |
| 2026-01-28 00:00:00 | validation |  | other | ordinary_similar:workday | 187.69418000000002 | 157.78912492281344 | 2024-01-24|2025-01-22|2025-01-15 | before:Spring Festival:w3|before:Spring Festival:w1|before:Spring Festival:w2 | 36.500000|31.250000|27.166667 | 0.384548|0.329236|0.286216 |
| 2026-01-29 00:00:00 | validation |  | other | ordinary_similar:workday | 192.63412 | 149.78916549354997 | 2024-01-25|2025-01-23|2025-01-16 | before:Spring Festival:w3|before:Spring Festival:w1|before:Spring Festival:w2 | 36.500000|31.250000|27.166667 | 0.384548|0.329236|0.286216 |
| 2026-01-30 00:00:00 | validation |  | other | ordinary_similar:workday | 188.1111 | 145.74490529462162 | 2024-01-26|2025-01-24|2025-01-17 | before:Spring Festival:w3|before:Spring Festival:w1|before:Spring Festival:w2 | 36.500000|31.250000|27.166667 | 0.384548|0.329236|0.286216 |
| 2026-01-31 00:00:00 | validation |  | other | ordinary_similar:weekend | 212.76416 | 175.05767873171627 | 2025-01-18|2024-01-27|2026-01-24 | before:Spring Festival:w2|before:Spring Festival:w2|after:New Year's Day:w3 | 27.166667|25.000000|21.994444 | 0.366320|0.337104|0.296577 |
| 2026-02-01 00:00:00 | test |  | other | ordinary_similar:weekend | 145.12928 | 143.3787095067537 | 2025-01-19|2024-01-28|2025-02-09 | before:Spring Festival:w2|before:Spring Festival:w2|after:Spring Festival:w1 | 35.916667|32.750000|28.583333 | 0.369323|0.336761|0.293916 |
| 2026-02-02 00:00:00 | test |  | other | ordinary_similar:workday | 161.43041 | 148.47267984557524 | 2025-01-20|2024-01-29|2025-02-10 | before:Spring Festival:w2|before:Spring Festival:w2|after:Spring Festival:w1 | 35.916667|32.750000|28.583333 | 0.369323|0.336761|0.293916 |
| 2026-02-03 00:00:00 | test |  | other | ordinary_similar:workday | 161.74930999999998 | 174.27135602277704 | 2024-01-30|2025-02-11|2025-02-18 | before:Spring Festival:w2|after:Spring Festival:w1|after:Spring Festival:w2 | 32.750000|28.583333|27.000000 | 0.370755|0.323585|0.305660 |
| 2026-02-04 00:00:00 | test |  | other | ordinary_similar:workday | 151.92070999999999 | 170.56725586806397 | 2024-01-31|2025-01-15|2025-02-12 | before:Spring Festival:w2|before:Spring Festival:w2|after:Spring Festival:w2 | 32.750000|32.750000|31.083333 | 0.339085|0.339085|0.321829 |
| 2026-02-05 00:00:00 | test |  | other | ordinary_similar:workday | 138.89411 | 172.05853749400444 | 2024-02-01|2025-01-16|2025-02-13 | before:Spring Festival:w2|before:Spring Festival:w2|after:Spring Festival:w2 | 36.500000|32.750000|31.083333 | 0.363787|0.326412|0.309801 |
| 2026-02-06 00:00:00 | test |  | other | ordinary_similar:workday | 119.40401000000001 | 170.4857845062404 | 2024-02-02|2025-01-17|2025-02-14 | before:Spring Festival:w2|before:Spring Festival:w2|after:Spring Festival:w2 | 36.500000|32.750000|31.083333 | 0.363787|0.326412|0.309801 |
| 2026-02-07 00:00:00 | test |  | other | ordinary_similar:weekend | 99.65304 | 167.36151023014372 | 2024-01-27|2025-01-18|2025-02-15 | before:Spring Festival:w2|before:Spring Festival:w2|after:Spring Festival:w2 | 32.750000|32.750000|31.083333 | 0.339085|0.339085|0.321829 |
| 2026-02-08 00:00:00 | test |  | other | ordinary_similar:weekend | 78.00518 | 140.3560008602798 | 2025-02-09|2025-02-16|2026-02-01 | after:Spring Festival:w1|after:Spring Festival:w2|before:Spring Festival:w2 | 37.666667|28.583333|25.494444 | 0.410561|0.311554|0.277885 |
| 2026-02-09 00:00:00 | test |  | other | ordinary_similar:workday | 65.06534 | 162.56201567090685 | 2025-02-10|2024-02-05|2025-02-17 | after:Spring Festival:w1|before:Spring Festival:w1|after:Spring Festival:w2 | 37.666667|36.500000|28.583333 | 0.366586|0.355231|0.278183 |
| 2026-02-10 00:00:00 | test |  | other | ordinary_similar:workday | 53.83794 | 147.06430893224106 | 2025-02-11|2024-02-06|2025-01-21 | after:Spring Festival:w1|before:Spring Festival:w1|before:Spring Festival:w1 | 37.666667|36.500000|32.750000 | 0.352299|0.341387|0.306313 |
| 2026-02-11 00:00:00 | test |  | other | ordinary_similar:workday | 48.05767 | 143.0910836802828 | 2025-02-12|2025-01-22|2025-02-19 | after:Spring Festival:w2|before:Spring Festival:w1|after:Spring Festival:w3 | 35.166667|32.750000|28.583333 | 0.364421|0.339378|0.296200 |
| 2026-02-12 00:00:00 | test | Spring Festival | pre | holiday_family:Spring Festival | 44.057860000000005 | 38.35426 |  |  |  |  |
| 2026-02-13 00:00:00 | test | Spring Festival | pre | holiday_family:Spring Festival | 42.37535 | 36.29208 |  |  |  |  |
| 2026-02-14 00:00:00 | test | Spring Festival | pre | holiday_family:Spring Festival | 36.43889 | 32.54136 |  |  |  |  |
| 2026-02-15 00:00:00 | test | Spring Festival | holiday | holiday_family:Spring Festival | 34.46095 | 30.57832 |  |  |  |  |
| 2026-02-16 00:00:00 | test | Spring Festival | holiday | holiday_family:Spring Festival | 33.8435 | 29.453665 |  |  |  |  |
| 2026-02-17 00:00:00 | test | Spring Festival | holiday | holiday_family:Spring Festival | 33.05507 | 29.18533 |  |  |  |  |
| 2026-02-18 00:00:00 | test | Spring Festival | holiday | holiday_family:Spring Festival | 32.9267 | 45.796127500000004 |  |  |  |  |
| 2026-02-19 00:00:00 | test | Spring Festival | holiday | holiday_family:Spring Festival | 33.99973 | 43.57836999999999 |  |  |  |  |
| 2026-02-20 00:00:00 | test | Spring Festival | holiday | holiday_family:Spring Festival | 34.74053000000001 | 31.73568625 |  |  |  |  |
| 2026-02-21 00:00:00 | test | Spring Festival | holiday | holiday_family:Spring Festival | 36.01708 | 32.360277499999995 |  |  |  |  |
| 2026-02-22 00:00:00 | test | Spring Festival | holiday | holiday_family:Spring Festival | 36.10798 | 32.99019875 |  |  |  |  |
| 2026-02-23 00:00:00 | test | Spring Festival | holiday | holiday_family:Spring Festival | 39.51663 | 36.991249999999994 |  |  |  |  |
| 2026-02-24 00:00:00 | test | Spring Festival | post | holiday_family:Spring Festival | 52.02335 | 46.47628 |  |  |  |  |
| 2026-02-25 00:00:00 | test | Spring Festival | post | holiday_family:Spring Festival | 70.36535 | 67.46796 |  |  |  |  |
| 2026-02-26 00:00:00 | test | Spring Festival | post | holiday_family:Spring Festival | 89.62225000000001 | 85.84483 |  |  |  |  |
| 2026-02-27 00:00:00 | test |  | other | ordinary_similar:workday | 104.64878999999999 | 137.19393527412936 | 2025-02-28|2024-02-23|2025-02-21 | after:Spring Festival:w4|after:Spring Festival:w1|after:Spring Festival:w3 | 37.166667|36.500000|31.250000 | 0.354249|0.347895|0.297855 |
| 2026-03-01 00:00:00 | test |  | other | ordinary_similar:weekend | 106.72689 | 103.64205060457328 | 2025-03-02|2025-02-09|2025-02-23 | after:Spring Festival:w4|after:Spring Festival:w1|after:Spring Festival:w3 | 37.166667|32.750000|28.500000 | 0.377646|0.332769|0.289585 |
| 2026-03-02 00:00:00 | test |  | other | ordinary_similar:workday | 142.10988 | 120.8173536648292 | 2025-03-03|2025-02-10|2025-02-24 | after:Spring Festival:w4|after:Spring Festival:w1|after:Spring Festival:w3 | 37.166667|32.750000|28.500000 | 0.377646|0.332769|0.289585 |
| 2026-03-03 00:00:00 | test |  | other | ordinary_similar:workday | 133.3215 | 135.73556856520213 | 2025-03-04|2025-02-18|2024-02-27 | after:Spring Festival:w4|after:Spring Festival:w2|after:Spring Festival:w2 | 37.166667|35.916667|32.750000 | 0.351181|0.339370|0.309449 |
| 2026-03-04 00:00:00 | test |  | other | ordinary_similar:workday | 154.7264 | 123.51009228517195 | 2024-02-28|2025-02-12|2025-03-05 | after:Spring Festival:w2|after:Spring Festival:w2|ordinary:workday | 32.750000|32.750000|31.166667 | 0.338793|0.338793|0.322414 |
| 2026-03-05 00:00:00 | test |  | other | ordinary_similar:workday | 159.3516 | 131.41447328282595 | 2024-02-29|2025-02-13|2025-03-06 | after:Spring Festival:w2|after:Spring Festival:w2|ordinary:workday | 32.750000|32.750000|31.166667 | 0.338793|0.338793|0.322414 |
| 2026-03-06 00:00:00 | test |  | other | ordinary_similar:workday | 163.78331 | 144.78758594639396 | 2024-03-01|2025-02-14|2025-03-07 | after:Spring Festival:w2|after:Spring Festival:w2|before:Tomb-sweeping Day:w4 | 36.000000|32.750000|31.166667 | 0.360300|0.327773|0.311927 |
| 2026-03-07 00:00:00 | test |  | other | ordinary_similar:weekend | 165.04480999999998 | 146.6897608375952 | 2024-03-02|2025-02-15|2025-03-01 | after:Spring Festival:w2|after:Spring Festival:w2|after:Spring Festival:w4 | 36.000000|32.750000|31.250000 | 0.360000|0.327500|0.312500 |
| 2026-03-08 00:00:00 | test |  | other | ordinary_similar:weekend | 135.2253 | 122.23510264602277 | 2024-02-25|2025-02-16|2025-03-02 | after:Spring Festival:w2|after:Spring Festival:w2|after:Spring Festival:w4 | 32.750000|32.750000|31.250000 | 0.338501|0.338501|0.322997 |
| 2026-03-09 00:00:00 | test |  | other | ordinary_similar:workday | 179.74920000000003 | 147.0246235548542 | 2024-02-26|2025-02-17|2025-03-03 | after:Spring Festival:w2|after:Spring Festival:w2|after:Spring Festival:w4 | 32.750000|32.750000|31.250000 | 0.338501|0.338501|0.322997 |

- 预测输出：`/Users/cayron/work/demo/pipeline/tianlang/results/best_d6_tianlang_2024fill_shared_test_daily.csv`
- 摘要输出：`/Users/cayron/work/demo/pipeline/tianlang/results/best_d6_tianlang_2024fill_shared_summary.json`