WITH paired AS (
SELECT a.side, a.restricted-b.restricted AS restricted, a.residence-b.residence AS residence, a.rms-b.rms AS rms
FROM trials a JOIN trials b ON a.condition=b.condition AND a.replicate=b.replicate
WHERE a.side<>'' AND a.grp='响应' AND b.grp='无响应'),
long_delta AS (
SELECT side,'限制首次进入时间 / s' AS metric,restricted AS delta FROM paired UNION ALL
SELECT side,'驻留比例',residence FROM paired UNION ALL SELECT side,'RMS 温差 / K',rms FROM paired)
SELECT side,metric,AVG(delta) AS difference,lo AS ci_low,hi AS ci_high,display,COUNT(*) AS pairs,4000 AS bootstrap_resamples
FROM long_delta JOIN intervals USING(side,metric) GROUP BY side,metric ORDER BY side,metric;
