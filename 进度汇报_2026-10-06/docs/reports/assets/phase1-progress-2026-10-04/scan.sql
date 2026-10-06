SELECT 'σ=' || printf('%g',a.noise) || ' K；τ慢=' || printf('%g',a.tau) || ' s' AS condition,
a.noise AS noise_K,a.tau AS slow_memory_s,AVG(a.residence-b.residence) AS residence_difference,COUNT(*) AS pairs
FROM trials a JOIN trials b ON a.condition=b.condition AND a.replicate=b.replicate
WHERE a.side='' AND a.grp='响应' AND b.grp='无响应'
GROUP BY a.noise,a.tau ORDER BY a.noise,a.tau;
