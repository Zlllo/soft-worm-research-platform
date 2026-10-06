SELECT side, grp AS "group", COUNT(*) AS pairs, AVG(reached) AS reached,
arrival_ci.display AS reached_ci, AVG(restricted) AS restricted_s, AVG(residence) AS residence, AVG(rms) AS rms_K
FROM trials JOIN arrival_ci USING(side,grp) WHERE side <> '' GROUP BY side,grp ORDER BY side,grp;
