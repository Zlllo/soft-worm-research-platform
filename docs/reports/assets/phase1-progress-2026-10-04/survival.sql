SELECT grid.t AS time_s,
AVG(CASE WHEN trials.reached=1 AND (trials.entry<grid.t OR (grid.phase=1 AND trials.entry=grid.t)) THEN 0.0 ELSE 1.0 END) AS survival,
grid.side || '·' || grid.grp AS series,COUNT(*) AS pairs,180 AS window_s
FROM grid JOIN trials ON grid.side=trials.side AND grid.grp=trials.grp
GROUP BY grid.side,grid.grp,grid.t,grid.phase ORDER BY series,grid.t,grid.phase;
