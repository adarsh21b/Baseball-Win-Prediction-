DROP TABLE IF EXISTS annual_avg;

CREATE TABLE annual_avg AS
SELECT b.batter, YEAR(g.local_date) AS Y, CASE WHEN SUM(b.atBat) = 0 THEN 0 ELSE SUM(b.Hit) / SUM(b.atBat) END AS batter_annual_avg
FROM batter_counts b JOIN game g ON b.game_id = g.game_id GROUP BY b.batter, YEAR(g.local_date) ORDER BY b.batter, YEAR(g.local_date)
;

CREATE TABLE historic_avg AS
SELECT b.batter, CASE WHEN SUM(b.atBat) = 0 THEN 0 ELSE SUM(b.Hit) / SUM(b.atBat) END AS batter_historic_avg
FROM batter_counts b JOIN game g ON b.game_id = g.game_id GROUP BY b.batter ORDER BY b.batter
;

SELECT g.game_id, bc.batter, DATE(g.local_date) AS game_date, bc.atBat AS at_bat, bc.Hit AS hit, IF(SUM(bc2.at_bat) = 0, 0, SUM(bc2.hit) / SUM(bc2.at_bat)) AS rolling_avg
FROM batter_counts bc JOIN game g ON bc.game_id = g.game_id LEFT JOIN batter_counts bc2 ON bc.batter = bc2.batter AND bc2.game_id != bc.game_id AND g.local_date >= DATE_SUB(DATE(bc2.game_id), INTERVAL 100 DAY) AND g.local_date < DATE(bc2.local_date) GROUP BY g.game_id, bc.batter, game_date ORDER BY bc.batter, game_date
;
