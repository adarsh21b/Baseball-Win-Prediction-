USE baseball;

DROP TABLE IF EXISTS batter_sums_new;

CREATE TEMPORARY TABLE batter_sums_new AS
SELECT g.game_id
    , batter
    , YEAR(g.local_date) AS game_year
    , Date(g.local_date) AS game_date
    , SUM(b.Hit) AS hit_sum
    , SUM(b.atBat) AS atBat_sum
FROM batter_counts b JOIN game g ON b.game_id = g.game_id
GROUP BY batter, game_date, g.game_id
;

CREATE INDEX IF NOT EXISTS ind_batter_counts_game_id ON batter_counts (game_id);
CREATE INDEX IF NOT EXISTS ind_game_local_date ON game (local_date);
CREATE INDEX IF NOT EXISTS index_batter_rolling ON batter_sums_new(batter);
CREATE INDEX IF NOT EXISTS index__date_rolling_new ON batter_sums_new(game_date);

DROP TABLE IF EXISTS Rolling_AVG_100_day;

CREATE TABLE Rolling_AVG_100_day AS
SELECT b2.game_id
    , b2.batter
    , DATE(b2.game_date) AS game_date
    , ROUND(COALESCE(SUM(b1.hit_sum) / NULLIF(SUM(b1.atbat_sum), 0), 0), 4) AS  rolling_avg
FROM batter_sums_new AS b2 LEFT JOIN batter_sums_new  AS b1 ON ( DATEDIFF(b2.game_date, b1.game_date) <= 100
    AND DATEDIFF(b2.game_date, b1.game_date) > 0 AND b2.batter = b1.batter)
WHERE  b2.game_id = 12560
GROUP BY b2.game_id, b2.batter, b2.game_date ORDER BY b2.batter, b2.game_date
;
