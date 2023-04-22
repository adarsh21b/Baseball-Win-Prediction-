DROP TABLE IF EXISTS pitcher_temp;

CREATE TEMPORARY TABLE pitcher_temp AS
SELECT
    pc.`Double`
    , pc.Triple
    , pc.Home_Run
    , pc.Hit
    , pc.Walk
    , pc.atBat
    , pc.outsPlayed
    , pc.team_id
    , pc.game_id
    , pc.Strikeout
    , pc.pitcher
    , pc.Catcher_Interference
    , pc.plateApperance
    , pc.Sacrifice_Bunt_DP
    , pc.Sac_Fly
    , pc.Intent_Walk
    , pc.Hit_By_Pitch
    , DATE(pc.updatedDate) AS pitcher_date
    , CASE WHEN (pc.atBat) != 0 THEN ((2 * pc.`Double`) + (2 * pc.Triple) + (3 * pc.Home_Run)) / (pc.atBat) ELSE 0 END AS Isolated_power
    , CASE WHEN (pc.outsPlayed / 3) != 0 THEN ((pc.Hit + pc.Walk) / (pc.outsplayed / 3)) ELSE 0 END AS Whip
    , CASE WHEN (pc.outsPlayed / 3) != 0 THEN (3 + (((13 * (pc.Home_Run)) + (3 * ((pc.Walk) + (pc.Hit_By_Pitch))) - (2 * (pc.Strikeout))) / (pc.outsplayed / 3))) ELSE 0 END AS DICE
    , CASE WHEN (pc.outsplayed / 3) != 0 THEN (pc.Home_Run / (9 * (pc.outsplayed / 3))) ELSE 0 END AS Home_Run_per_Nine_Innings
    , CASE WHEN (pc.outsplayed / 3) != 0 THEN (pc.Hit  / (9 * (pc.outsplayed / 3))) ELSE 0 END AS Hits_per_Nine_Innings
    , CASE WHEN (pc.plateApperance - pc.Walk - pc.Hit_By_Pitch - pc.Sacrifice_Bunt_DP - pc.Sac_Fly - pc.Catcher_Interference ) != 0 THEN (pc.Hit / (pc.plateApperance - pc.Walk - pc.Hit_By_Pitch - pc.Sacrifice_Bunt_DP - pc.Sac_Fly - pc.Catcher_Interference )) ELSE 0 END AS Batting_Avg_Against
    , (9 * (pc.Hit + pc.Walk + pc.Hit_By_Pitch ) * (0.89 * (1.255 * (pc.Hit - pc.Home_Run) + 4 * (pc.Home_Run)) + 0.56 * (pc.Walk + pc.Hit_By_Pitch - pc.Intent_Walk))) AS CERA
    , CASE WHEN (pc.outsplayed / 3) != 0 THEN ((pc.Strikeout + pc.Walk) / (pc.outsplayed / 3)) ELSE 0 END AS PFR
    , CASE WHEN (pc.Walk) != 0 THEN (pc.Strikeout / pc.Walk) ELSE 0 END AS StrikeOutToWalkRatio
    , pc.Walk + pc.Hit + pc.Hit_By_Pitch AS TimesOnBase
    , g.local_date AS game_date
FROM
    pitcher_counts pc
    LEFT JOIN game g ON pc.game_id = g.game_id
;

CREATE INDEX game_id_idx ON pitcher_counts (game_id);
CREATE INDEX game_date_idx ON pitcher_counts (game_date);
CREATE INDEX team_id_idx ON pitcher_counts (team_id);

DROP TABLE IF EXISTS pitcher_temp_100_days;

CREATE  TEMPORARY TABLE pitcher_temp_100_days AS
SELECT
    CASE WHEN (SUM(pc.atBat)) != 0 THEN ((2 * SUM(pc.`Double`)) + (2 * SUM(pc.Triple)) + (3 * SUM(pc.Home_Run))) / (SUM(pc.atBat)) ELSE 0 END AS Isolated_power_100
    , CASE WHEN (SUM(pc.outsPlayed) / 3) != 0 THEN ((SUM(pc.Hit) + SUM(pc.Walk)) / (SUM(pc.outsplayed) / 3)) ELSE 0 END AS Whip_100
    , CASE WHEN (SUM(pc.outsPlayed) / 3) != 0 THEN (3 + (((13 * (SUM(pc.Home_Run))) + (3 * ((SUM(pc.Walk)) + (SUM(pc.Hit_By_Pitch)))) - (2 * (SUM(pc.Strikeout)))) / (SUM(pc.outsplayed) / 3))) ELSE 0 END AS DICE_100
    , CASE WHEN (SUM(pc.outsplayed) / 3) != 0 THEN (SUM(pc.Home_Run) / (9 * (SUM(pc.outsplayed) / 3))) ELSE 0 END AS Home_Run_per_Nine_Innings_100
    , CASE WHEN (SUM(pc.outsplayed) / 3) != 0 THEN (SUM(pc.Hit) / (9 * (SUM(pc.outsplayed) / 3))) ELSE 0 END AS Hits_per_Nine_Innings_100
    , CASE WHEN (SUM(pc.plateApperance) - SUM(pc.Walk) - SUM(pc.Hit_By_Pitch) - SUM(pc.Sacrifice_Bunt_DP) - SUM(pc.Sac_Fly) - SUM(pc.Catcher_Interference)) != 0 THEN
            (SUM(pc.Hit) / (SUM(pc.plateApperance) - SUM(pc.Walk) - SUM(pc.Hit_By_Pitch) - SUM(pc.Sacrifice_Bunt_DP) - SUM(pc.Sac_Fly) - SUM(pc.Catcher_Interference))) ELSE 0 END AS Batting_Avg_Against_100
    , (9 * (SUM(pc.Hit) + SUM(pc.Walk) + SUM(pc.Hit_By_Pitch)) * (0.89 * (1.255 * (SUM(pc.Hit) - SUM(pc.Home_Run)) + 4 * (SUM(pc.Home_Run))) + 0.56 * (SUM(pc.Walk) + SUM(pc.Hit_By_Pitch) - SUM(pc.Intent_Walk)))) AS CERA_100
    , CASE WHEN (SUM(pc.outsplayed) / 3) != 0 THEN ((SUM(pc.Strikeout) + SUM(pc.Walk)) / (SUM(pc.outsplayed) / 3)) ELSE 0 END AS PFR_100
    , CASE WHEN (SUM(pc.Walk)) != 0 THEN (SUM(pc.Strikeout) / SUM(pc.Walk)) ELSE 0 END AS StrikeOutToWalkRatio_100
    , SUM(pc.Walk) + SUM(pc.Hit) + SUM(pc.Hit_By_Pitch) AS TimesOnBase_100
FROM
    pitcher_temp pc1
    LEFT JOIN pitcher_temp pc ON pc1.team_id = pc.team_id
        AND pc.game_date >= DATE_ADD(pc1.game_date, INTERVAL -100 DAY)
        AND pc.game_date < pc1.game_date
GROUP BY pc1.game_id, pc1.team_id
ORDER BY pc1.game_id, pc1.team_id
;

CREATE INDEX game_id_idx_100 ON pitcher_temp_100_days (game_id);
CREATE INDEX game_date_idx_100 ON pitcher_temp_100_days (game_date);
CREATE INDEX team_id_idx_100 ON pitcher_temp_100_days (team_id);

DROP TABLE IF EXISTS final_baseball;

CREATE TABLE final_baseball AS
SELECT
    CASE WHEN b.winner_home_or_away = 'A' THEN 0 WHEN b.winner_home_or_away = 'H' THEN 1 END AS Home_Team_Wins
    , (pt_home.Isolated_power_100  - pt_away.Isolated_power_100 ) AS Isolated_Power_Diff
    , (pt_home.DICE_100 - pt_away.DICE_100 ) AS DICE_diff
    , (pt_home.Whip_100 - pt_away.Whip_100 ) AS Whip_100_diff
    , (pt_home.Home_Run_per_Nine_Innings_100 - pt_away.Home_Run_per_Nine_Innings_100 ) AS Home_Run_per_Nine_Innings_diff
    , (pt_home.Hits_per_Nine_Innings_100 - pt_away.Hits_per_Nine_Innings_100 ) AS Hits_per_Nine_Innings_diff
    , (pt_home.Batting_Avg_Against_100 - pt_away.Batting_Avg_Against_100 ) AS Batting_Avg_Against_diff
    , (pt_home.CERA_100 - pt_away.CERA_100 ) AS CERA_diff
    , (pt_home.PFR_100 - pt_away.PFR_100 ) AS PFR_diff
    , (pt_home.TimesOnBase_100 - pt_away.TimesOnBase_100) AS TimesOnBaseDiff
    , (pt_home.StrikeOutToWalkRatio_100 - pt_away.StrikeOutToWalkRatio_100) AS StrikeOutToWalkRatio_diff
FROM
    game g
    LEFT JOIN pitcher_temp_100_days AS pt_away ON pt_away.game_id = g.game_id
        AND pt_away.pitcher = g.away_pitcher
        AND pt_away.team_id = g.away_team_id
    LEFT JOIN pitcher_temp_100_days pt_home ON pt_home.game_id = g.game_id
        AND pt_home.pitcher = g.home_pitcher
        AND pt_home.team_id = g.home_team_id
    LEFT JOIN boxscore b ON g.game_id = b.game_id
WHERE b.winner_home_or_away != ''
;
