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
