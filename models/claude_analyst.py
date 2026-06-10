"""
models/claude_analyst.py
------------------------
Uses Claude to analyze qualitative factors and adjust win probability.

Claude receives: team stats, recent form, injuries, odds context.
Claude outputs:  adjusted win probability + reasoning.

This runs AFTER the statistical model to overlay qualitative judgment.
"""

import json
import logging
import re

import anthropic

logger = logging.getLogger(__name__)

# Sport-specific analyst personas injected at the top of every prompt.
# Role shapes how Claude frames the problem; framework sets the priority order
# for what actually drives outcomes in that sport.
_ANALYST_PERSONAS: dict[str, dict[str, str]] = {
    "basketball_nba": {
        "role": "an expert NBA betting analyst who specializes in advanced team efficiency metrics",
        "framework": (
            "1. NET RATING is your primary lens — offensive and defensive efficiency matter more than raw record.\n"
            "2. Weight the last 10 games heavily; momentum and form are real in the NBA.\n"
            "3. Rest and back-to-backs are meaningful — a back-to-back team loses ~2-3% win probability.\n"
            "4. In playoffs, home court and series pressure dominate; teams facing elimination often overperform.\n"
            "5. Caution: in playoffs, current form matters more than regular-season record."
        ),
    },
    "baseball_mlb": {
        "role": "an expert MLB betting analyst who weighs starting pitching, team quality, bullpen depth, and park context together",
        "framework": (
            "1. Starting pitcher quality is a meaningful input — ERA, recent form, and handedness against the opposing lineup all matter. "
            "However, starters typically pitch 5-6 innings; a bullpen advantage or disadvantage often decides close games. "
            "A strong starter on a weak team is not an automatic edge.\n"
            "2. Team run differential is a better signal of true quality than win percentage over a 162-game season. "
            "A team with a large run differential edge should temper an unfavorable pitching matchup.\n"
            "3. Bullpen ERA and recent usage matter — a taxed or poor bullpen erases a starter's advantage in the 6th inning onward.\n"
            "4. Home/road splits and park factors are real — some parks inflate offense significantly and affect pitcher ERA.\n"
            "5. Streaks and momentum regress hard in baseball. Be skeptical of hot/cold narratives.\n"
            "6. MLB books price starter quality efficiently — only act on a pitching edge when team quality and bullpen also support the lean. "
            "Do not let a single ERA gap drive the decision when other indicators conflict.\n"
            "7. Default to 'pass' when the edge is below 4-5% or when factors point in opposite directions."
        ),
    },
    "icehockey_nhl": {
        "role": "an expert NHL betting analyst who evaluates goaltender matchups and special teams as the primary drivers of game outcomes",
        "framework": (
            "1. GOALTENDER performance is paramount — save percentage and goals-against average outweigh team record.\n"
            "2. Special teams (power play %, penalty kill %) create consistent, repeatable edges.\n"
            "3. In playoffs, home ice is strong (~58% home win rate); series pressure and elimination urgency are real.\n"
            "4. Playoff hockey is defensive — fade high-scoring teams when they face disciplined, defensive opponents.\n"
            "5. Short rest significantly impacts goaltender performance — treat rest_days ≤ 1 as a back-to-back and discount that team's win probability by 3-5%."
        ),
    },
    "basketball_wnba": {
        "role": "an expert WNBA betting analyst who weights individual player impact heavily due to the league's smaller rosters",
        "framework": (
            "1. With only 11-12 active players, one star's absence can shift win probability by 10-15% — injury data is critical.\n"
            "2. Home court advantage is smaller than the NBA (~53-55% home win rate); don't overweight it.\n"
            "3. Point differential is the best efficiency signal — win percentage is volatile on a short schedule.\n"
            "4. Travel fatigue is significant in the WNBA due to compressed scheduling.\n"
            "5. Market efficiency is lower than NBA/MLB — genuine edges are more likely when roster news is fresh."
        ),
    },
    "soccer_fifa_world_cup": {
        "role": "an expert FIFA World Cup betting analyst who combines team quality, recent international form, and tournament context",
        "framework": (
            "1. Recent international form (last 5 games) is your primary momentum signal, alongside tournament record and goal difference once group games have been played — current form and results matter more than reputation. You are NOT given a reliable FIFA world ranking, so do not invent one or lean on remembered rankings from your training data.\n"
            "2. This is a 3-way market. Your adjusted_home_prob = P(home wins). A home ML bet LOSES on a draw. Draw probability in World Cup group stage is ~25% — a marginal favorite may not offer value once draw probability is accounted for.\n"
            "3. Home/away here is mostly nominal: the great majority of group-stage matches are played at NEUTRAL venues, so the 'home' label carries little or no true home-field advantage. The exception is the host nations (Mexico, USA, Canada): when one of them is playing inside its own country, treat it as a real and meaningful home-crowd boost. Use the Venue line to judge whether the home team is actually playing at home before applying any home advantage.\n"
            "4. Tournament stage context: teams facing elimination play with urgency; teams already qualified for the knockout round may rotate key players — factor this into your assessment when group standings are available.\n"
            "5. Tactical matchups matter in international soccer — a disciplined counter-attacking side vs a high-press possession team can produce a result that surprises pure quality rankings.\n"
            "6. Player absences from injury data are decisive in soccer — a missing striker, goalkeeper, or holding midfielder can shift win probability by 5-10%.\n"
            "7. Suspension data is unavailable — if you strongly suspect a key player is suspended (e.g., yellow card accumulation), flag this uncertainty in your reasoning but do not treat it as confirmed.\n"
            "8. World Cup markets are among the most efficient in sports. Default to 'pass' unless multiple factors align and the edge exceeds 4-5%."
        ),
    },
    "americanfootball_nfl": {
        "role": "an expert NFL betting analyst who treats quarterback play and offensive line health as the foundation of every game evaluation",
        "framework": (
            "1. Starting QB quality is the single biggest factor — a backup QB shifts win probability by 10-15%.\n"
            "2. Offensive line health: use sacks_allowed as a proxy — teams allowing 3+ sacks/game have degraded pass protection.\n"
            "3. Offensive EPA/play and Defensive EPA allowed/play are the most reliable efficiency metrics; weight them above win% and points/game.\n"
            "4. Turnover differential is a strong game-level predictor — teams with +3 or better season differential win at ~65% rate.\n"
            "5. Weather (outdoor only): wind_speed > 15 mph suppresses passing by ~10%; temp < 32°F adds further suppression; adjust both teams equally unless one is a run-first offense.\n"
            "6. Short rest (rest_days ≤ 4, i.e. Thursday games) is worth -3 to -5% win probability for the short-rest team.\n"
            "7. Home field is worth ~3 points; stronger in cold outdoor venues (Lambeau, Arrowhead, Highmark).\n"
            "8. The NFL market is the most efficient of all major sports — default to 'pass' when edges are below 4% or data is thin."
        ),
    },
}

_DEFAULT_PERSONA = {
    "role": "an expert sports betting analyst",
    "framework": "Consider home field advantage, rest/fatigue, recent momentum, injury impact, and matchup style.",
}


class ClaudeAnalyst:

    def __init__(self, api_key: str, model: str = "claude-opus-4-7"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model

    def analyze_game(
        self,
        game: dict,
        home_stats: dict,
        away_stats: dict,
        base_home_prob: float,
        home_injuries: list | None = None,
        away_injuries: list | None = None,
        home_roster: str = "",
        away_roster: str = "",
        sport: str = "basketball_nba",
        series_context: str | None = None,
        starting_pitchers: dict | None = None,
        weather: dict | None = None,
    ) -> dict:
        """
        Asks Claude to analyze the game and return an adjusted home win probability.

        Returns:
            {
                "adjusted_home_prob": float,
                "confidence":         str,   # "low" | "medium" | "high"
                "reasoning":          str,
                "bet_recommendation": str,   # "home_ml" | "away_ml" | "over" | "under" | "pass"
            }
        """
        prompt = self._build_prompt(game, home_stats, away_stats, base_home_prob,
                                    home_injuries or [], away_injuries or [],
                                    home_roster, away_roster, sport, series_context,
                                    starting_pitchers or {}, weather)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text
            return self._parse_response(raw, base_home_prob)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {
                "adjusted_home_prob":  base_home_prob,
                "confidence":          "low",
                "reasoning":           f"Claude unavailable: {e}",
                "bet_recommendation":  "pass",
            }

    def _build_prompt(self, game: dict, home_stats: dict, away_stats: dict,
                      base_prob: float, home_injuries: list, away_injuries: list,
                      home_roster: str, away_roster: str,
                      sport: str = "basketball_nba",
                      series_context: str | None = None,
                      starting_pitchers: dict | None = None,
                      weather: dict | None = None) -> str:
        home = game["home_team"]
        away = game["away_team"]

        def _pct(val) -> str:
            try:
                return f"{float(val):.0%}"
            except (TypeError, ValueError):
                return "N/A"

        def fmt_injuries(injuries: list) -> str:
            if not injuries:
                return "  None reported"
            return "\n".join(
                f"  - {i['player']}: {i['status']}" + (f" ({i['detail']})" if i.get('detail') else "")
                for i in injuries
            )

        def build_record(stats: dict) -> str:
            w   = stats.get("W") or stats.get("wins")
            l   = stats.get("L") or stats.get("losses")
            otl = stats.get("otLosses") or stats.get("OTL")
            ties = stats.get("ties")
            if w is None and l is None:
                return "?-?"
            w_str = int(w) if w is not None else "?"
            l_str = int(l) if l is not None else "?"
            if ties is not None:
                return f"{w_str}-{int(ties)}-{l_str}"
            record = f"{w_str}-{l_str}"
            if otl is not None:
                record += f"-{int(otl)}"
            return record

        def _stat(stats: dict, *keys, fmt=None) -> str:
            for k in keys:
                v = stats.get(k)
                if v is not None:
                    try:
                        f_val = float(v)
                        if str(f_val) == "nan":
                            continue
                        return fmt(f_val) if fmt else str(v)
                    except (ValueError, TypeError):
                        return str(v)
            return "N/A"

        def build_stats_block(stats: dict, side: str = "") -> str:
            if sport == "basketball_nba":
                return "\n".join([
                    f"- Net Rating: {_stat(stats, 'NET_RATING')}",
                    f"- Off Rating: {_stat(stats, 'OFF_RATING')}",
                    f"- Def Rating: {_stat(stats, 'DEF_RATING')}",
                    f"- Pace: {_stat(stats, 'PACE')}",
                    f"- Last 10 games: {_pct(stats.get('win_pct_l10'))} win rate",
                    f"- Back-to-back: {'Yes' if stats.get('is_back_to_back') else 'No'}",
                    f"- Rest days: {stats.get('rest_days', 'N/A')}",
                ])
            if sport == "baseball_mlb":
                sp = starting_pitchers or {}
                pitcher = sp.get(side, {})
                pitcher_line = (
                    f"- Starting pitcher: {pitcher['name']} "
                    f"({pitcher.get('wins','?')}-{pitcher.get('losses','?')}, "
                    f"{pitcher.get('era','?')} ERA)"
                    if pitcher.get("name") and pitcher["name"] != "TBD"
                    else "- Starting pitcher: TBD"
                )
                _int  = lambda v: str(int(v))
                _dec2 = lambda v: f"{v:.2f}"
                streak_raw = stats.get("streak")
                try:
                    sv = int(float(streak_raw))
                    streak_str = f"W{sv}" if sv >= 0 else f"L{abs(sv)}"
                except (TypeError, ValueError):
                    streak_str = "N/A"
                return "\n".join([
                    pitcher_line,
                    f"- Team ERA (all pitchers): {_stat(stats, 'team_era')} | WHIP: {_stat(stats, 'team_whip')} | K/9: {_stat(stats, 'team_k9')} | BB/9: {_stat(stats, 'team_bb9')}",
                    f"- Bullpen ERA: {_stat(stats, 'bullpen_era')}",
                    f"- Offense — OPS: {_stat(stats, 'team_ops')} | OBP: {_stat(stats, 'team_obp')} | SLG: {_stat(stats, 'team_slg')}",
                    f"- Win %: {_stat(stats, 'winPercent', fmt=lambda v: f'{v:.1%}')}",
                    f"- Runs/game (for): {_stat(stats, 'avgPointsFor', fmt=_dec2)}",
                    f"- Runs/game (against): {_stat(stats, 'avgPointsAgainst', fmt=_dec2)}",
                    f"- Run differential/game: {_stat(stats, 'differential', fmt=_dec2)}",
                    f"- Home: {_stat(stats, 'homeWins', fmt=_int)}-{_stat(stats, 'homeLosses', fmt=_int)}",
                    f"- Road: {_stat(stats, 'roadWins', fmt=_int)}-{_stat(stats, 'roadLosses', fmt=_int)}",
                    f"- Streak: {streak_str}",
                ])
            if sport == "icehockey_nhl":
                _dec2 = lambda v: f"{v:.2f}"
                _pct1 = lambda v: f"{v:.1f}%"
                gname = stats.get("goalie_name")
                if gname:
                    sv  = _stat(stats, "goalie_sv_pct", fmt=lambda v: f"{v:.3f}")
                    gaa = _stat(stats, "goalie_gaa",    fmt=_dec2)
                    goalie_line = f"{gname} — SV% {sv} | GAA {gaa}"
                else:
                    goalie_line = "N/A"
                return "\n".join([
                    f"- Points: {_stat(stats, 'points')}",
                    f"- Goals/game (for): {_stat(stats, 'pointsFor', fmt=_dec2)}",
                    f"- Goals/game (against): {_stat(stats, 'pointsAgainst', fmt=_dec2)}",
                    f"- Goal differential: {_stat(stats, 'differential', 'pointDifferential')}",
                    f"- Power play: {_stat(stats, 'pp_pct', fmt=_pct1)} | Penalty kill: {_stat(stats, 'pk_pct', fmt=_pct1)}",
                    f"- Shots/game: {_stat(stats, 'shots_for_pg', fmt=_dec2)} (for) / {_stat(stats, 'shots_against_pg', fmt=_dec2)} (against)",
                    f"- Top goalie: {goalie_line}",
                    f"- Rest days: {_stat(stats, 'rest_days')}",
                    f"- Last 10: {_stat(stats, 'Last Ten Games')}",
                    f"- Streak: {_stat(stats, 'streak')}",
                ])
            if sport == "basketball_wnba":
                _dec2 = lambda v: f"{v:.2f}"
                _pct1 = lambda v: f"{v:.1f}%"
                return "\n".join([
                    f"- Win %: {_stat(stats, 'winPercent', fmt=lambda v: f'{v:.1%}')}",
                    f"- Points/game (for): {_stat(stats, 'avgPointsFor', 'pointsFor', fmt=lambda v: f'{v:.1f}')}",
                    f"- Points/game (against): {_stat(stats, 'avgPointsAgainst', 'pointsAgainst', fmt=lambda v: f'{v:.1f}')}",
                    f"- Point differential: {_stat(stats, 'differential', 'pointDifferential', fmt=_dec2)}",
                    f"- Shooting: FG {_stat(stats, 'fg_pct', fmt=_pct1)} | 3PT {_stat(stats, 'three_pct', fmt=_pct1)}",
                    f"- Ball control: A/TO ratio {_stat(stats, 'ast_to_ratio', fmt=_dec2)} | Turnovers/game {_stat(stats, 'avg_turnovers', fmt=_dec2)}",
                    f"- Rebounds/game: {_stat(stats, 'avg_rebounds', fmt=_dec2)} (off: {_stat(stats, 'avg_off_rebounds', fmt=_dec2)})",
                    f"- Defense: Steals/game {_stat(stats, 'avg_steals', fmt=_dec2)} | Blocks/game {_stat(stats, 'avg_blocks', fmt=_dec2)}",
                    f"- Streak: {_stat(stats, 'streak')}",
                ])
            if sport == "americanfootball_nfl":
                _dec1 = lambda v: f"{v:.1f}"
                _dec2 = lambda v: f"{v:.2f}"
                _int  = lambda v: str(int(v))
                _sgn  = lambda v: f"+{int(v)}" if v >= 0 else str(int(v))
                return "\n".join([
                    f"- Win %: {_stat(stats, 'winPercent', fmt=lambda v: f'{v:.1%}')}",
                    f"- Points/game: {_stat(stats, 'pointsFor', fmt=_dec1)} (for) / {_stat(stats, 'pointsAgainst', fmt=_dec1)} (against)",
                    f"- Yards/game: passing {_stat(stats, 'pass_yards_pg', fmt=_dec1)} | rushing {_stat(stats, 'rush_yards_pg', fmt=_dec1)}",
                    f"- Turnover diff: {_stat(stats, 'to_differential', fmt=_sgn)} (giveaways {_stat(stats, 'giveaways', fmt=_int)} / takeaways {_stat(stats, 'takeaways', fmt=_int)})",
                    f"- Sacks allowed: {_stat(stats, 'sacks_allowed', fmt=_int)} | Defensive sacks: {_stat(stats, 'def_sacks', fmt=_int)}",
                    f"- Offensive EPA/play: {_stat(stats, 'off_epa_per_play', fmt=_dec2)} | Defensive EPA allowed/play: {_stat(stats, 'def_epa_allowed_per_play', fmt=_dec2)}",
                    f"- Rest days: {_stat(stats, 'rest_days')}",
                    f"- Streak: {_stat(stats, 'streak')}",
                ])
            if sport == "soccer_fifa_world_cup":
                form = stats.get("form", "N/A")
                w    = int(stats.get("wins", 0) or 0)
                d    = int(stats.get("ties", 0) or 0)
                l    = int(stats.get("losses", 0) or 0)
                pts  = int(stats.get("points", 0) or 0)
                gf   = int(stats.get("goals_for", 0) or 0)
                ga   = int(stats.get("goals_against", 0) or 0)
                gd   = int(stats.get("goal_diff", 0) or 0)
                gd_str   = f"+{gd}" if gd >= 0 else str(gd)
                return "\n".join([
                    f"- Form (last 5 intl): {form}",
                    f"- Tournament record: {w}W-{d}D-{l}L ({pts} pts)",
                    f"- Goals: {gf} for, {ga} against (GD: {gd_str})",
                ])
            return "- Stats: Not available"

        home_record = build_record(home_stats)
        away_record = build_record(away_stats)

        series_block = (
            f"\nPLAYOFF SERIES CONTEXT:\n- Current series standing: {series_context}\n"
            if series_context else ""
        )

        if weather:
            if weather.get("is_dome"):
                weather_block = "\nWEATHER: Indoor venue — weather has no effect on play.\n"
            else:
                weather_block = (
                    f"\nWEATHER (at game time, outdoor venue):\n"
                    f"- Temperature: {weather.get('temp', 'N/A')}°F\n"
                    f"- Wind: {weather.get('wind_speed', 'N/A')} mph\n"
                    f"- Conditions: {weather.get('description', 'N/A')}\n"
                    f"- Precip chance: {weather.get('precip_pct', 'N/A')}%\n"
                )
        else:
            weather_block = ""

        missing = []
        if not home_stats:
            missing.append(home)
        if not away_stats:
            missing.append(away)
        missing_note = (
            f"\nDATA WARNING: Current-season statistics are unavailable for: {', '.join(missing)}. "
            "For any team listed here, rely ONLY on the odds and roster/injury data above — "
            "do NOT fill gaps using knowledge of past rosters, historical performance, or "
            "reputation from your training data. Treat missing-data teams as unknown strength.\n"
        ) if missing else ""

        persona = _ANALYST_PERSONAS.get(sport, _DEFAULT_PERSONA)

        venue_line = f"\nVenue: {game['venue']}" if game.get("venue") else ""

        return f"""You are {persona['role']}. Analyze this upcoming game and provide a win probability estimate.

ANALYST FRAMEWORK FOR THIS SPORT:
{persona['framework']}

GAME: {away} @ {home}
Sport: {sport}
Commence: {game.get('commence_time', 'Unknown')}{venue_line}

ODDS:
- {home} moneyline: {game.get('home_ml', 'N/A')} (implied: {game.get('home_implied', 0):.1%})
- {away} moneyline: {game.get('away_ml', 'N/A')} (implied: {game.get('away_implied', 0):.1%})
- Total line: {game.get('total_line', 'N/A')}
{series_block}{weather_block}
HOME TEAM ({home}):
- Record: {home_record}
{build_stats_block(home_stats, "home")}
- Current roster: {home_roster or 'Not available'}
- Injuries (Out/Doubtful/Questionable):
{fmt_injuries(home_injuries)}

AWAY TEAM ({away}):
- Record: {away_record}
{build_stats_block(away_stats, "away")}
- Current roster: {away_roster or 'Not available'}
- Injuries (Out/Doubtful/Questionable):
{fmt_injuries(away_injuries)}

STATISTICAL MODEL ESTIMATE: {home} win probability = {base_prob:.1%}
{missing_note}
Apply your framework above to this data and estimate win probability. Weigh all relevant factors together — no single factor should dominate unless the data is overwhelmingly one-sided. In your reasoning, explain your logic naturally; do NOT cite rule numbers or use phrases like "per the framework" or "Framework #1".

CRITICAL PLAYER DATA RULE: Your reasoning may ONLY name specific players who appear in the roster or injury list provided above. Do NOT name any player from your training data who is not listed — rosters change constantly and your training data is stale. If a player you recall is not in the data above, they may have been traded, cut, or retired. Mentioning a player not in the provided data is a factual error. Base all player-specific claims strictly on the lists above.

Respond ONLY with valid JSON in this exact format:
{{
  "adjusted_home_prob": <float between 0 and 1>,
  "confidence": "<low|medium|high>",
  "reasoning": "<2-3 sentences explaining your adjustment>",
  "bet_recommendation": "<home_ml|away_ml|over|under|pass>"
}}"""

    def _parse_response(self, raw: str, fallback_prob: float) -> dict:
        # First attempt: standard JSON parse
        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            data  = json.loads(raw[start:end])
            prob  = float(data.get("adjusted_home_prob", fallback_prob))
            return self._build_result(prob, data, fallback_prob)
        except Exception:
            pass

        # Second attempt: regex extraction — handles stray quotes / minor malformed JSON
        try:
            prob_match = re.search(r'"adjusted_home_prob"\s*:\s*([0-9.]+)', raw)
            conf_match = re.search(r'"confidence"\s*:\s*"(low|medium|high)"', raw)
            rec_match  = re.search(r'"bet_recommendation"\s*:\s*"([^"]+)"', raw)
            # Reasoning: grab everything between the first " after the key and the last "
            # before the next key or closing brace — tolerates embedded stray quotes
            reason_match = re.search(r'"reasoning"\s*:\s*"(.*?)"(?:\s*,\s*"|\s*\})', raw, re.DOTALL)

            prob = float(prob_match.group(1)) if prob_match else fallback_prob
            data = {
                "adjusted_home_prob": prob,
                "confidence":         conf_match.group(1) if conf_match else "low",
                "reasoning":          reason_match.group(1).strip() if reason_match else "",
                "bet_recommendation": rec_match.group(1)  if rec_match  else "pass",
            }
            logger.warning(f"Used regex fallback to parse Claude response")
            return self._build_result(prob, data, fallback_prob)
        except Exception as e:
            logger.warning(f"Failed to parse Claude response: {e}\nRaw: {raw}")
            return {
                "adjusted_home_prob":  fallback_prob,
                "confidence":          "low",
                "reasoning":           "Parse error — using model probability",
                "bet_recommendation":  "pass",
            }

    @staticmethod
    def _build_result(prob: float, data: dict, fallback_prob: float) -> dict:
        prob = max(0.05, min(0.95, float(prob)))
        return {
            "adjusted_home_prob":  prob,
            "confidence":          data.get("confidence", "low"),
            "reasoning":           data.get("reasoning", ""),
            "bet_recommendation":  data.get("bet_recommendation", "pass"),
        }
