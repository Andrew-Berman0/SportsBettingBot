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
        "role": "an expert MLB betting analyst who treats each game as a starting-pitcher matchup first and a team contest second",
        "framework": (
            "1. STARTING PITCHER ERA is the single most important factor — a 2.50 ERA vs a 5.00 ERA starter can shift the true line by 15-20% on its own.\n"
            "2. Team win percentage is noisy over 162 games; run differential is a better signal of true team quality.\n"
            "3. Home/road splits matter — factor in park effects and travel.\n"
            "4. Do NOT overweight hot or cold streaks — regression to the mean is strong in baseball.\n"
            "5. MLB books are highly efficient; be skeptical of any edge above 4-5% and lean toward 'pass' when uncertain.\n"
            "6. Underdogs with an elite starter facing an average starter are the most consistently undervalued spot in baseball."
        ),
    },
    "icehockey_nhl": {
        "role": "an expert NHL betting analyst who evaluates goaltender matchups and special teams as the primary drivers of game outcomes",
        "framework": (
            "1. GOALTENDER performance is paramount — save percentage and goals-against average outweigh team record.\n"
            "2. Special teams (power play %, penalty kill %) create consistent, repeatable edges.\n"
            "3. In playoffs, home ice is strong (~58% home win rate); series pressure and elimination urgency are real.\n"
            "4. Playoff hockey is defensive — fade high-scoring teams when they face disciplined, defensive opponents.\n"
            "5. Short rest (back-to-back playoff games) significantly impacts goaltender performance."
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
    "americanfootball_nfl": {
        "role": "an expert NFL betting analyst who treats quarterback play and offensive line health as the foundation of every game evaluation",
        "framework": (
            "1. Starting QB quality is the single biggest factor — a backup QB shifts win probability by 10-15%.\n"
            "2. Offensive line health determines both run efficiency and pass protection; it amplifies or limits the QB.\n"
            "3. Weather is a major factor in outdoor stadiums — wind above 15 mph and rain suppress passing and scoring.\n"
            "4. Home field is worth ~3 points; stronger in cold-weather outdoor venues.\n"
            "5. The NFL market is the most efficient of all major sports — be very conservative about claiming edges and default to 'pass' when the data is thin."
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
                                    starting_pitchers or {})

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
                      starting_pitchers: dict | None = None) -> str:
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
            w = stats.get("W") or stats.get("wins")
            l = stats.get("L") or stats.get("losses")
            otl = stats.get("otLosses") or stats.get("OTL")
            if w is None and l is None:
                return "?-?"
            record = f"{int(w) if w is not None else '?'}-{int(l) if l is not None else '?'}"
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
                return "\n".join([
                    pitcher_line,
                    f"- Win %: {_stat(stats, 'winPercent', fmt=lambda v: f'{v:.1%}')}",
                    f"- Runs/game (for): {_stat(stats, 'pointsFor', fmt=lambda v: f'{v:.2f}')}",
                    f"- Runs/game (against): {_stat(stats, 'pointsAgainst', fmt=lambda v: f'{v:.2f}')}",
                    f"- Run differential: {_stat(stats, 'differential', 'pointDifferential')}",
                    f"- Home: {_stat(stats, 'homeWins')}-{_stat(stats, 'homeLosses')}",
                    f"- Road: {_stat(stats, 'roadWins')}-{_stat(stats, 'roadLosses')}",
                    f"- Last 10: {_stat(stats, 'Last Ten Games')}",
                    f"- Streak: {_stat(stats, 'streak')}",
                ])
            if sport == "icehockey_nhl":
                return "\n".join([
                    f"- Points: {_stat(stats, 'points')}",
                    f"- Goals/game (for): {_stat(stats, 'pointsFor', fmt=lambda v: f'{v:.2f}')}",
                    f"- Goals/game (against): {_stat(stats, 'pointsAgainst', fmt=lambda v: f'{v:.2f}')}",
                    f"- Goal differential: {_stat(stats, 'differential', 'pointDifferential')}",
                    f"- Last 10: {_stat(stats, 'Last Ten Games')}",
                    f"- Streak: {_stat(stats, 'streak')}",
                ])
            if sport == "basketball_wnba":
                return "\n".join([
                    f"- Win %: {_stat(stats, 'winPercent', fmt=lambda v: f'{v:.1%}')}",
                    f"- Points/game (for): {_stat(stats, 'avgPointsFor', 'pointsFor', fmt=lambda v: f'{v:.1f}')}",
                    f"- Points/game (against): {_stat(stats, 'avgPointsAgainst', 'pointsAgainst', fmt=lambda v: f'{v:.1f}')}",
                    f"- Point differential: {_stat(stats, 'differential', 'pointDifferential')}",
                    f"- Home: {_stat(stats, 'Home')}",
                    f"- Road: {_stat(stats, 'Road')}",
                    f"- Last 10: {_stat(stats, 'Last Ten Games')}",
                    f"- Streak: {_stat(stats, 'streak')}",
                ])
            if sport == "americanfootball_nfl":
                return "\n".join([
                    f"- Win %: {_stat(stats, 'winPercent', fmt=lambda v: f'{v:.1%}')}",
                    f"- Points/game (for): {_stat(stats, 'pointsFor', fmt=lambda v: f'{v:.1f}')}",
                    f"- Points/game (against): {_stat(stats, 'pointsAgainst', fmt=lambda v: f'{v:.1f}')}",
                    f"- Point differential: {_stat(stats, 'differential', 'pointDifferential')}",
                    f"- Streak: {_stat(stats, 'streak')}",
                ])
            return "- Stats: Not available"

        home_record = build_record(home_stats)
        away_record = build_record(away_stats)

        series_block = (
            f"\nPLAYOFF SERIES CONTEXT:\n- Current series standing: {series_context}\n"
            if series_context else ""
        )

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

        return f"""You are {persona['role']}. Analyze this upcoming game and provide a win probability estimate.

ANALYST FRAMEWORK FOR THIS SPORT:
{persona['framework']}

GAME: {away} @ {home}
Sport: {sport}
Commence: {game.get('commence_time', 'Unknown')}

ODDS:
- {home} moneyline: {game.get('home_ml', 'N/A')} (implied: {game.get('home_implied', 0):.1%})
- {away} moneyline: {game.get('away_ml', 'N/A')} (implied: {game.get('away_implied', 0):.1%})
- Total line: {game.get('total_line', 'N/A')}
{series_block}
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
Apply your framework above to this data and estimate win probability. Follow the priority order in the framework — don't treat all factors equally.

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
