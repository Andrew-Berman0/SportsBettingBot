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

        return f"""You are an expert sports betting analyst. Analyze this upcoming game and provide a win probability estimate.

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
Based on this data, provide your analysis. Consider: home field advantage, rest/fatigue, recent momentum, injury impact, matchup style, series momentum and elimination pressure (if applicable), and any other relevant factors you know about these teams.

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
