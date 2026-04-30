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
        prompt = self._build_prompt(game, home_stats, away_stats, base_home_prob)

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

    def _build_prompt(self, game: dict, home_stats: dict, away_stats: dict, base_prob: float) -> str:
        home = game["home_team"]
        away = game["away_team"]

        home_record = f"{home_stats.get('W', '?')}-{home_stats.get('L', '?')}"
        away_record = f"{away_stats.get('W', '?')}-{away_stats.get('L', '?')}"

        return f"""You are an expert sports betting analyst. Analyze this upcoming game and provide a win probability estimate.

GAME: {away} @ {home}
Commence: {game.get('commence_time', 'Unknown')}

ODDS:
- {home} moneyline: {game.get('home_ml', 'N/A')} (implied: {game.get('home_implied', 0):.1%})
- {away} moneyline: {game.get('away_ml', 'N/A')} (implied: {game.get('away_implied', 0):.1%})
- Total line: {game.get('total_line', 'N/A')}

HOME TEAM ({home}):
- Record: {home_record}
- Net Rating: {home_stats.get('NET_RATING', 'N/A')}
- Off Rating: {home_stats.get('OFF_RATING', 'N/A')}
- Def Rating: {home_stats.get('DEF_RATING', 'N/A')}
- Pace: {home_stats.get('PACE', 'N/A')}
- Last 10 games: {home_stats.get('win_pct_l10', 'N/A'):.0%} win rate
- Back-to-back: {'Yes' if home_stats.get('is_back_to_back') else 'No'}
- Rest days: {home_stats.get('rest_days', 'N/A')}

AWAY TEAM ({away}):
- Record: {away_record}
- Net Rating: {away_stats.get('NET_RATING', 'N/A')}
- Off Rating: {away_stats.get('OFF_RATING', 'N/A')}
- Def Rating: {away_stats.get('DEF_RATING', 'N/A')}
- Pace: {away_stats.get('PACE', 'N/A')}
- Last 10 games: {away_stats.get('win_pct_l10', 'N/A'):.0%} win rate
- Back-to-back: {'Yes' if away_stats.get('is_back_to_back') else 'No'}
- Rest days: {away_stats.get('rest_days', 'N/A')}

STATISTICAL MODEL ESTIMATE: {home} win probability = {base_prob:.1%}

Based on this data, provide your analysis. Consider: home court advantage, rest/fatigue, recent momentum, matchup style, and any other relevant factors you know about these teams.

Respond ONLY with valid JSON in this exact format:
{{
  "adjusted_home_prob": <float between 0 and 1>,
  "confidence": "<low|medium|high>",
  "reasoning": "<2-3 sentences explaining your adjustment>",
  "bet_recommendation": "<home_ml|away_ml|over|under|pass>"
}}"""

    def _parse_response(self, raw: str, fallback_prob: float) -> dict:
        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            data  = json.loads(raw[start:end])
            prob  = float(data.get("adjusted_home_prob", fallback_prob))
            prob  = max(0.05, min(0.95, prob))   # clamp to [5%, 95%]
            return {
                "adjusted_home_prob":  prob,
                "confidence":          data.get("confidence", "low"),
                "reasoning":           data.get("reasoning", ""),
                "bet_recommendation":  data.get("bet_recommendation", "pass"),
            }
        except Exception as e:
            logger.warning(f"Failed to parse Claude response: {e}\nRaw: {raw}")
            return {
                "adjusted_home_prob":  fallback_prob,
                "confidence":          "low",
                "reasoning":           "Parse error",
                "bet_recommendation":  "pass",
            }
