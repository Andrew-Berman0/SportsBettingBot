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

# Per-analyst version. Each sport's persona is effectively its own analyst, so
# versions are tracked PER SPORT — bump ONLY the sport whose persona/framework/edge
# logic you changed, and add a changelog line. Stamped on each bet/pass at
# evaluation time (via analyst_version_for) and carried into game_outcomes.jsonl,
# so every sport's calibration history segments by its own logic version.
# A change that affects everyone (e.g. swapping the model) means bumping every entry.
#
# Changelog:
#   ALL SPORTS +1 (2026-07-08): reason-to-the-number output fix (nba 2->3, wnba 4->5,
#       mlb 3->4, nfl 1->2, nhl 1->2, wc 1->2). The JSON output emitted adjusted_home_prob
#       FIRST and reasoning LAST, so the model committed to a probability before writing
#       any reasoning — the number was an un-reasoned gut estimate and the "magnitude
#       discipline" prose was written afterward to rationalize it (observed: prose said
#       "modest, a few points" while the model had moved the line 12pt). Reordered the
#       schema so reasoning comes FIRST and adjusted_home_prob LAST, and required the
#       reasoning to state the market implied %, the settled %, and the point gap — so the
#       number is the conclusion of the reasoning, not the premise. Global change, so all
#       active sports bumped to keep calibration samples coherent. Paired with a hard
#       max_divergence backstop in config (MLB/WNBA).
#   baseball_mlb=1 (2026-06-13): first versioned MLB persona — season team-quality
#       stats treated as already priced, no home-team fades on aggregates, concrete
#       game-specific edge required; MLB min_edge 3%->5%.
#   baseball_mlb=2 (2026-06-14): two-part fix for ERA over-weighting that produced
#       implausible ~19pt single-game divergences.
#       (a) magnitude discipline — v1 required a concrete edge but didn't cap how far
#           it could move the number; now caps deviation at ~8-10 points absent a
#           roster-level mismatch.
#       (b) starter sample/peripherals — the pitcher line now includes IP/GS and
#           WHIP/K9/K-BB (from ESPN core) so the small-sample-ERA rule is actionable:
#           Claude can tell whether an ERA is over 80 IP or 2 starts, and whether the
#           peripherals support it. (Folded into v2; both shipped before any v2 game
#           logged, so the sample stays coherent.)
#   basketball_nba=2 (2026-06-19): player-level data. Like WNBA v3, the NBA persona
#       leaned on "player availability is decisive" but was fed only team ratings +
#       injury NAMES. The risk: a key ROOKIE/recent call-up is both missing from the
#       data AND from Claude's training memory, so an injury to (or the presence of)
#       that player can't be evaluated. Added NBAStatsFetcher.get_player_stats (top
#       scorers' PPG/RPG/APG/min from ESPN core — independent of nba_api) surfaced as
#       "Key players (season avg)"; rule 3 now gauges absences by production/minutes.
#       key_players is a critical data-gap field.
#   baseball_mlb=3 (2026-06-19): away-side skepticism. Rec-accuracy audit across all
#       evaluated games showed a strong asymmetry — Claude's HOME leans hit 63% (MLB v2
#       72%) but its AWAY leans hit only 46% overall (MLB post-legacy 3/18 ≈ 17%, worse
#       than a coin flip). The v2 home-fade fix worked, but Claude still over-rates the
#       road team / underdog and invents away value that doesn't materialize. New rule 10
#       applies a higher bar to AWAY leans: favor the road side only on an unambiguous,
#       game-specific edge (ace vs call-up, confirmed home-side absence), never on
#       aggregates / form / "live dog" feel. Home logic unchanged (it's working).
#   basketball_wnba=4 (2026-06-20): away-skepticism + magnitude discipline (ports what
#       MLB v2/v3 already have). A live v3 bet (Storm +240 over a -300 Mercury) fading a
#       favorite ~13pt on PUBLIC injuries the line already priced exposed three gaps the
#       rec-audit predicted: WNBA away leans hit ~40%, no magnitude cap, no max_edge.
#       Added rule 2 (visible injuries are already priced — only late-breaking is an
#       edge), rule 6 (away-side skepticism), rule 7 (cap divergence ~8-10pt). Paired
#       with a baseball-style max_edge=0.12 backstop in config. Home/player logic from
#       v3 unchanged.
#   basketball_wnba=3 (2026-06-19): player-level data. The persona was built around
#       "individual player impact" (rule 1) but was only fed team aggregates + injury
#       NAMES — no per-player production, so it couldn't gauge how much an absence
#       mattered and fell back on stale name-recognition. Added a per-player fetcher
#       (WNBAStatsFetcher.get_player_stats → top scorers' PPG/RPG/APG/min from ESPN
#       core) surfaced as "Key players (season avg)" in the prompt; rule 1 now tells
#       Claude to gauge injuries by actual production/minutes. key_players is a critical
#       data-gap field (alerts if it comes back empty).
#   basketball_wnba=2 (2026-06-17): same home-fade pattern as MLB v0 surfaced in the
#       data (Claude rated home ~7pt below market; away-leans 1/5). Reframed rule 2
#       (home is real and priced — don't fade it on aggregates) and rule 5 (softer
#       market still prices quality/home; require a concrete edge, keep divergences
#       modest). Small sample (~10 games), so a lighter touch than the MLB overhaul.
#   mma_ufc=2 (2026-06-17): v1 was so pass-biased it bet nothing in its first weeks —
#       rules 6 & 8 said "default to pass" and "thin data = pass", which vetoed even
#       clean edges (e.g. a +7pt edge on a +174 live dog still returned pass because
#       the fighter was thin-data). Reoriented: pass is reserved for genuinely close/
#       contradictory reads and true debutants (1-2 bouts), not ordinary uncertainty;
#       a sizable, well-grounded edge is now bettable, and live underdogs the market
#       underrates are flagged as where MMA value concentrates (new rule 9). Heavy-
#       favorite skepticism and the -300 min_odds backstop are unchanged. This is a
#       deliberate learning experiment to generate settled UFC results — watch ROI.
#   all others=1 (2026-06-13): first versioned state of each persona (unchanged at
#       versioning introduction). Model: Sonnet 4.6 across all. Outcomes logged
#       before this carry no analyst_version.
#
ANALYST_VERSIONS: dict[str, int] = {
    "basketball_nba":        3,
    "basketball_wnba":       5,
    "americanfootball_nfl":  2,
    "baseball_mlb":          4,
    "icehockey_nhl":         2,
    "soccer_fifa_world_cup": 2,
    "mma_ufc":               2,   # inactive
}


def analyst_version_for(sport: str) -> int:
    """Version of the analyst (persona) for a sport — stamped on its bets/passes."""
    return ANALYST_VERSIONS.get(sport, 1)

# Sport-specific analyst personas injected at the top of every prompt.
# Role shapes how Claude frames the problem; framework sets the priority order
# for what actually drives outcomes in that sport.
_ANALYST_PERSONAS: dict[str, dict[str, str]] = {
    "basketball_nba": {
        "role": "an expert NBA betting analyst who specializes in advanced team efficiency metrics",
        "framework": (
            "1. NET RATING is your primary lens — offensive and defensive efficiency matter more than raw record.\n"
            "2. Weight the last 10 games heavily; momentum and form are real in the NBA. Last-10 and last-5 average point differential are sharper form signals than win rate alone — a team winning by shrinking margins is cooling off.\n"
            "3. Player availability is decisive — NBA outcomes swing hard on who is in or out. Cross-reference the injury list against the KEY PLAYERS line (each team's top players with season PPG/RPG/APG/minutes): an injured 25-PPG, 36-min starter can shift win probability 8-15%, while a missing end-of-bench player barely matters. Gauge an absence by that player's actual production and minutes — NOT by name recognition. This matters most for rookies and recent call-ups you may not otherwise know: trust the KEY PLAYERS stats over any memory. Do not assume a player listed out will play.\n"
            "4. Rest and back-to-backs are meaningful — a back-to-back team loses ~2-3% win probability.\n"
            "5. Home court is worth ~2-3% in the regular season (home teams win ~55-58%); in the playoffs, home court and series pressure dominate, and teams facing elimination often overperform.\n"
            "6. Caution: in playoffs, current form matters more than regular-season record."
        ),
    },
    "baseball_mlb": {
        "role": "an expert MLB betting analyst who weighs starting pitching, team quality, bullpen depth, and park context together",
        "framework": (
            "1. Starting pitcher quality is your main GAME-SPECIFIC input — weigh season ERA and the team's overall pitching numbers. "
            "The pitcher's throwing hand is given (LHP/RHP); a platoon mismatch can matter, but you lack the opposing lineup's splits vs left/right, so treat handedness as a light factor. "
            "Starters typically pitch 5-6 innings, so a bullpen advantage often decides close games. A strong starter on a weak team is not an automatic edge.\n"
            "2. CRITICAL: the season team-quality stats shown (run differential, OPS, team ERA, win%) are aggregates the market has ALREADY priced into the line. "
            "'This team is better by run differential/OPS' is NOT an edge — it is information the book already has. Use these stats only to sanity-check, never as the primary reason to take a side or fade a price.\n"
            "3. Bullpen ERA matters — a poor bullpen erases a starter's advantage from the 6th inning onward. Only season bullpen ERA is provided, not recent usage, so do not speculate about fatigue you can't see.\n"
            "4. Do NOT fade the home team just because the road team has better aggregate season stats. The market prices home field and team quality efficiently — a road team that looks better on paper is usually already reflected in the line. Treat Home/Road records as context, not a reason to override the market.\n"
            "5. Streaks and momentum regress hard in baseball. Be skeptical of hot/cold narratives.\n"
            "6. To take a side you need a CONCRETE, game-specific reason the market line is wrong: a clear starting-pitcher mismatch (over a believable sample, not 1-2 starts), a confirmed key injury or lineup absence, or a distinct bullpen edge in a likely-close game. Without a specific reason like that, defer to the market and pass.\n"
            "7. MAGNITUDE DISCIPLINE — this is critical. MLB single-game win probabilities are compressed: even the best team vs the worst rarely prices beyond ~65/35, and an ERA edge between two real MLB starters is worth only a few points. Even a clear starter AND bullpen edge rarely justifies moving the game's win probability more than ~8-10 points away from the market's implied number. If your estimate diverges from the market by more than ~10 points, that is a red flag that you are overweighting ERA — pull back toward the market unless there is a roster-level mismatch (a genuine ace facing a replacement-level call-up). The market already prices the starters and bullpens; your edge is at the margin, not a wholesale repricing.\n"
            "8. Judge the starter's ERA by its sample and peripherals (both are now provided). Use IP/GS for sample size — an ERA over 80+ IP is meaningful, but a shiny or ugly ERA under ~20 IP (1-3 starts) is mostly noise. Cross-check ERA against WHIP, K/9, and K/BB: if the ERA is far better than the peripherals suggest, it is likely lucky and will regress — trust the peripherals over a small-sample ERA when they conflict.\n"
            "9. Default to 'pass' when the edge is below 5%, when factors conflict, or when your lean rests mainly on season team-quality aggregates rather than a specific game-level edge.\n"
            "10. AWAY-SIDE SKEPTICISM — calibration shows your road/underdog 'value' picks have been markedly less reliable than your home picks; you tend to over-rate the road team and find away value that doesn't materialize. The market prices road teams and underdogs accurately. Apply a HIGHER bar to any AWAY lean: only favor the road side over the market when the concrete, game-specific edge from rule 6 is unambiguous and clearly points to the away team (e.g. a genuine ace vs a replacement-level call-up, or a confirmed key absence on the home side) — never on aggregate stats, recent form, or a general sense the dog is live. If an away lean rests on anything softer than that, pass."
        ),
    },
    "icehockey_nhl": {
        "role": "an expert NHL betting analyst who evaluates goaltender matchups and special teams as the primary drivers of game outcomes",
        "framework": (
            "1. GOALTENDER performance is paramount — save percentage and goals-against average outweigh team record. Note the goalie shown is the team's season primary, NOT a confirmed starter for tonight; if a backup starts, the real matchup differs, so do not over-anchor on these numbers when the edge is thin.\n"
            "2. Special teams (power play %, penalty kill %) create consistent, repeatable edges.\n"
            "3. Home ice is worth ~2-3% in the regular season; in the playoffs it is stronger (~58% home win rate) and series pressure and elimination urgency are real.\n"
            "4. Playoff hockey is defensive — fade high-scoring teams when they face disciplined, defensive opponents.\n"
            "5. Short rest significantly impacts goaltender performance — treat rest_days ≤ 1 as a back-to-back and discount that team's win probability by 3-5%."
        ),
    },
    "basketball_wnba": {
        "role": "an expert WNBA betting analyst who weights individual player impact heavily due to the league's smaller rosters",
        "framework": (
            "1. With only 11-12 active players, one star's absence can shift win probability by 10-15%. You are given each team's KEY PLAYERS with season per-game production (PPG/RPG/APG/minutes) — cross-reference the injury list against it: an injured 18-PPG, 32-min starter is a major blow, a missing deep-bench player is nearly irrelevant. Gauge an absence by the player's actual production and role (minutes), NOT by name recognition or memory. Two healthy rosters of comparable top-end production is not an edge — it's already in the line.\n"
            "2. CRITICAL — injuries you can see are ALREADY PRICED. The Out/Doubtful list you are given is public; the book set the line knowing it. A star being out is NOT an edge just because you can name the points it removes — the price already reflects it. Listed injuries are only an edge if they are genuinely LATE-BREAKING (a scratch the market hasn't moved on yet). When a heavy favorite is already heavy DESPITE known injuries on both sides, that is the market telling you it has done this math — do not re-derive the absences and conclude the line is wrong.\n"
            "3. Home court is real in the WNBA (~53-55% home win rate) and the market already prices it — do NOT rate the home team below the market on aggregate stats alone, or fade the home side just because the road team looks better on paper.\n"
            "4. Point differential is the best efficiency signal — win percentage is volatile on a short schedule.\n"
            "5. Rest matters in the WNBA's compressed schedule — use the Rest days value; a team on 0-1 days rest (a back-to-back or close to it) is at a real disadvantage worth a few points of win probability.\n"
            "6. AWAY-SIDE SKEPTICISM — calibration shows your road/underdog 'value' picks are markedly less reliable than your home picks; you tend to over-rate the road team and the live dog. The market prices road teams and underdogs accurately. Apply a HIGHER bar to any AWAY lean: only favor the road side over the market on an unambiguous, late-breaking, game-specific edge — never on aggregate stats, season form, or already-public injuries. If an away lean rests on anything softer than that, pass.\n"
            "7. MAGNITUDE DISCIPLINE — WNBA single-game probabilities are compressed and the market is efficient on public information. Even a clear, concrete edge rarely justifies moving the game's win probability more than ~8-10 points from the market's implied number. If your estimate diverges from the market by more than ~10 points, that is a red flag you are overweighting something already priced (usually known injuries) — pull back toward the market unless you have genuine late-breaking news. A large divergence is far more often a model error than real value.\n"
            "8. WNBA markets are softer than the NBA/MLB, but they still price team quality and home court — your edge comes from a CONCRETE, game-specific reason the line is wrong (late-breaking injury/roster news, a clear rest or travel disadvantage), not from a general sense the market is beatable."
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
            "1. Starting QB quality is the single biggest factor — if the injury list shows the starting QB out or doubtful, a backup starting shifts win probability by 10-15%. No QB rating is provided, so base QB assessments on the injury and roster data rather than assumptions.\n"
            "2. Offensive line health: use sacks allowed per game as a proxy — teams allowing 3+ sacks/game have degraded pass protection.\n"
            "3. Offensive EPA/play and Defensive EPA allowed/play are the most reliable efficiency metrics; weight them above win% and points/game.\n"
            "4. Turnover differential is a strong game-level predictor — teams with +3 or better season differential win at ~65% rate.\n"
            "5. Weather (outdoor only): wind_speed > 15 mph suppresses passing by ~10%; temp < 32°F adds further suppression; adjust both teams equally unless one is a run-first offense.\n"
            "6. Short rest (rest_days ≤ 4, i.e. Thursday games) is worth -3 to -5% win probability for the short-rest team.\n"
            "7. Home field is worth ~3 points; stronger in cold outdoor venues (Lambeau, Arrowhead, Highmark).\n"
            "8. The NFL market is the most efficient of all major sports — default to 'pass' when edges are below 4% or data is thin."
        ),
    },
    "mma_ufc": {
        "role": "an expert UFC/MMA betting analyst who reads style matchups, physical edges, and finishing ability",
        "framework": (
            "1. STYLE MATCHUP is your primary lens — infer each fighter's style from the stats: high strikes-landed-per-minute with good accuracy signals a volume striker; high takedown average plus submission attempts signals a grappler. A grappler who can impose takedowns on a fighter who needs the fight standing is a real stylistic edge — and the reverse when a takedown-reliant fighter meets a pure striker who can stay upright.\n"
            "2. IMPORTANT: only OFFENSIVE stats are provided (strikes landed/min, striking accuracy, takedown average/accuracy, submission attempts). You are NOT given strikes absorbed, striking defense, or takedown defense — so do not state a fighter's defense or durability as fact; infer it cautiously and flag the uncertainty.\n"
            "3. Physical edges matter: a large reach advantage favors a rangy, accurate striker; a southpaw-vs-orthodox matchup can disrupt the orthodox fighter. Weigh reach and stance alongside style, not in isolation.\n"
            "4. Age and decline: MMA fighters fall off sharply after ~35, especially cardio and chin. Lean toward the younger fighter when ages diverge and the older one is past their mid-30s.\n"
            "5. Finishing tendency: a high KO/TKO% means real one-shot upset equity even as an underdog; a decision-heavy fighter is lower variance. Use this when weighing live dogs against favorites.\n"
            "6. MMA is extremely high-variance — anyone can be finished by a single strike. Be skeptical of heavy favorites priced -350 or worse. But do NOT reflexively pass every fight: when style, physicals, age, and finishing tendency line up behind one fighter and produce a clear edge over the market, that IS the bettable spot — back it. Reserve 'pass' for genuinely close or internally contradictory reads, not for the ordinary uncertainty that every fight carries.\n"
            "7. Records reflect quality but you are NOT given strength of schedule — do not over-read a gaudy record that may be built on weak opposition.\n"
            "8. Data quality gates conviction, not betting outright. A true debutant or a fighter with only 1-2 pro bouts is tiny-sample — pass there. But a fighter with a real bout history whose stats, physicals, and style produce a sizable, well-grounded edge is bettable even when you'd like more data; don't let ordinary thin-ness veto a clear edge.\n"
            "9. Live underdogs are where MMA value concentrates: a stylistic/physical/finishing edge on a +money fighter the market underrates is exactly the bet to make — do not pass it just because the other fighter is favored."
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
                "bet_recommendation": str,   # "home_ml" | "away_ml" | "pass"
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
                _sgn1 = lambda v: f"{v:+.1f}"
                kp = stats.get("key_players") or []
                players_str = " | ".join(
                    f"{p.get('name')} ({p.get('pos','')}) "
                    f"{p.get('ppg','?')}p/{p.get('rpg','?')}r/{p.get('apg','?')}a, {p.get('mpg','?')}min"
                    for p in kp
                ) or "N/A"
                return "\n".join([
                    f"- Net Rating: {_stat(stats, 'NET_RATING')}",
                    f"- Off Rating: {_stat(stats, 'OFF_RATING')}",
                    f"- Def Rating: {_stat(stats, 'DEF_RATING')}",
                    f"- Pace: {_stat(stats, 'PACE')}",
                    f"- Last 10: {_pct(stats.get('win_pct_l10'))} win rate | "
                    f"{_stat(stats, 'avg_diff_l10', fmt=_sgn1)} avg point diff",
                    f"- Last 5: {_pct(stats.get('win_pct_l5'))} win rate | "
                    f"{_stat(stats, 'avg_diff_l5', fmt=_sgn1)} avg point diff",
                    f"- Key players (season avg): {players_str}",
                    f"- Back-to-back: {'Yes' if stats.get('is_back_to_back') else 'No'}",
                    f"- Rest days: {stats.get('rest_days', 'N/A')}",
                ])
            if sport == "baseball_mlb":
                sp = starting_pitchers or {}
                pitcher = sp.get(side, {})
                throws = pitcher.get("throws")
                hand_str = f", {throws}HP" if throws in ("L", "R") else ""
                if pitcher.get("name") and pitcher["name"] != "TBD":
                    extras = []
                    if pitcher.get("ip"):
                        extras.append(f"{pitcher['ip']} IP" + (f"/{pitcher['gs']} GS" if pitcher.get("gs") else ""))
                    if pitcher.get("whip"):
                        extras.append(f"WHIP {pitcher['whip']}")
                    if pitcher.get("k9"):
                        extras.append(f"{pitcher['k9']} K/9")
                    if pitcher.get("kbb"):
                        extras.append(f"{pitcher['kbb']} K/BB")
                    extra_str = ("  [" + " | ".join(extras) + "]") if extras else ""
                    pitcher_line = (
                        f"- Starting pitcher: {pitcher['name']} "
                        f"({pitcher.get('wins','?')}-{pitcher.get('losses','?')}, "
                        f"{pitcher.get('era','?')} ERA{hand_str}){extra_str}"
                    )
                else:
                    pitcher_line = "- Starting pitcher: TBD"
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
                # ESPN NHL standings reports season-total goals (no per-game field);
                # divide by games played for a usable rate.
                gp = stats.get("gamesPlayed")
                def _per_game(total_key: str) -> str:
                    try:
                        return f"{float(stats.get(total_key)) / float(gp):.2f}"
                    except (TypeError, ValueError, ZeroDivisionError):
                        return "N/A"
                streak_raw = stats.get("streak")
                try:
                    sv_streak = int(float(streak_raw))
                    streak_str = f"W{sv_streak}" if sv_streak >= 0 else f"L{abs(sv_streak)}"
                except (TypeError, ValueError):
                    streak_str = "N/A"
                last10 = stats.get("Last Ten Games")
                last10 = last10.split(",")[0] if isinstance(last10, str) else "N/A"
                return "\n".join([
                    f"- Points: {_stat(stats, 'points', fmt=lambda v: str(int(v)))}",
                    f"- Goals/game (for): {_per_game('pointsFor')}",
                    f"- Goals/game (against): {_per_game('pointsAgainst')}",
                    f"- Goal differential/game: {_stat(stats, 'differential', fmt=lambda v: f'{v:+.2f}')}",
                    f"- Power play: {_stat(stats, 'pp_pct', fmt=_pct1)} | Penalty kill: {_stat(stats, 'pk_pct', fmt=_pct1)}",
                    f"- Shots/game: {_stat(stats, 'shots_for_pg', fmt=_dec2)} (for) / {_stat(stats, 'shots_against_pg', fmt=_dec2)} (against)",
                    f"- Top goalie (season primary, not a confirmed starter): {goalie_line}",
                    f"- Rest days: {_stat(stats, 'rest_days')}",
                    f"- Last 10: {last10}",
                    f"- Streak: {streak_str}",
                ])
            if sport == "basketball_wnba":
                _dec2 = lambda v: f"{v:.2f}"
                _pct1 = lambda v: f"{v:.1f}%"
                streak_raw = stats.get("streak")
                try:
                    sv = int(float(streak_raw))
                    streak_str = f"W{sv}" if sv >= 0 else f"L{abs(sv)}"
                except (TypeError, ValueError):
                    streak_str = "N/A"
                kp = stats.get("key_players") or []
                players_str = " | ".join(
                    f"{p.get('name')} ({p.get('pos','')}) "
                    f"{p.get('ppg','?')}p/{p.get('rpg','?')}r/{p.get('apg','?')}a, {p.get('mpg','?')}min"
                    for p in kp
                ) or "N/A"
                return "\n".join([
                    f"- Win %: {_stat(stats, 'winPercent', fmt=lambda v: f'{v:.1%}')}",
                    f"- Points/game (for): {_stat(stats, 'avgPointsFor', 'pointsFor', fmt=lambda v: f'{v:.1f}')}",
                    f"- Points/game (against): {_stat(stats, 'avgPointsAgainst', 'pointsAgainst', fmt=lambda v: f'{v:.1f}')}",
                    f"- Point differential: {_stat(stats, 'differential', 'pointDifferential', fmt=_dec2)}",
                    f"- Home: {_stat(stats, 'Home')} | Road: {_stat(stats, 'Road')}",
                    f"- Last 10: {_stat(stats, 'Last Ten Games')}",
                    f"- Shooting: FG {_stat(stats, 'fg_pct', fmt=_pct1)} | 3PT {_stat(stats, 'three_pct', fmt=_pct1)}",
                    f"- Ball control: A/TO ratio {_stat(stats, 'ast_to_ratio', fmt=_dec2)} | Turnovers/game {_stat(stats, 'avg_turnovers', fmt=_dec2)}",
                    f"- Rebounds/game: {_stat(stats, 'avg_rebounds', fmt=_dec2)} (off: {_stat(stats, 'avg_off_rebounds', fmt=_dec2)})",
                    f"- Defense: Steals/game {_stat(stats, 'avg_steals', fmt=_dec2)} | Blocks/game {_stat(stats, 'avg_blocks', fmt=_dec2)}",
                    f"- Key players (season avg): {players_str}",
                    f"- Rest days: {stats.get('rest_days', 'N/A')}",
                    f"- Streak: {streak_str}",
                ])
            if sport == "americanfootball_nfl":
                _dec1 = lambda v: f"{v:.1f}"
                _dec2 = lambda v: f"{v:.2f}"
                _int  = lambda v: str(int(v))
                _sgn  = lambda v: f"+{int(v)}" if v >= 0 else str(int(v))
                # ESPN NFL standings reports season-total points (no per-game field
                # and no gamesPlayed); derive games from the record for a usable rate.
                try:
                    gp_nfl = sum(float(stats.get(k, 0) or 0) for k in ("wins", "losses", "ties"))
                except (TypeError, ValueError):
                    gp_nfl = 0
                def _ppg(total_key: str) -> str:
                    try:
                        return f"{float(stats.get(total_key)) / gp_nfl:.1f}" if gp_nfl else "N/A"
                    except (TypeError, ValueError, ZeroDivisionError):
                        return "N/A"
                streak_raw = stats.get("streak")
                try:
                    sv_streak = int(float(streak_raw))
                    streak_str = f"W{sv_streak}" if sv_streak >= 0 else f"L{abs(sv_streak)}"
                except (TypeError, ValueError):
                    streak_str = "N/A"
                return "\n".join([
                    f"- Win %: {_stat(stats, 'winPercent', fmt=lambda v: f'{v:.1%}')}",
                    f"- Points/game: {_ppg('pointsFor')} (for) / {_ppg('pointsAgainst')} (against)",
                    f"- Yards/game: passing {_stat(stats, 'pass_yards_pg', fmt=_dec1)} | rushing {_stat(stats, 'rush_yards_pg', fmt=_dec1)}",
                    f"- Turnover diff: {_stat(stats, 'to_differential', fmt=_sgn)} (giveaways {_stat(stats, 'giveaways', fmt=_int)} / takeaways {_stat(stats, 'takeaways', fmt=_int)})",
                    f"- Sacks allowed/game: {_stat(stats, 'sacks_allowed_pg', fmt=_dec1)} | Defensive sacks/game: {_stat(stats, 'def_sacks_pg', fmt=_dec1)}",
                    f"- Offensive EPA/play: {_stat(stats, 'off_epa_per_play', fmt=_dec2)} | Defensive EPA allowed/play: {_stat(stats, 'def_epa_allowed_per_play', fmt=_dec2)}",
                    f"- Home: {_stat(stats, 'Home')} | Road: {_stat(stats, 'Road')}",
                    f"- Rest days: {_stat(stats, 'rest_days')}",
                    f"- Streak: {streak_str}",
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
            if sport == "mma_ufc":
                reach = stats.get("reach")
                try:
                    reach_str = f"{float(reach):.0f}\"" if reach not in (None, "") else "N/A"
                except (TypeError, ValueError):
                    reach_str = "N/A"
                return "\n".join([
                    f"- Age: {_stat(stats, 'age')} | Height: {stats.get('height') or 'N/A'} | Reach: {reach_str} | Stance: {stats.get('stance') or 'N/A'}",
                    f"- Striking: {_stat(stats, 'strikeLPM')} sig. strikes landed/min at {_stat(stats, 'strikeAccuracy')}% accuracy",
                    f"- Grappling: {_stat(stats, 'takedownAvg')} takedowns/15min at {_stat(stats, 'takedownAccuracy')}% accuracy | {_stat(stats, 'submissionAvg')} sub attempts/15min",
                    f"- Finishing rate: KO {_stat(stats, 'koPercentage')}% | TKO {_stat(stats, 'tkoPercentage')}% | decision {_stat(stats, 'decisionPercentage')}%",
                ])
            return "- Stats: Not available"

        is_ufc = sport == "mma_ufc"
        if is_ufc:
            # Individual sport: record comes as a "W-L-D" string; no roster/injuries.
            home_record = home_stats.get("record") or "N/A"
            away_record = away_stats.get("record") or "N/A"
            matchup_line = f"FIGHT: {home} vs {away}"
            home_label, away_label = f"FIGHTER A ({home})", f"FIGHTER B ({away})"
            home_extra = away_extra = ""
            player_rule = (
                "CRITICAL DATA RULE: Base your analysis ONLY on the two fighters named above and the "
                "stats provided. Do NOT pull in details from your training data — a fighter's recent "
                "results, finishes, injuries, weight-cut problems, or camp changes you recall may be "
                "outdated or wrong. If the data above is sparse, treat that fighter as an unknown rather "
                "than filling gaps from memory."
            )
        else:
            home_record = build_record(home_stats)
            away_record = build_record(away_stats)
            matchup_line = f"GAME: {away} @ {home}"
            home_label, away_label = f"HOME TEAM ({home})", f"AWAY TEAM ({away})"
            home_extra = (
                f"\n- Current roster: {home_roster or 'Not available'}\n"
                f"- Injuries (Out/Doubtful/Questionable):\n{fmt_injuries(home_injuries)}"
            )
            away_extra = (
                f"\n- Current roster: {away_roster or 'Not available'}\n"
                f"- Injuries (Out/Doubtful/Questionable):\n{fmt_injuries(away_injuries)}"
            )
            player_rule = (
                "CRITICAL PLAYER DATA RULE: Your reasoning may ONLY name specific players who appear in "
                "the roster or injury list provided above. Do NOT name any player from your training data "
                "who is not listed — rosters change constantly and your training data is stale. If a player "
                "you recall is not in the data above, they may have been traded, cut, or retired. Mentioning "
                "a player not in the provided data is a factual error. Base all player-specific claims "
                "strictly on the lists above."
            )

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

{matchup_line}
Sport: {sport}
Commence: {game.get('commence_time', 'Unknown')}{venue_line}

ODDS:
- {home} moneyline: {game.get('home_ml', 'N/A')} (implied: {game.get('home_implied', 0):.1%})
- {away} moneyline: {game.get('away_ml', 'N/A')} (implied: {game.get('away_implied', 0):.1%})
- Total line: {game.get('total_line', 'N/A')}
{series_block}{weather_block}
{home_label}:
- Record: {home_record}
{build_stats_block(home_stats, "home")}{home_extra}

{away_label}:
- Record: {away_record}
{build_stats_block(away_stats, "away")}{away_extra}

STATISTICAL MODEL ESTIMATE: {home} win probability = {base_prob:.1%}
{missing_note}
Apply your framework above to this data and estimate win probability. Weigh all relevant factors together — no single factor should dominate unless the data is overwhelmingly one-sided. In your reasoning, explain your logic naturally; do NOT cite rule numbers or use phrases like "per the framework" or "Framework #1".

{player_rule}

Reason to your number — do NOT state a probability first and justify it after. Work through the logic, and BEFORE committing to the number state three things explicitly: the market's implied home probability, the home probability you are settling on, and the gap between them in percentage points. Keep that gap proportionate to your evidence and to the magnitude limits in your framework — a large gap demands a concrete roster-level justification, never aggregates or already-priced information.

Respond ONLY with valid JSON in this exact format. The field order matters: "reasoning" comes FIRST and "adjusted_home_prob" LAST — your probability must be the conclusion of your reasoning, not the premise.
{{
  "reasoning": "<2-4 sentences. State the market's implied home %, the % you are settling on, and the point gap between them, and why that gap is justified.>",
  "confidence": "<low|medium|high>",
  "adjusted_home_prob": <float between 0 and 1 — must equal the settled % you stated in your reasoning>,
  "bet_recommendation": "<home_ml|away_ml|pass>"
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
