"""
Backfills reasoning for today's not-yet-started games in broker_state.json.
Re-evaluates with the updated ClaudeAnalyst and patches only the 'reasoning' field.

Usage (from repo root, with venv active):
    python scripts/backfill_today.py
"""
import sys, os, json, shutil
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / '.env')

from SportsBettingBot.data.odds_fetcher import OddsFetcher
from SportsBettingBot.data.stats_fetcher import (
    ESPNStatsFetcher, MLBStatsFetcher, NBAStatsFetcher, WNBAStatsFetcher,
)
from SportsBettingBot.data.roster_fetcher import RosterFetcher
from SportsBettingBot.data.injury_fetcher import InjuryFetcher
from SportsBettingBot.models.claude_analyst import ClaudeAnalyst

ET  = ZoneInfo('America/New_York')
now = datetime.now(timezone.utc)
today_et = now.astimezone(ET).date()

STATE_PATH = REPO_ROOT / 'broker_state.json'
shutil.copy(STATE_PATH, str(STATE_PATH) + '.pre_backfill')
print('Backup saved to broker_state.json.pre_backfill\n')

with open(STATE_PATH) as f:
    state = json.load(f)

odds_f  = OddsFetcher(api_key=os.getenv('ODDS_API_KEY'))
espn    = ESPNStatsFetcher()
mlb_adv = MLBStatsFetcher()
wnba_st = WNBAStatsFetcher()
roster  = RosterFetcher()
injury  = InjuryFetcher()
claude  = ClaudeAnalyst(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Cache stats per sport so we don't re-fetch for every game
_stats_cache = {}

def get_stats_df(sport):
    if sport in _stats_cache:
        return _stats_cache[sport]
    if sport == 'baseball_mlb':
        espn_df = espn.get_team_stats(sport)
        adv_df  = mlb_adv.get_team_stats()
        if not espn_df.empty and not adv_df.empty:
            df = espn_df.merge(adv_df, on='team', how='left')
        else:
            df = espn_df
    elif sport == 'basketball_wnba':
        espn_df = espn.get_team_stats(sport)
        wnba_adv = wnba_st.get_team_stats()
        if not wnba_adv.empty and not espn_df.empty:
            df = espn_df.merge(wnba_adv, on='team', how='left')
        else:
            df = espn_df
    else:
        df = espn.get_team_stats(sport)
    _stats_cache[sport] = df
    return df

def get_team_stats(name, sport):
    df = get_stats_df(sport)
    if df.empty:
        return {}
    nick = name.split()[-1]
    row = df[df['team'].str.split().str[-1].str.lower() == nick.lower()]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()

# Cache odds per sport
_odds_cache = {}

def get_game_odds(sport, away, home):
    if sport not in _odds_cache:
        games = odds_f.get_upcoming_games(sport, ['draftkings','fanduel','betmgm'])
        _odds_cache[sport] = games
    for g in _odds_cache[sport]:
        if g['away_team'] == away and g['home_team'] == home:
            return g
    return None

updated = 0
errors  = 0

for section in ('open_bets', 'passed_games'):
    for bet in state.get(section, []):
        ct = bet.get('commence_time', '')
        if not ct:
            continue
        try:
            gdt = datetime.fromisoformat(ct.replace('Z', '+00:00')).astimezone(ET)
        except Exception:
            continue
        if gdt.date() != today_et or gdt <= now:
            continue

        away  = bet['away_team']
        home  = bet['home_team']
        sport = bet.get('sport', '')
        print('Re-evaluating: {} @ {}  [{}]'.format(away, home, section))

        try:
            game = get_game_odds(sport, away, home)
            if game is None:
                # Build a minimal game dict from the stored bet so Claude can still run
                game = {
                    'home_team': home,
                    'away_team': away,
                    'commence_time': ct,
                    'home_ml': bet.get('odds') if bet.get('bet_type') == 'home_ml' else None,
                    'away_ml': bet.get('odds') if bet.get('bet_type') == 'away_ml' else None,
                    'home_implied': bet.get('home_implied', 0.5),
                    'away_implied': bet.get('away_implied', 0.5),
                }
                print('  (no live odds — using stored bet data)')

            home_stats = get_team_stats(home, sport)
            away_stats = get_team_stats(away, sport)
            home_inj   = injury.get_team_injuries(home, sport=sport, max_age_minutes=60)
            away_inj   = injury.get_team_injuries(away, sport=sport, max_age_minutes=60)
            home_r     = roster.get_roster_string(home, sport=sport)
            away_r     = roster.get_roster_string(away, sport=sport)
            pitchers   = espn.get_starting_pitchers(home, away) if sport == 'baseball_mlb' else {}
            series_ctx = None

            result = claude.analyze_game(
                game, home_stats, away_stats,
                base_home_prob=game.get('home_implied', 0.5),
                home_injuries=home_inj, away_injuries=away_inj,
                home_roster=home_r, away_roster=away_r,
                sport=sport,
                starting_pitchers=pitchers,
                series_context=series_ctx,
            )

            old_reasoning = bet.get('reasoning', '')
            bet['reasoning'] = result['reasoning']
            print('  OLD: {}'.format(old_reasoning[:120]))
            print('  NEW: {}'.format(result['reasoning'][:120]))
            updated += 1

        except Exception as e:
            print('  ERROR: {}'.format(e))
            errors += 1

        print()

with open(STATE_PATH, 'w') as f:
    json.dump(state, f, indent=2, default=str)

print('Done. Updated={} Errors={}'.format(updated, errors))
print('broker_state.json written.')
