#!/usr/bin/env bash
# Open Kip in Chromium on the Pi's own screen once the kip service answers.
#   kip-open.sh           an app window (the desktop icon and menu entry)
#   kip-open.sh --start   the same window, maximized (at login); minimize or close it as usual
#   kip-open.sh --kiosk   full screen with no browser bars or window buttons; Alt+F4 closes it
# Kip windows use their own Chromium profile, so they open as Kip even while a normal Chromium is running.
set -u

URL="http://127.0.0.1:8000"
PROFILE="$HOME/.config/kip-chromium"

# Kip loads its models at boot, so wait up to two minutes for it to answer.
for _ in $(seq 120); do
  curl -fsS -o /dev/null --max-time 2 "$URL/api/health" && break
  sleep 1
done

flags=(--user-data-dir="$PROFILE" --no-first-run --noerrdialogs --password-store=basic)
case "${1:-}" in
  --kiosk) exec chromium "${flags[@]}" --kiosk --disable-infobars --disable-session-crashed-bubble "$URL" ;;
  --start) exec chromium "${flags[@]}" --start-maximized --app="$URL" ;;
esac
exec chromium "${flags[@]}" --app="$URL"
