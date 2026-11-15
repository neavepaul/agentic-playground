#!/usr/bin/env bash
# Put Kip on the Pi's desktop: a desktop icon, a menu entry (Internet > Kip) and, by default,
# kiosk mode (Kip opens full screen when the desktop logs in). Run after deploy/install.sh,
# as your normal user:
#   bash deploy/desktop.sh             icon, menu entry and kiosk
#   bash deploy/desktop.sh --no-kiosk  icon and menu entry only (turns kiosk off if it was on)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPEN="$REPO/deploy/kip-open.sh"
ICON="$REPO/deploy/kip.svg"
APPS="$HOME/.local/share/applications"
AUTOSTART="$HOME/.config/autostart"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"

[ "$(id -u)" -ne 0 ] || { echo "Run this as your normal user, not root." >&2; exit 1; }
command -v chromium >/dev/null 2>&1 || { echo "Chromium is missing: sudo apt-get install chromium" >&2; exit 1; }
chmod +x "$OPEN"

entry() {  # entry <file> <extra Exec argument>
  cat > "$1" <<EOF
[Desktop Entry]
Type=Application
Name=Kip
Comment=Talk to Kip
Exec="$OPEN" $2
Icon=$ICON
Terminal=false
Categories=Network;
EOF
  chmod +x "$1"
}

mkdir -p "$APPS" "$DESKTOP"
entry "$APPS/kip.desktop" ""
entry "$DESKTOP/kip.desktop" ""
# The Pi's desktop (pcmanfm) shows an untrusted launcher by its file name and asks before running it.
# It reads the file the moment it appears, so mark it trusted and then make it reappear.
if gio set "$DESKTOP/kip.desktop" metadata::trust true 2>/dev/null; then
  mv "$DESKTOP/kip.desktop" "$APPS/kip.desktop.tmp"
  sleep 1
  mv "$APPS/kip.desktop.tmp" "$DESKTOP/kip.desktop"
fi
echo "Added Kip to the desktop and to the menu (Internet > Kip)."

if [ "${1:-}" = "--no-kiosk" ]; then
  rm -f "$AUTOSTART/kip-kiosk.desktop"
  echo "Kiosk mode is off."
else
  mkdir -p "$AUTOSTART"
  entry "$AUTOSTART/kip-kiosk.desktop" "--kiosk"
  echo "Kiosk mode is on: Kip opens full screen after the desktop logs in. Alt+F4 closes it."
  echo "Turn it off with: bash deploy/desktop.sh --no-kiosk"
fi
