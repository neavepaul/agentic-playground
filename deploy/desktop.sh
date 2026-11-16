#!/usr/bin/env bash
# Put Kip on the Pi's desktop: a desktop icon, a menu entry (Internet > Kip) and, by default, Kip
# opening in a maximized window when the desktop logs in (minimize or close it like any window).
# Run after deploy/install.sh, as your normal user:
#   bash deploy/desktop.sh                 icon, menu entry, and Kip in a window at login
#   bash deploy/desktop.sh --kiosk         Kip full screen at login instead, with no window buttons
#                                          (Alt+F4 closes it): for a Pi that only ever shows Kip
#   bash deploy/desktop.sh --no-autostart  icon and menu entry only (--no-kiosk does the same)
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

case "${1:-}" in
  --no-autostart|--no-kiosk)
    rm -f "$AUTOSTART/kip-kiosk.desktop"
    echo "Kip no longer opens at login; the icon still opens it." ;;
  --kiosk)
    mkdir -p "$AUTOSTART"
    entry "$AUTOSTART/kip-kiosk.desktop" "--kiosk"
    echo "Kiosk mode is on: Kip opens full screen after the desktop logs in, with no window buttons."
    echo "Alt+F4 closes it. Back to a window with: bash deploy/desktop.sh" ;;
  *)
    mkdir -p "$AUTOSTART"
    entry "$AUTOSTART/kip-kiosk.desktop" "--start"
    echo "Kip opens in a maximized window after the desktop logs in: minimize or close it like any window."
    echo "Not at login: bash deploy/desktop.sh --no-autostart" ;;
esac
