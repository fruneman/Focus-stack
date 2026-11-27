#!/usr/bin/env bash
# Jednorazowa instalacja Focus Stack na Fedorze (bez uprawnień administratora poza dnf).
# Po instalacji aplikacja jest w menu GNOME jako "Focus Stack".
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.local/share/focus-stack"
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"

echo "==> Instalacja wymaganych pakietów systemowych (dnf poprosi o hasło)"
sudo dnf install -y python3-gobject gtk4 libadwaita python3-opencv python3-numpy python3-pillow

echo "==> Kopiowanie programu do $DEST"
mkdir -p "$DEST" "$BIN" "$APPS"
rm -rf "$DEST/focusstack"
cp -r "$SRC/focusstack" "$DEST/"

cat > "$BIN/focus-stack" <<EOF
#!/usr/bin/env bash
cd "$DEST" && exec python3 -m focusstack "\$@"
EOF
chmod +x "$BIN/focus-stack"

cat > "$APPS/io.github.focusstack.FocusStack.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Focus Stack
Comment=Łączenie zdjęć makro o różnej płaszczyźnie ostrości
Exec=$BIN/focus-stack %F
Icon=camera-photo
Terminal=false
Categories=Graphics;Photography;
MimeType=image/jpeg;image/tiff;image/png;
StartupNotify=true
EOF
update-desktop-database "$APPS" 2>/dev/null || true

echo "==> Gotowe. Uruchom 'Focus Stack' z menu aplikacji."
