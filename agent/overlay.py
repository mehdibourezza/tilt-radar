"""
TiltRadar Overlay — persistent HUD + transient notification cards.

Two visual components:
  1. Persistent HUD (left side): always-visible tilt bars for all 10 players,
     grouped as YOU / ALLIES / ENEMIES. Updates every 5 seconds silently.
  2. Notification cards (right side): transient pop-ups when tilt escalates
     meaningfully. Auto-dismiss after 7 seconds. Anti-spam rules apply.

Requires:
  - LoL in BORDERLESS WINDOWED mode (Settings → Video → Window Mode)
  - Python tkinter (built-in)
  - Windows only (uses ctypes for click-through)
"""

import tkinter as tk
import threading
import ctypes
import queue
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# --- Colors ---
BG_TRANSPARENT      = "#0f0800"   # used as transparent key on notification window
BG_HUD              = "#0d0d1a"   # dark navy for HUD panel
HUD_ALPHA           = 0.90

TEXT_PRIMARY        = "#ffffff"
TEXT_SECONDARY      = "#e0c89a"   # warm beige — champion names
TEXT_DIM            = "#8888aa"   # section headers
BAR_BG_COLOR        = "#2a2a40"   # visible bar track (not near-black)

# Enemy: orange/red
BORDER_HIGH_ENEMY   = "#ff6a00"
BORDER_MED_ENEMY    = "#ffaa00"
BAR_HIGH_ENEMY      = "#ff4500"
BAR_MED_ENEMY       = "#ff9500"
BAR_ZERO_ENEMY      = "#3a2010"   # visible dark-orange trough

# Self: purple/violet
BORDER_HIGH_SELF    = "#cc44ff"
BORDER_MED_SELF     = "#9922cc"
BAR_HIGH_SELF       = "#aa00ff"
BAR_MED_SELF        = "#7700bb"
BAR_ZERO_SELF       = "#25103a"   # visible dark-purple trough

# Ally: cyan/teal
BORDER_HIGH_ALLY    = "#00ccff"
BORDER_MED_ALLY     = "#0099cc"
BAR_HIGH_ALLY       = "#00aadd"
BAR_MED_ALLY        = "#007799"
BAR_ZERO_ALLY       = "#0a2535"   # visible dark-cyan trough

FONT = "Arial"

# --- Notification card dimensions (right side) ---
NOTIF_WIDTH         = 280
NOTIF_HEIGHT        = 60
NOTIF_GAP           = 8
SCREEN_MARGIN_RIGHT = 20
SCREEN_MARGIN_TOP   = 80
FADE_DURATION       = 7.0

# --- HUD panel dimensions (left side) ---
HUD_WIDTH           = 230
HUD_BAR_W           = 80
HUD_ROW_H           = 22          # taller rows to avoid clipping on any DPI
HUD_PAD_X           = 8
HUD_X               = 10
HUD_Y               = 120         # below LoL's top-left HP bars (adjust if needed)

# --- Anti-spam thresholds for pop-up notifications ---
MIN_SCORE_TO_NOTIFY = 0.50
MIN_SCORE_JUMP      = 0.20


@dataclass
class Notification:
    champion: str
    summoner: str
    tilt_score: float
    tilt_type: str
    exploit: str | None
    new_signals: list[str] = field(default_factory=list)
    player_type: str = "enemy"


class TiltOverlay:
    """
    Transparent, always-on-top, click-through overlay.

    Manages two sub-windows in a single tkinter thread:
      - Persistent HUD panel (Toplevel, left): shows tilt bars for all 10 players
      - Notification host (Tk root, right): transparent container for pop-up cards

    Usage:
        overlay = TiltOverlay()
        overlay.start()
        overlay.update_hud(players_list)    # called every report — updates HUD bars
        overlay.notify(Notification(...))   # called when a spike is worth showing
        overlay.stop()
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._root: tk.Tk | None = None
        self._hud: tk.Toplevel | None = None

        # Notification state
        self._active_cards: list[dict] = []
        self._last_notified: dict[str, float] = {}

        # HUD state — filled once first report arrives
        self._hud_rows: dict[str, dict] = {}    # summonerName → {bar_canvas, bar_rect, score_var, score_lbl}
        self._hud_built = False
        self._hud_placeholder: tk.Label | None = None
        self._hud_content: tk.Frame | None = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._queue.put(None)

    def notify(self, notif: Notification):
        """Queue a pop-up notification (thread-safe)."""
        self._queue.put(notif)

    def update_hud(self, players: list[dict]):
        """Queue a HUD refresh with the latest player list (thread-safe)."""
        self._queue.put(players)   # list → HUD update, Notification → card pop-up

    def should_notify(self, summoner: str, score: float, new_signals: list[str]) -> bool:
        """Anti-spam: only pop-up on meaningful changes."""
        last = self._last_notified.get(summoner, 0.0)
        if score < MIN_SCORE_TO_NOTIFY:
            return False
        if last == 0.0 and score >= MIN_SCORE_TO_NOTIFY:
            return True   # first time crossing the threshold
        if score - last >= MIN_SCORE_JUMP:
            return True   # significant escalation
        if new_signals:
            return True   # new signal type appeared
        return False

    # =========================================================================
    # Tkinter thread
    # =========================================================================

    def _run(self):
        # --- Root window: transparent, right side, hosts notification cards ---
        self._root = tk.Tk()
        root = self._root

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 1.0)
        root.configure(bg=BG_TRANSPARENT)
        root.attributes("-transparentcolor", BG_TRANSPARENT)

        # update() forces the HWND to be created so Win32 calls work correctly.
        # click-through is applied AFTER update() — calling it before gives an invalid handle.
        root.update()
        screen_w = root.winfo_screenwidth()
        notif_x = screen_w - NOTIF_WIDTH - SCREEN_MARGIN_RIGHT
        notif_h = (NOTIF_HEIGHT + NOTIF_GAP) * 5
        root.geometry(f"{NOTIF_WIDTH}x{notif_h}+{notif_x}+{SCREEN_MARGIN_TOP}")
        self._enable_click_through(root)

        # --- HUD window: solid panel, left side, always visible ---
        self._hud = tk.Toplevel(root)
        hud = self._hud
        hud.overrideredirect(True)
        hud.attributes("-topmost", True)
        hud.attributes("-alpha", HUD_ALPHA)
        hud.configure(bg=BG_HUD)
        hud.geometry(f"{HUD_WIDTH}x400+{HUD_X}+{HUD_Y}")  # generous initial height; resized after build
        hud.update()   # ensure HUD HWND exists before Win32 click-through call
        self._enable_click_through(hud)

        # HUD header bar
        header = tk.Frame(hud, bg="#12122a")
        header.pack(fill="x")
        tk.Label(
            header, text="◈  TILT RADAR",
            fg="#7070cc", bg="#12122a",
            font=(FONT, 7, "bold"),
            anchor="w", padx=HUD_PAD_X, pady=4,
        ).pack(side="left")

        # Thin separator
        tk.Frame(hud, bg="#2a2a4a", height=1).pack(fill="x")

        # Placeholder shown before first game data
        self._hud_placeholder = tk.Label(
            hud, text="Waiting for game…",
            fg=TEXT_DIM, bg=BG_HUD,
            font=(FONT, 7), padx=HUD_PAD_X, pady=8,
        )
        self._hud_placeholder.pack(fill="x")

        # Content area — player rows go here
        self._hud_content = tk.Frame(hud, bg=BG_HUD)
        self._hud_content.pack(fill="both", expand=True, pady=(0, 4))

        self._poll_queue()
        root.mainloop()

    def _poll_queue(self):
        """Drain the queue without blocking the tkinter event loop."""
        try:
            while True:
                item = self._queue.get_nowait()
                if item is None:
                    self._root.destroy()
                    return
                elif isinstance(item, list):
                    try:
                        self._refresh_hud(item)
                    except Exception:
                        logger.exception("HUD refresh failed")
                else:
                    try:
                        self._show_card(item)
                    except Exception:
                        logger.exception("Notification card failed")
        except queue.Empty:
            pass
        self._root.after(100, self._poll_queue)

    # =========================================================================
    # Persistent HUD
    # =========================================================================

    def _refresh_hud(self, players: list[dict]):
        if not self._hud_built:
            self._build_hud(players)
        else:
            self._update_hud_bars(players)

    def _build_hud(self, players: list[dict]):
        """Create all player rows once, on the first report."""
        # Remove placeholder
        if self._hud_placeholder:
            self._hud_placeholder.destroy()
            self._hud_placeholder = None

        self_list  = [p for p in players if p.get("player_type") == "self"]
        ally_list  = [p for p in players if p.get("player_type") == "ally"]
        enemy_list = [p for p in players if p.get("player_type") == "enemy"]

        for section, group in [("YOU", self_list), ("ALLIES", ally_list), ("ENEMIES", enemy_list)]:
            if not group:
                continue
            # Section label
            tk.Label(
                self._hud_content, text=section,
                fg=TEXT_DIM, bg=BG_HUD,
                font=(FONT, 6, "bold"),
                anchor="w", padx=HUD_PAD_X, pady=2,
            ).pack(fill="x", pady=(4, 1))
            for player in group:
                self._add_player_row(player)

        if not self._hud_rows:
            # No rows were added (empty player list) — restore placeholder and try again next report
            if self._hud_placeholder is None:
                self._hud_placeholder = tk.Label(
                    self._hud, text="Waiting for game…",
                    fg=TEXT_DIM, bg=BG_HUD,
                    font=(FONT, 7), padx=HUD_PAD_X, pady=8,
                )
                self._hud_placeholder.pack(fill="x")
            return

        self._hud_built = True
        # Resize HUD to exact content height — schedule via after() so mainloop
        # has had one cycle to render widgets (winfo_reqheight returns 0 before that)
        self._root.after(50, self._fit_hud_height)

    def _fit_hud_height(self):
        """Shrink the HUD to exactly its content height. Called via after() so mainloop has rendered."""
        self._hud.update_idletasks()
        h = self._hud.winfo_reqheight()
        if h > 0:
            self._hud.geometry(f"{HUD_WIDTH}x{h}+{HUD_X}+{HUD_Y}")

    def _add_player_row(self, player: dict):
        name     = player.get("summonerName", "?")
        champion = player.get("championName", "?")
        score    = player.get("tilt_score", 0.0)
        ptype    = player.get("player_type", "enemy")

        # No fixed height — let content determine row height (avoids DPI clipping)
        row = tk.Frame(self._hud_content, bg=BG_HUD)
        row.pack(fill="x", padx=HUD_PAD_X, pady=2)

        # Champion name, truncated
        champ_text = champion[:11] if len(champion) > 11 else champion
        tk.Label(
            row, text=champ_text,
            fg=TEXT_SECONDARY, bg=BG_HUD,
            font=(FONT, 8), anchor="w", width=10,
        ).pack(side="left")

        # Tilt bar (Canvas — allows partial-width fill)
        bar_canvas = tk.Canvas(
            row, bg=BAR_BG_COLOR,
            width=HUD_BAR_W, height=10,
            bd=0, highlightthickness=0,
        )
        bar_canvas.pack(side="left", padx=(4, 4))

        fill_w     = int(score * HUD_BAR_W)
        bar_color  = self._resolve_bar_color(score, ptype)
        bar_rect   = bar_canvas.create_rectangle(0, 0, fill_w, 10, fill=bar_color, outline="")

        # Score percentage (right-aligned)
        score_var = tk.StringVar(value=f"{score:.0%}")
        score_lbl = tk.Label(
            row, textvariable=score_var,
            fg=bar_color if score > 0 else TEXT_DIM,
            bg=BG_HUD, font=(FONT, 7), anchor="e", width=4,
        )
        score_lbl.pack(side="right")

        self._hud_rows[name] = {
            "bar_canvas": bar_canvas,
            "bar_rect":   bar_rect,
            "score_var":  score_var,
            "score_lbl":  score_lbl,
            "ptype":      ptype,
        }

    def _update_hud_bars(self, players: list[dict]):
        """Update existing bar widths and score labels in place."""
        for player in players:
            name  = player.get("summonerName", "")
            score = player.get("tilt_score", 0.0)
            row   = self._hud_rows.get(name)
            if row is None:
                continue

            ptype      = row["ptype"]
            bar_color  = self._resolve_bar_color(score, ptype)
            fill_w     = int(score * HUD_BAR_W)

            row["bar_canvas"].coords(row["bar_rect"], 0, 0, fill_w, 10)
            row["bar_canvas"].itemconfig(row["bar_rect"], fill=bar_color)
            row["score_var"].set(f"{score:.0%}")
            row["score_lbl"].config(fg=bar_color if score > 0 else TEXT_DIM)

    @staticmethod
    def _resolve_bar_color(score: float, ptype: str) -> str:
        if ptype == "self":
            if score >= 0.70: return BAR_HIGH_SELF
            if score >= 0.45: return BAR_MED_SELF
            return BAR_ZERO_SELF
        if ptype == "ally":
            if score >= 0.70: return BAR_HIGH_ALLY
            if score >= 0.45: return BAR_MED_ALLY
            return BAR_ZERO_ALLY
        # enemy
        if score >= 0.70: return BAR_HIGH_ENEMY
        if score >= 0.45: return BAR_MED_ENEMY
        return BAR_ZERO_ENEMY

    # =========================================================================
    # Transient notification cards (right side)
    # =========================================================================

    def _show_card(self, notif: Notification):
        high_color, med_color, bar_high, bar_med = {
            "self":  (BORDER_HIGH_SELF,  BORDER_MED_SELF,  BAR_HIGH_SELF,  BAR_MED_SELF),
            "ally":  (BORDER_HIGH_ALLY,  BORDER_MED_ALLY,  BAR_HIGH_ALLY,  BAR_MED_ALLY),
            "enemy": (BORDER_HIGH_ENEMY, BORDER_MED_ENEMY, BAR_HIGH_ENEMY, BAR_MED_ENEMY),
        }.get(notif.player_type, (BORDER_HIGH_ENEMY, BORDER_MED_ENEMY, BAR_HIGH_ENEMY, BAR_MED_ENEMY))

        border_color = high_color if notif.tilt_score >= 0.70 else med_color
        bar_color    = bar_high   if notif.tilt_score >= 0.70 else bar_med

        card = tk.Frame(
            self._root,
            bg=border_color, bd=1, relief="flat",
            width=NOTIF_WIDTH, height=NOTIF_HEIGHT,
        )
        card.pack_propagate(False)

        inner = tk.Frame(card, bg="#0a0a16", padx=6, pady=4)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        # Row 1: champion name + tilt type + score
        top_row = tk.Frame(inner, bg="#0a0a16")
        top_row.pack(fill="x")

        tk.Label(
            top_row, text=notif.champion,
            fg=TEXT_PRIMARY, bg="#0a0a16",
            font=(FONT, 9, "bold"), anchor="w",
        ).pack(side="left")

        tk.Label(
            top_row, text=f"  {notif.tilt_type.upper()}",
            fg=border_color, bg="#0a0a16",
            font=(FONT, 8, "bold"), anchor="w",
        ).pack(side="left")

        tk.Label(
            top_row, text=f"{notif.tilt_score:.0%}",
            fg=border_color, bg="#0a0a16",
            font=(FONT, 9, "bold"), anchor="e",
        ).pack(side="right")

        # Row 2: mini tilt bar
        bar_row = tk.Frame(inner, bg="#0a0a16")
        bar_row.pack(fill="x", pady=(2, 2))

        bar_bg = tk.Frame(bar_row, bg=BAR_BG_COLOR, height=3)
        bar_bg.pack(fill="x")
        bar_bg.pack_propagate(False)

        fill_px = int(notif.tilt_score * NOTIF_WIDTH * 0.88)
        tk.Frame(bar_bg, bg=bar_color, width=fill_px, height=3).place(x=0, y=0)

        # Row 3: exploit hint
        if notif.exploit:
            short = notif.exploit[:48] + "…" if len(notif.exploit) > 48 else notif.exploit
            tk.Label(
                inner, text=f"→ {short}",
                fg=TEXT_SECONDARY, bg="#0a0a16",
                font=(FONT, 7), anchor="w",
                wraplength=NOTIF_WIDTH - 16, justify="left",
            ).pack(fill="x")

        card.place(x=0, y=0)
        self._restack_cards()

        card_entry = {"frame": card, "after_id": None}
        self._active_cards.insert(0, card_entry)

        after_id = self._root.after(
            int(FADE_DURATION * 1000),
            lambda: self._dismiss_card(card_entry),
        )
        card_entry["after_id"] = after_id

    def _restack_cards(self):
        y = 0
        for entry in self._active_cards:
            entry["frame"].place(x=0, y=y)
            y += NOTIF_HEIGHT + NOTIF_GAP

    def _dismiss_card(self, card_entry: dict):
        try:
            card_entry["frame"].destroy()
        except Exception:
            pass
        if card_entry in self._active_cards:
            self._active_cards.remove(card_entry)
        self._restack_cards()

    # =========================================================================
    # Windows click-through
    # =========================================================================

    @staticmethod
    def _enable_click_through(win: tk.Misc):
        """Make a tkinter window fully click-through (Windows only).

        Must be called AFTER win.update() so the HWND exists.
        Uses GetParent(winfo_id()) to get the real outer Win32 window handle
        (winfo_id() alone returns the inner Tk frame, a child window — setting
        WS_EX_LAYERED on a child window causes undefined / black rendering).
        """
        try:
            hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
            if not hwnd:
                hwnd = win.winfo_id()   # fallback for overrideredirect top-levels
            if not hwnd:
                return
            GWL_EXSTYLE       = -20
            WS_EX_LAYERED     = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT
            )
        except Exception as e:
            logger.warning(f"Could not enable click-through: {e}")
