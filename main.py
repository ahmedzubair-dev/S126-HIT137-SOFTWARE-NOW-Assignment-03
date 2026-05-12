"""
HIT137 Assignment 3 - Spot the Difference Game
=================================================
A desktop application demonstrating OOP, Tkinter GUI, and OpenCV image processing.

Classes:
    - ImageAlterer   : Handles all OpenCV image manipulation
    - GameState      : Manages game logic, scoring, and state
    - SpotTheDiffApp : Main Tkinter application / controller
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import random
import math


# ─────────────────────────────────────────────
#  1. ALTERATION HIERARCHY  (Polymorphism)
#
#  Alteration is an abstract base class.
#  Each subclass overrides apply() with its own
#  logic — this is classical OOP polymorphism:
#  one interface, five behaviours.
# ─────────────────────────────────────────────

class Alteration:
    """
    Abstract base class for a single image patch alteration.
    Subclasses MUST override apply().
    Demonstrates polymorphism: ImageAlterer calls alteration.apply()
    without knowing which concrete type it holds.
    """
    def apply(self, patch: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement apply()")

    def __repr__(self):
        return self.__class__.__name__


class ColourShiftAlteration(Alteration):
    """Shifts the hue and brightness of the patch in HSV space."""
    def apply(self, patch: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(patch.astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        shift = random.choice([20, -20, 30, -30])
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-30, 30), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8),
                            cv2.COLOR_HSV2BGR).astype(np.float32)


class BlurAlteration(Alteration):
    """Applies a Gaussian blur to the patch."""
    def apply(self, patch: np.ndarray) -> np.ndarray:
        ksize = random.choice([9, 11, 13])
        return cv2.GaussianBlur(patch.astype(np.uint8),
                                (ksize, ksize), 0).astype(np.float32)


class PixelSwapAlteration(Alteration):
    """Mirrors the patch horizontally (pixel swap)."""
    def apply(self, patch: np.ndarray) -> np.ndarray:
        return patch[:, ::-1, :]


class ContrastBoostAlteration(Alteration):
    """Amplifies local contrast by a random factor."""
    def apply(self, patch: np.ndarray) -> np.ndarray:
        alpha = random.uniform(1.4, 1.8)
        return np.clip(patch * alpha, 0, 255)


class TintOverlayAlteration(Alteration):
    """Adds a subtle colour tint to the patch."""
    TINTS = [(0, 0, 80), (0, 80, 0), (80, 0, 0), (0, 60, 60)]

    def apply(self, patch: np.ndarray) -> np.ndarray:
        tint = np.array(random.choice(self.TINTS), dtype=np.float32)
        return np.clip(patch + tint, 0, 255)


# ─────────────────────────────────────────────
#  2. IMAGE ALTERER  (OpenCV / image processing)
# ─────────────────────────────────────────────

class ImageAlterer:
    """
    Clones an image and applies exactly 5 non-overlapping alterations.
    Uses the Alteration hierarchy: calls alteration.apply() polymorphically
    without any if/elif branching — the correct behaviour is selected
    automatically based on the concrete subclass instance.
    """

    # Pool of concrete Alteration subclass instances
    ALTERATION_POOL = [
        ColourShiftAlteration(),
        BlurAlteration(),
        PixelSwapAlteration(),
        ContrastBoostAlteration(),
        TintOverlayAlteration(),
    ]

    MIN_PATCH_FRAC = 0.06
    MAX_PATCH_FRAC = 0.14
    NUM_DIFFERENCES = 5
    MAX_PLACEMENT_ATTEMPTS = 1000

    def __init__(self):
        self.difference_regions = []

    def create_modified_image(self, original_bgr: np.ndarray):
        """
        Clone the image and apply exactly 5 non-overlapping alterations.
        Returns (modified_bgr, difference_regions).
        """
        self.difference_regions = []
        modified = original_bgr.copy()
        h, w = original_bgr.shape[:2]
        if w < 250 or h < 250:
            raise ValueError("Image is too small. Please choose an image at least 250x250 pixels.")

        placed = 0
        attempts = 0

        while placed < self.NUM_DIFFERENCES and attempts < self.MAX_PLACEMENT_ATTEMPTS:
            attempts += 1

            pw = random.randint(int(w * self.MIN_PATCH_FRAC),
                                int(w * self.MAX_PATCH_FRAC))
            ph = random.randint(int(h * self.MIN_PATCH_FRAC),
                                int(h * self.MAX_PATCH_FRAC))

            margin = 10
            
            if w - pw - margin <= margin or h - ph - margin <= margin:
                continue
            x = random.randint(margin, w - pw - margin)
            y = random.randint(margin, h - ph - margin)

            new_region = (x, y, pw, ph)
            if self._overlaps_any(new_region):
                continue

            # True independent random choice — any type can appear multiple
            # times, satisfying "type chosen randomly" per the brief.
            alteration = random.choice(self.ALTERATION_POOL)
            modified = self._apply_patch(modified, x, y, pw, ph, alteration)
            self.difference_regions.append(new_region)
            placed += 1
        if placed < self.NUM_DIFFERENCES:
            raise ValueError("Could not place all 5 differences. Please choose a larger or less narrow image")
        return modified, self.difference_regions

    def _overlaps_any(self, region, padding=10):
        rx, ry, rw, rh = region
        for (ox, oy, ow, oh) in self.difference_regions:
            if not (rx + rw + padding <= ox or
                    ox + ow + padding <= rx or
                    ry + rh + padding <= oy or
                    oy + oh + padding <= ry):
                return True
        return False

    def _apply_patch(self, img: np.ndarray, x, y, w, h,
                     alteration: Alteration) -> np.ndarray:
        """
        Extract the patch, call alteration.apply() polymorphically,
        then write the result back into a copy of the image.
        """
        result = img.copy()
        patch = result[y:y+h, x:x+w].astype(np.float32)
        altered_patch = alteration.apply(patch)          # ← polymorphic call
        result[y:y+h, x:x+w] = altered_patch.astype(np.uint8)
        return result


# ─────────────────────────────────────────────
#  2. GAME STATE
# ─────────────────────────────────────────────

class GameState:
    """
    Tracks all runtime game state: which differences are found,
    mistake count, cumulative score, and whether the round is active.
    """

    MAX_MISTAKES = 3

    def __init__(self):
        self._differences_found = []
        self._mistakes = 0
        self._cumulative_score = 0
        self._locked = False
        self._revealed = False
        self._revealed_indices = []

    def new_round(self, num_differences: int):
        self._differences_found = [False] * num_differences
        self._mistakes = 0
        self._locked = False
        self._revealed = False
        self._revealed_indices = []

    @property
    def mistakes(self):
        return self._mistakes

    @property
    def cumulative_score(self):
        return self._cumulative_score

    @property
    def is_locked(self):
        return self._locked

    @property
    def is_revealed(self):
        return self._revealed

    @property
    def found_count(self):
        return sum(self._differences_found)

    @property
    def remaining(self):
        return len(self._differences_found) - self.found_count

    @property
    def all_found(self):
        return all(self._differences_found)

    def is_found(self, index: int):
        return self._differences_found[index]

    def unfound_indices(self):
        return [i for i, f in enumerate(self._differences_found) if not f]

    def register_correct_click(self, index: int):
        if not self._differences_found[index]:
            self._differences_found[index] = True
            self._cumulative_score += 10
            # Lock immediately once all differences are found so no
            # further clicks can register as mistakes.
            if all(self._differences_found):
                self._locked = True

    def register_mistake(self):
        self._mistakes += 1
        if self._mistakes >= self.MAX_MISTAKES:
            self._locked = True

    def reveal(self):
        self._locked = True
        self._revealed = True
        # Remember which indices were auto-revealed (not player-found) for blue circles
        self._revealed_indices = [i for i, f in enumerate(self._differences_found) if not f]
        # Mark all as found so remaining counter drops to 0
        self._differences_found = [True] * len(self._differences_found)

    def is_auto_revealed(self, index: int):
        """True if this difference was shown by Reveal button, not found by the player."""
        return index in self._revealed_indices

    def check_click(self, click_x, click_y, regions, tolerance=30):
        for i in self.unfound_indices():
            rx, ry, rw, rh = regions[i]
            cx_region = rx + rw // 2
            cy_region = ry + rh // 2
            dist = math.sqrt((click_x - cx_region) ** 2 + (click_y - cy_region) ** 2)
            half_diag = math.sqrt(rw ** 2 + rh ** 2) / 2
            if dist <= max(tolerance, half_diag * 0.6):
                return i
        return -1
    def check_any_difference_click(self, click_x, click_y, regions, tolerance=30):
        for i, (rx, ry, rw, rh) in enumerate(regions):
            cx_region = rx + rw // 2
            cy_region = ry + rh // 2
            dist = math.sqrt((click_x - cx_region) ** 2 + (click_y - cy_region) ** 2)
            half_diag = math.sqrt(rw ** 2 + rh ** 2) / 2
            if dist <= max(tolerance, half_diag * 0.6):
                return i
        return -1


# ─────────────────────────────────────────────
#  3. MAIN APPLICATION  (Tkinter GUI)
# ─────────────────────────────────────────────

class SpotTheDiffApp(tk.Tk):
    """
    Main application window. Inherits from tk.Tk.
    Coordinates ImageAlterer and GameState; manages all Tkinter widgets.

    KEY DESIGN: _refresh_canvas() always recomposes from clean images +
    current game state. This guarantees circles are always correct.
    """

    WINDOW_TITLE   = "Spot the Difference  |  HIT137"
    DISPLAY_HEIGHT = 360
    PANEL_WIDTH    = 460
    BG_COLOR       = "#1a1a2e"
    ACCENT         = "#e94560"
    TEXT_COLOR     = "#eaeaea"
    COLOR_FOUND    = (0,   0,   220)   # BGR red
    COLOR_REVEAL   = (200, 100,   0)   # BGR blue

    def __init__(self):
        super().__init__()

        self._alterer = ImageAlterer()
        self._state   = GameState()

        # Clean (circle-free) scaled display images
        self._clean_orig = None
        self._clean_mod  = None

        self._regions      = []
        self._orig_regions = []
        self._scale        = 1.0
        self._img_x_off    = 0    # horizontal centering offset within panel
        self._img_y_off    = 0    # vertical centering offset within panel

        self._tk_orig = None
        self._tk_mod  = None

        self._setup_ui()
        self._update_hud()

    # ── UI construction ───────────────────────────────────────────────

    def _setup_ui(self):
        self.title(self.WINDOW_TITLE)
        self.configure(bg=self.BG_COLOR)
        self.resizable(False, False)

        # Title bar
        title_bar = tk.Frame(self, bg=self.BG_COLOR)
        title_bar.pack(fill=tk.X, padx=16, pady=(12, 0))
        tk.Label(title_bar, text="SPOT  THE  DIFFERENCE",
                 font=("Courier New", 18, "bold"),
                 bg=self.BG_COLOR, fg=self.ACCENT).pack(side=tk.LEFT)
        tk.Label(title_bar, text="HIT137 · Assignment 3",
                 font=("Courier New", 10),
                 bg=self.BG_COLOR, fg="#888").pack(side=tk.RIGHT, pady=4)

        # Toolbar
        toolbar = tk.Frame(self, bg="#0f3460", pady=8)
        toolbar.pack(fill=tk.X, pady=(8, 0))

        btn_style = dict(
            font=("Courier New", 10, "bold"),
            bg=self.ACCENT, fg="white",
            activebackground="#c73652", activeforeground="white",
            relief=tk.FLAT, padx=14, pady=6, cursor="hand2"
        )

        self._btn_load = tk.Button(toolbar, text="Load Image",
                                   command=self._load_image, **btn_style)
        self._btn_load.pack(side=tk.LEFT, padx=(12, 6))

        self._btn_reveal = tk.Button(toolbar, text="Reveal All",
                                     command=self._reveal_all,
                                     state=tk.DISABLED, **btn_style)
        self._btn_reveal.pack(side=tk.LEFT, padx=6)

        # HUD
        hud = tk.Frame(toolbar, bg="#0f3460")
        hud.pack(side=tk.RIGHT, padx=12)
        lbl_style = dict(bg="#0f3460", fg=self.TEXT_COLOR,
                         font=("Courier New", 10))
        self._lbl_remaining = tk.Label(hud, text="Remaining: -", **lbl_style)
        self._lbl_remaining.grid(row=0, column=0, padx=10)
        self._lbl_mistakes = tk.Label(hud, text="Mistakes: 0/3", **lbl_style)
        self._lbl_mistakes.grid(row=0, column=1, padx=10)
        self._lbl_score = tk.Label(hud, text="Score: 0", **lbl_style)
        self._lbl_score.grid(row=0, column=2, padx=10)

        # Column headers
        header_frame = tk.Frame(self, bg=self.BG_COLOR)
        header_frame.pack(fill=tk.X)
        lh = dict(bg=self.BG_COLOR, fg="#888", font=("Courier New", 9))
        tk.Label(header_frame, text="ORIGINAL  (reference)",
                 width=self.PANEL_WIDTH // 7, **lh).pack(
            side=tk.LEFT, padx=(self.PANEL_WIDTH // 3, 0))
        tk.Label(header_frame, text="MODIFIED  (click here)",
                 width=self.PANEL_WIDTH // 7, **lh).pack(
            side=tk.LEFT, padx=(self.PANEL_WIDTH // 3, 0))

        # Canvas
        canvas_frame = tk.Frame(self, bg="#111")
        canvas_frame.pack()
        total_w = self.PANEL_WIDTH * 2 + 4

        self._canvas = tk.Canvas(canvas_frame,
                                 width=total_w, height=self.DISPLAY_HEIGHT,
                                 bg="#0a0a1a", highlightthickness=0,
                                 cursor="crosshair")
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        mid = self.DISPLAY_HEIGHT // 2
        self._canvas.create_text(self.PANEL_WIDTH // 2, mid,
                                 text="Load an image to begin",
                                 fill="#444", font=("Courier New", 14))
        self._canvas.create_text(self.PANEL_WIDTH + self.PANEL_WIDTH // 2, mid,
                                 text="Load an image to begin",
                                 fill="#444", font=("Courier New", 14))

        # Status bar
        self._status_var = tk.StringVar(value="Load an image to start playing.")
        tk.Label(self, textvariable=self._status_var,
                 font=("Courier New", 9), bg="#0f3460",
                 fg="#aaa", anchor=tk.W, padx=12, pady=4).pack(
            fill=tk.X, side=tk.BOTTOM)

    # ── image loading ─────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.JPG *.PNG *.BMP"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", f"Could not open image:\n{path}")
            return
        try:
            mod_bgr, self._orig_regions = self._alterer.create_modified_image(bgr)
        except ValueError as e:
            messagebox.showerror("Image not suitable",str(e))
            return
        self._scale, self._clean_orig, self._clean_mod, \
            self._img_x_off, self._img_y_off = self._scale_for_display(bgr, mod_bgr)

        # Scale regions to display coords only — NO centering offset baked in.
        # The offset is added separately when drawing circles (_refresh_canvas)
        # so that click coords (which have the offset subtracted) match correctly.
        self._regions = [
            (int(x * self._scale),
             int(y * self._scale),
             int(w * self._scale),
             int(h * self._scale))
            for (x, y, w, h) in self._orig_regions
        ]

        self._state.new_round(len(self._orig_regions))
        self._refresh_canvas()
        self._update_hud()
        self._btn_reveal.config(state=tk.NORMAL)
        self._set_status("Image loaded. Find all 5 differences in the RIGHT panel!")

    def _scale_for_display(self, orig, mod):
        oh, ow = orig.shape[:2]
        scale = min(self.PANEL_WIDTH / ow, self.DISPLAY_HEIGHT / oh, 1.0)
        nw, nh = int(ow * scale), int(oh * scale)
        d_orig = cv2.resize(orig, (nw, nh), interpolation=cv2.INTER_AREA)
        d_mod  = cv2.resize(mod,  (nw, nh), interpolation=cv2.INTER_AREA)
        # Offsets to center the image within its panel
        x_off = (self.PANEL_WIDTH   - nw) // 2
        y_off = (self.DISPLAY_HEIGHT - nh) // 2
        return scale, d_orig, d_mod, x_off, y_off

    # ── rendering ────────────────────────────────────────────────────

    def _refresh_canvas(self):
        """
        Recompose both frames from clean images + current state, then
        redraw the canvas. Called after every state change.
        """
        if self._clean_orig is None:
            return

        frame_orig = self._clean_orig.copy()
        frame_mod  = self._clean_mod.copy()

        for i, (rx, ry, rw, rh) in enumerate(self._regions):
            # cx/cy are pure image-pixel coords — NO canvas offset.
            # cv2.circle draws on the image array itself, which has no
            # knowledge of where the image sits on the canvas.
            # The canvas offset is applied only when placing the image
            # via create_image() below.
            cx = rx + rw // 2
            cy = ry + rh // 2
            radius = max(rw, rh) // 2 + 8

            if self._state.is_auto_revealed(i):
                cv2.circle(frame_orig, (cx, cy), radius, self.COLOR_REVEAL, 3)
                cv2.circle(frame_mod,  (cx, cy), radius, self.COLOR_REVEAL, 3)
            elif self._state.is_found(i):
                cv2.circle(frame_orig, (cx, cy), radius, self.COLOR_FOUND, 3)
                cv2.circle(frame_mod,  (cx, cy), radius, self.COLOR_FOUND, 3)

        self._tk_orig = self._to_photoimage(frame_orig)
        self._tk_mod  = self._to_photoimage(frame_mod)

        self._canvas.delete("all")
        self._canvas.create_image(self._img_x_off, self._img_y_off,
                                  anchor=tk.NW, image=self._tk_orig)
        self._canvas.create_image(self.PANEL_WIDTH + 2 + self._img_x_off,
                                  self._img_y_off,
                                  anchor=tk.NW, image=self._tk_mod)
        self._canvas.create_line(self.PANEL_WIDTH, 0,
                                 self.PANEL_WIDTH, self.DISPLAY_HEIGHT,
                                 fill="#555", width=2)

    @staticmethod
    def _to_photoimage(bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _draw_canvas_overlay(self, line1: str, line2: str, color: str):
        """
        Draw a semi-transparent banner across the centre of the canvas
        with two lines of text. Used for Game Over and Well Done states.

        Tkinter has no native transparency, so we simulate it with a
        stipple-filled rectangle (gives a 50% see-through dark wash).
        """
        cw = self.PANEL_WIDTH * 2 + 4
        cy = self.DISPLAY_HEIGHT // 2

        banner_h = 90
        y1 = cy - banner_h // 2
        y2 = cy + banner_h // 2

        # Dark wash — stipple gives the semi-transparent effect
        self._canvas.create_rectangle(
            0, y1, cw, y2,
            fill="#000000", stipple="gray50", outline=""
        )
        # Solid backing strip for legibility
        self._canvas.create_rectangle(
            0, y1, cw, y2,
            fill="", outline=color, width=2
        )

        # Primary message
        self._canvas.create_text(
            cw // 2, cy - 18,
            text=line1,
            fill=color,
            font=("Courier New", 16, "bold"),
            anchor=tk.CENTER
        )
        # Secondary hint
        self._canvas.create_text(
            cw // 2, cy + 18,
            text=line2,
            fill="#cccccc",
            font=("Courier New", 11),
            anchor=tk.CENTER
        )

    # ── click handling ────────────────────────────────────────────────

    def _on_canvas_click(self, event):
        if self._clean_orig is None:
            return

        if self._state.is_locked:
            self._set_status("Round over - load a new image to continue.")
            return

        if event.x <= self.PANEL_WIDTH + 2:
            self._set_status("Click on the MODIFIED (right) image to find differences.")
            return

        # Translate click to image-local scaled coordinates.
        # Subtract the panel divider AND the centering offset so the result
        # is in the same coordinate space as self._regions (pure scaled coords,
        # origin = top-left of the image, not the panel).
        local_x = event.x - (self.PANEL_WIDTH + 2) - self._img_x_off
        local_y = event.y - self._img_y_off

        img_h, img_w = self._clean_mod.shape[:2]
        if local_x < 0 or local_y < 0 or local_x >= img_w or local_y >= img_h:
            self._set_status("Click inside the modified image area")
            return
    

        #matched = self._state.check_click(local_x, local_y, self._regions)
        matched_any = self._state.check_any_difference_click(local_x, local_y, self._regions)

        if matched_any >= 0 and self._state.is_found(matched_any):
            self._set_status("You already found that difference.")
            return

        matched = self._state.check_click(local_x, local_y, self._regions)
        
        if matched >= 0:
            self._state.register_correct_click(matched)
            self._refresh_canvas()
            self._update_hud()

            if self._state.all_found:
                self._draw_canvas_overlay(
                    "YOU FOUND ALL 5!",
                    f"Score: {self._state.cumulative_score}  –  Load a new image to continue",
                    "#00cc66"
                )
                self._set_status("All 5 found! Load a new image to keep playing.")
                self._btn_reveal.config(state=tk.DISABLED)
                messagebox.showinfo(
                    "Well done!",
                    f"You found all 5 differences!\n"
                    f"Cumulative score: {self._state.cumulative_score}"
                )
            else:
                self._set_status(
                    f"Correct! {self._state.remaining} difference(s) remaining.")
        else:
            self._state.register_mistake()
            self._update_hud()

            if self._state.is_locked:
                self._refresh_canvas()
                self._draw_canvas_overlay(
                    f"GAME OVER  –  {self._state.found_count} / 5 found",
                    "Load a new image or press Reveal All",
                    self.ACCENT
                )
                self._set_status(
                    f"3 mistakes reached! {self._state.found_count}/5 found. "
                    "Load a new image or press Reveal All.")
                messagebox.showwarning(
                    "Too many mistakes",
                    f"You have made 3 mistakes.\n"
                    f"Differences found: {self._state.found_count} / 5\n\n"
                    "Load a new image or press 'Reveal All' to see the answers."
                )
            else:
                left = GameState.MAX_MISTAKES - self._state.mistakes
                self._set_status(f"Wrong click - {left} mistake(s) remaining.")

    # ── reveal ────────────────────────────────────────────────────────

    def _reveal_all(self):
        if self._clean_orig is None:
            return
        self._state.reveal()
        self._refresh_canvas()
        self._draw_canvas_overlay(
            "DIFFERENCES REVEALED",
            "Load a new image to restart",
            "#4ea8de"
        )
        self._update_hud()
        self._btn_reveal.config(state=tk.DISABLED)
        self._set_status("Differences revealed in blue. Load a new image to restart.")

    # ── HUD ───────────────────────────────────────────────────────────

    def _update_hud(self):
        remaining = self._state.remaining if self._clean_orig is not None else "-"
        mistakes  = self._state.mistakes
        score     = self._state.cumulative_score
        self._lbl_remaining.config(text=f"Remaining: {remaining}")
        self._lbl_mistakes.config(
            text=f"Mistakes: {mistakes}/3",
            fg=self.ACCENT if mistakes >= GameState.MAX_MISTAKES else self.TEXT_COLOR
        )
        self._lbl_score.config(text=f"Score: {score}")

    def _set_status(self, msg: str):
        self._status_var.set(msg)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = SpotTheDiffApp()
    app.mainloop()
