# HIT137 Assignment 3 – Spot the Difference

A desktop "Spot the Difference" game built with Python, Tkinter, and OpenCV.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Python 3.9+ recommended. Tkinter is included in the standard library.

## How to Play

1. Click **Load Image** and choose any JPG, PNG, or BMP file.
2. Two images appear side-by-side. The **left** image is the original; the **right** is the modified copy.
3. Click on the **right image** where you think a difference is hidden.
   - A **red circle** appears on both images when you find one.
   - Wrong clicks are counted as **mistakes** (maximum 3 per image).
4. Find all 5 differences to complete the round and earn points (10 per difference).
5. Use **Reveal All** to show remaining differences in blue (remaining counter drops to 0).
6. Load a new image to keep playing – your score is cumulative.

## Architecture

| Class | Responsibility |
|---|---|
| `Alteration` | Abstract base class – declares the `apply()` interface |
| `ColourShiftAlteration` | Shifts HSV hue and brightness on a patch |
| `BlurAlteration` | Applies Gaussian blur to a patch |
| `PixelSwapAlteration` | Mirrors a patch horizontally |
| `ContrastBoostAlteration` | Amplifies local contrast on a patch |
| `TintOverlayAlteration` | Adds a semi-transparent colour tint to a patch |
| `ImageAlterer` | Clones the image; places 5 non-overlapping alterations via polymorphic dispatch |
| `GameState` | Game logic – found/unfound tracking, mistake counting, score |
| `SpotTheDiffApp` | Tkinter GUI – layout, event handling, rendering (inherits `tk.Tk`) |

## OOP Principles Demonstrated

**Encapsulation** – `GameState` exposes only controlled public methods and properties;
all internal lists and counters are private (`_differences_found`, `_mistakes`, etc.).

**Constructor** – every class initialises its own state via `__init__`.

**Inheritance** – `SpotTheDiffApp` extends `tk.Tk`. Each of the five alteration
classes (`ColourShiftAlteration`, `BlurAlteration`, etc.) inherits from `Alteration`.

**Polymorphism** – `Alteration` declares `apply(patch)` as an abstract interface
(raises `NotImplementedError`). Each subclass overrides it with its own behaviour.
`ImageAlterer._apply_patch()` calls `alteration.apply(patch)` without any `if/elif`
branching — Python resolves the correct implementation at runtime based on the
concrete subclass instance held in the pool. This is classical subtype polymorphism.

**Class interaction** – `SpotTheDiffApp` owns and coordinates both `ImageAlterer`
and `GameState`. `ImageAlterer` owns a pool of `Alteration` subclass instances.
