from pathlib import Path
from io import BytesIO
import html

import cv2
import numpy as np
from PIL import Image
from rembg import remove


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_IMAGE = ROOT / "assets" / "portrait.jpg"
OUTPUT_IMAGE = ROOT / "assets" / "generated" / "portrait.svg"

COLS = 90

FONT_SIZE = 12.9
CHAR_W = FONT_SIZE * 0.600

RAMP = " .`:-=+*cs#%@"

DISPLAY_WIDTH = 460

ROW_DELAY = 0.09
ROW_DURATION = 0.75

# One colour for the whole portrait.
TEXT_COLOR = "#111111"


# ============================================================
# Background removal
# ============================================================

def remove_background(image_path: Path) -> np.ndarray:
    """
    Remove the background using rembg.

    Returns:
        BGR image with the background converted to white.
    """

    with open(image_path, "rb") as f:
        source = f.read()

    result = remove(source)

    image = Image.open(BytesIO(result)).convert("RGBA")

    rgba = np.array(image)

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    # White background.
    background = np.full_like(rgb, 255)

    alpha_float = alpha[:, :, None].astype(np.float32) / 255.0

    composited = (
        rgb.astype(np.float32) * alpha_float
        + background.astype(np.float32) * (1 - alpha_float)
    )

    return cv2.cvtColor(
        composited.astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )


# ============================================================
# Image processing
# ============================================================

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Convert the image into a high-contrast grayscale image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth skin while preserving edges.
    gray = cv2.bilateralFilter(
        gray,
        d=9,
        sigmaColor=75,
        sigmaSpace=75,
    )

    # Local contrast enhancement.
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    # Darkening curve.
    #
    # This makes facial features such as:
    # - glasses
    # - eyebrows
    # - lips
    # - nose shadows
    #
    # survive the ASCII conversion.
    normalized = gray.astype(np.float32) / 255.0

    darkened = np.power(normalized, 1.7)

    gray = np.clip(
        darkened * 255,
        0,
        255,
    ).astype(np.uint8)

    return gray


# ============================================================
# Crop and resize
# ============================================================

def prepare_image(gray: np.ndarray) -> np.ndarray:
    """
    Crop the image to the subject and resize it
    to the desired ASCII width.
    """

    height, width = gray.shape

    # Preserve the whole image if no better crop is known.
    crop = gray

    crop_height, crop_width = crop.shape

    # ASCII characters are roughly twice as tall
    # as they are wide.
    aspect_ratio = crop_height / crop_width

    rows = max(
        1,
        int(COLS * aspect_ratio * 0.48),
    )

    resized = cv2.resize(
        crop,
        (COLS, rows),
        interpolation=cv2.INTER_AREA,
    )

    return resized


# ============================================================
# ASCII conversion
# ============================================================

def image_to_ascii(image: np.ndarray) -> list[str]:
    """
    Convert grayscale pixels to ASCII characters.
    """

    ramp_length = len(RAMP)

    lines = []

    for row in image:
        line = ""

        for pixel in row:

            # White = blank.
            # Black = dense character.
            brightness = pixel / 255.0

            index = int(
                (1.0 - brightness)
                * (ramp_length - 1)
            )

            index = max(
                0,
                min(index, ramp_length - 1),
            )

            line += RAMP[index]

        lines.append(line)

    return lines


# ============================================================
# SVG helpers
# ============================================================

def svg_escape(text: str) -> str:
    """
    Safely escape text for SVG.
    """

    return html.escape(
        text,
        quote=False,
    )


# ============================================================
# SVG generation
# ============================================================

def create_svg(lines: list[str]) -> str:
    """
    Create an animated SVG.

    Each row is revealed from left to right.
    """

    rows = len(lines)

    width = COLS * CHAR_W
    line_height = FONT_SIZE * 1.05
    height = rows * line_height + FONT_SIZE

    svg = []

    # --------------------------------------------------------
    # SVG root
    # --------------------------------------------------------

    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:.2f} {height:.2f}" '
        f'width="{DISPLAY_WIDTH}" '
        f'role="img" '
        f'aria-label="Animated ASCII portrait">'
    )

    # --------------------------------------------------------
    # Embedded styling
    # --------------------------------------------------------

    svg.append(
        "<style>"
        ".ascii { "
        "font-family: 'JetBrains Mono', 'Courier New', monospace; "
        f"font-size: {FONT_SIZE}px; "
        "font-weight: 400; "
        f"fill: {TEXT_COLOR}; "
        "white-space: pre; "
        "}"
        "</style>"
    )

    # --------------------------------------------------------
    # Definitions
    # --------------------------------------------------------

    svg.append("<defs>")

    # Create a clip path for every row.
    for i in range(rows):

        y = i * line_height

        begin = i * ROW_DELAY

        svg.append(
            f'<clipPath id="row{i}">'
        )

        svg.append(
            f'<rect '
            f'x="0" '
            f'y="{y:.2f}" '
            f'width="0" '
            f'height="{FONT_SIZE * 1.5:.2f}">'
        )

        svg.append(
            f'<animate '
            f'attributeName="width" '
            f'from="0" '
            f'to="{width:.2f}" '
            f'dur="{ROW_DURATION}s" '
            f'begin="{begin:.2f}s" '
            f'fill="freeze"/>'
        )

        svg.append("</rect>")

        svg.append("</clipPath>")

    svg.append("</defs>")

    # --------------------------------------------------------
    # Render each ASCII row
    # --------------------------------------------------------

    for i, line in enumerate(lines):

        y = i * line_height

        text = svg_escape(line)

        svg.append(
            f'<g clip-path="url(#row{i})">'
        )

        svg.append(
            f'<text '
            f'class="ascii" '
            f'x="0" '
            f'y="{y + FONT_SIZE:.2f}" '
            f'xml:space="preserve">'
            f'{text}'
            f'</text>'
        )

        svg.append("</g>")

    # --------------------------------------------------------
    # Close SVG
    # --------------------------------------------------------

    svg.append("</svg>")

    return "\n".join(svg)


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("======================================")
    print(" ASCII Portrait Generator")
    print("======================================")

    print(f"Input : {INPUT_IMAGE}")
    print(f"Output: {OUTPUT_IMAGE}")

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Input image not found: {INPUT_IMAGE}"
        )

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    print("\n[1/5] Removing background...")

    image = remove_background(INPUT_IMAGE)

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    print("[2/5] Processing image...")

    gray = process_image(image)

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    print("[3/5] Preparing image...")

    prepared = prepare_image(gray)

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    print("[4/5] Converting to ASCII...")

    lines = image_to_ascii(prepared)

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    print("[5/5] Creating animated SVG...")

    svg = create_svg(lines)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_IMAGE.write_text(
        svg,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("======================================")
    print(" Done!")
    print("======================================")

    print(f"Generated: {OUTPUT_IMAGE}")
    print(f"Columns  : {COLS}")
    print(f"Rows     : {len(lines)}")
    print(f"SVG size : {len(svg):,} bytes")

    print("======================================")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()

