"""Create stable, high-contrast image fixtures for the LymphImagen PoC."""

from pathlib import Path

from PIL import Image, ImageDraw


OUT = Path(__file__).parent
SKIN = (205, 145, 95)
SKIN_ALT = (190, 125, 80)
COIN = (209, 169, 52)
COIN_EDGE = (129, 94, 18)


def canvas(width=640, height=420, background=(246, 248, 250)):
    return Image.new("RGB", (width, height), background)


def draw_limb(draw, x, y, width, height, kind="arm", color=SKIN):
    """Draw a single separated, front-facing limb silhouette."""
    if kind == "leg":
        points = [
            (x + width // 5, y), (x + width * 4 // 5, y),
            (x + width, y + height // 4), (x + width * 3 // 4, y + height),
            (x + width // 4, y + height), (x, y + height // 4),
        ]
    else:
        points = [
            (x + width // 4, y), (x + width * 3 // 4, y),
            (x + width, y + height // 5), (x + width * 3 // 4, y + height),
            (x + width // 4, y + height), (x, y + height // 5),
        ]
    draw.polygon(points, fill=color)
    draw.ellipse((x + width // 4, y - 3, x + width * 3 // 4, y + 18), fill=color)
    draw.ellipse((x + width // 4, y + height - 18, x + width * 3 // 4, y + height + 3), fill=color)


def draw_coin(draw, cx, cy, diameter):
    radius = diameter // 2
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=COIN, outline=COIN_EDGE, width=3)


def save(image, name):
    image.save(OUT / name, format="PNG", optimize=False)


def main():
    image = canvas()
    d = ImageDraw.Draw(image)
    draw_limb(d, 105, 65, 80, 285, "arm", SKIN)
    draw_limb(d, 440, 85, 68, 245, "arm", SKIN_ALT)
    save(image, "arms_01.png")

    image = canvas(background=(250, 247, 242))
    d = ImageDraw.Draw(image)
    draw_limb(d, 100, 45, 94, 320, "arm", SKIN_ALT)
    draw_limb(d, 445, 85, 80, 245, "arm", SKIN)
    save(image, "arms_02.png")

    image = canvas(height=500)
    d = ImageDraw.Draw(image)
    draw_limb(d, 90, 50, 130, 370, "leg", SKIN)
    draw_limb(d, 430, 85, 105, 315, "leg", SKIN_ALT)
    save(image, "legs_01.png")

    image = canvas(height=500, background=(247, 249, 246))
    d = ImageDraw.Draw(image)
    draw_limb(d, 95, 85, 112, 315, "leg", SKIN_ALT)
    draw_limb(d, 425, 45, 135, 375, "leg", SKIN)
    save(image, "legs_02.png")

    image = canvas()
    d = ImageDraw.Draw(image)
    draw_limb(d, 95, 55, 76, 300, "arm", SKIN)
    draw_coin(d, 505, 315, 48)
    save(image, "arm_coin_01.png")

    image = canvas(background=(248, 246, 241))
    d = ImageDraw.Draw(image)
    draw_limb(d, 110, 70, 90, 265, "arm", SKIN_ALT)
    draw_coin(d, 510, 310, 56)
    save(image, "arm_coin_02.png")

    image = canvas(height=500)
    d = ImageDraw.Draw(image)
    draw_limb(d, 100, 45, 128, 375, "leg", SKIN)
    draw_coin(d, 510, 420, 52)
    save(image, "leg_coin_01.png")

    image = canvas(height=500, background=(249, 247, 243))
    d = ImageDraw.Draw(image)
    draw_limb(d, 115, 70, 110, 320, "leg", SKIN_ALT)
    draw_coin(d, 510, 420, 44)
    save(image, "leg_coin_02.png")


if __name__ == "__main__":
    main()
