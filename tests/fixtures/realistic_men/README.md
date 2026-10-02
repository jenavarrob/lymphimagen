# Realistic image fixtures

These eight images are synthetic photorealistic fixtures generated from the user-provided clinical-photo reference (figure 3 style). They are for development and UI/manual testing, not clinical evidence or ground truth.

- `arms_01.png`, `arms_02.png`: two separate arms/forearms
- `legs_01.png`, `legs_02.png`: two separate legs
- `arm_coin_01.png`, `arm_coin_02.png`: one arm/forearm and a visible 1 EUR coin
- `leg_coin_01.png`, `leg_coin_02.png`: one leg and a visible 1 EUR coin

Because photorealistic fixtures contain shadows, gradients, hair, fingers, feet, and floor/wall edges, they are kept separate from the deterministic segmentation fixtures. Use them for visual/manual regression and use `tests/fixtures/*.png` for stable algorithm tests until the segmentation model is upgraded.
