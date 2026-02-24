from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from commurenew_agent.image_generation import edit_image_with_gemini_nanobanana


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    source = Path("inputs/siteImgs/非机动车停车位-1.jpg")
    output = Path("output/test_nanobanana_non_motor_parking.png")
    prompt = (
        "在不改变原始构图和拍摄角度的前提下，对该住区非机动车停车位进行更新改造："
        "增加有序停车线、规范停车棚、补充安全充电设施与导视标识，"
        "保持真实可落地的中国城市住区风格。"
    )

    if not source.exists():
        raise FileNotFoundError(f"测试图片不存在: {source}")

    out = edit_image_with_gemini_nanobanana(
        prompt=prompt,
        source_image_path=source,
        output_path=output,
        model="gemini-3-pro-image-preview",
    )
    print(f"Generated image saved to: {out}")


if __name__ == "__main__":
    main()
