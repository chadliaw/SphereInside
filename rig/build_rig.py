"""拍攝支架的參數化模型。

    pip install cadquery
    python rig/build_rig.py

輸出 rig/stl/ 下各零件的 STL（已擺成列印的方向）與 rig/rig.step（組裝後的樣子）。

座標（組裝後、仰角 0° 時）：原點是兩軸的交點，也就是鏡頭要對準的位置。
+z 朝上（水平旋轉軸），x 軸是俯仰軸，鏡頭朝 +y，手機螢幕朝 −y、機背貼在 y = 0 的平面上。
"""

import math
from pathlib import Path

import cadquery as cq

# ---- 五金 -----------------------------------------------------------------
BEARING_OD = 22.0  # 608 軸承
BEARING_W = 7.0
BEARING_FIT = 0.15  # 軸承座直徑的餘量
M8_HOLE = 8.4
M8_HEX = 13.4  # M8 六角頭對邊 13，加上餘量
M8_HEAD = 6.0  # 六角頭沉入的深度
RACE_CLEAR = 14.5  # 軸承座底部的開孔，只讓外圈靠在座上
M4_HOLE = 4.5
M4_HEX = 7.3
TRIPOD_HEX = 11.4  # 1/4"-20 螺帽對邊 11.11
TRIPOD_HOLE = 6.8
TRIPOD_FLOOR = 1.6  # 螺帽下方的厚度；腳架螺絲只凸出約 4.5 mm，不能太厚

# ---- 尺寸 -----------------------------------------------------------------
SWING = 150.0  # 鏡頭到手機底端的最大距離，決定立柱的高度
WIDTH = 34.0  # 臂與立柱在 y 方向的寬度
ARM_T = 14.0
UPRIGHT_T = 14.0
UPRIGHT_TOP = 16.0  # 立柱高出俯仰軸的量
BASE_R = 42.0
BASE_H = 30.0
BASE_LEDGE = 6.0
DETENTS = 12  # 水平每 30° 一格
DETENT_R = 2.0
DETENT_DEPTH = 1.0

POST_X0, POST_X1 = 52.0, 64.0  # 俯仰座：立柱內側面在 x = POST_X1
POST_Y0, POST_Y1 = -18.0, 10.0
BAR_T = 10.0  # 橫桿厚度（y 方向），正面在 y = 0，手機機背貼在這一面
BAR_Z0, BAR_Z1 = -62.0, -38.0
BAR_X0 = -90.0
SLOTS = ((-86.0, -22.0), (10.0, 46.0))  # 夾爪螺絲的滑槽
SLOT_Z = -50.0

JAW_W = 12.0
JAW_H = 15.0  # 夾爪高出橫桿正面的量（朝螢幕的方向）
JAW_WALL = 5.0  # 斜面開始的高度
JAW_LIP = 4.0  # 勾在橫桿上緣、防止夾爪轉動的凸緣

ARM_TOP = -(SWING + 3.0)
ARM_BOTTOM = ARM_TOP - ARM_T
ARM_X0 = -BASE_R
ARM_X1 = POST_X1 + UPRIGHT_T
BASE_BOTTOM = ARM_BOTTOM - BASE_H
TRIPOD_OFFSET = 24.0  # 腳架螺孔偏離水平軸的距離；軸心已被軸承佔用

OUT = Path(__file__).resolve().parent


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def cyl_z(x, y, z0, z1, d):
    return cq.Workplane("XY").circle(d / 2).extrude(z1 - z0).translate((x, y, z0))


def hex_z(x, y, z0, z1, across_flats):
    return cq.Workplane("XY").polygon(6, across_flats / math.cos(math.radians(30))).extrude(z1 - z0).translate((x, y, z0))


def along_x(solid):
    """把沿 z 軸建立的形狀轉成沿 x 軸：(x, y, z) → (z, y, −x)。"""
    return solid.rotate((0, 0, 0), (0, 1, 0), 90)


def along_y(solid):
    """把沿 z 軸建立的形狀轉成沿 y 軸：(x, y, z) → (x, z, −y)。"""
    return solid.rotate((0, 0, 0), (1, 0, 0), -90)


def base():
    """底座：鎖在腳架上，軸承從下方壓入，上緣一圈定位凹槽。"""
    top, seat = ARM_BOTTOM, ARM_BOTTOM - BASE_LEDGE
    part = cyl_z(0, 0, BASE_BOTTOM, top, 2 * BASE_R)
    part = part.cut(cyl_z(0, 0, seat - 1, top + 1, RACE_CLEAR))
    part = part.cut(cyl_z(0, 0, seat - BEARING_W - 1, seat, BEARING_OD + BEARING_FIT))
    # 軸承座以下放寬，軸承才不必一路緊配壓到底；也留給螺帽與套筒空間
    part = part.cut(cyl_z(0, 0, BASE_BOTTOM - 1, seat - BEARING_W - 1, BEARING_OD + 1.0))
    part = part.cut(hex_z(0, -TRIPOD_OFFSET, BASE_BOTTOM + TRIPOD_FLOOR, top + 1, TRIPOD_HEX))
    part = part.cut(cyl_z(0, -TRIPOD_OFFSET, BASE_BOTTOM - 1, top, TRIPOD_HOLE))
    for k in range(DETENTS):
        a = 2 * math.pi * k / DETENTS
        r = BASE_R + DETENT_R - DETENT_DEPTH
        part = part.cut(cyl_z(r * math.cos(a), r * math.sin(a), BASE_BOTTOM - 1, top + 1, 2 * DETENT_R))
    return part


def arm():
    """水平臂與立柱：繞鉛直軸旋轉，立柱頂端裝俯仰軸的軸承。"""
    h = WIDTH / 2
    part = box(ARM_X0, ARM_X1, -h, h, ARM_BOTTOM, ARM_TOP)
    part = part.union(box(POST_X1, ARM_X1, -h, h, ARM_BOTTOM, UPRIGHT_TOP))
    gusset = cq.Workplane("XY").polyline([(POST_X1, ARM_TOP), (POST_X1 - 25, ARM_TOP), (POST_X1, ARM_TOP + 25)]).close().extrude(WIDTH)
    part = part.union(gusset.rotate((0, 0, 0), (1, 0, 0), 90).translate((0, h, 0)))

    # 水平軸：螺栓頭沉在臂的上表面，不凸出，才不會擋到手機
    part = part.cut(cyl_z(0, 0, ARM_BOTTOM - 1, ARM_TOP + 1, M8_HOLE))
    part = part.cut(hex_z(0, 0, ARM_TOP - M8_HEAD, ARM_TOP + 1, M8_HEX))

    # 俯仰軸：軸承從立柱外側壓入
    seat = ARM_X1 - BEARING_W
    part = part.cut(along_x(cyl_z(0, 0, POST_X1 - 1, seat + 1, RACE_CLEAR)))
    part = part.cut(along_x(cyl_z(0, 0, seat, ARM_X1 + 1, BEARING_OD + BEARING_FIT)))

    # 定位彈片：垂在底座外緣，尖端的凸條卡進凹槽
    gap, t, length, bump = 0.5, 2.0, 24.0, 1.3
    x = BASE_R + gap
    part = part.union(box(x, x + t, -7, 7, ARM_BOTTOM - length, ARM_BOTTOM))
    ridge = cyl_z(x + DETENT_R - bump, 0, ARM_BOTTOM - length, ARM_BOTTOM - length + 8, 2 * DETENT_R)
    part = part.union(ridge.intersect(box(x - bump, x + t, -7, 7, ARM_BOTTOM - length, ARM_BOTTOM)))
    return part


def cradle():
    """俯仰座與橫桿：繞俯仰軸旋轉，手機機背貼在橫桿正面。"""
    part = box(POST_X0, POST_X1, POST_Y0, POST_Y1, BAR_Z0, UPRIGHT_TOP)
    part = part.union(box(BAR_X0, POST_X0, 0, BAR_T, BAR_Z0, BAR_Z1))
    part = part.cut(along_x(cyl_z(0, 0, POST_X0 - 1, POST_X1 + 1, M8_HOLE)))
    part = part.cut(along_x(hex_z(0, 0, POST_X0 - 1, POST_X0 + M8_HEAD, M8_HEX)))
    for x0, x1 in SLOTS:
        slot = box(x0, x1, -1, BAR_T + 1, SLOT_Z - M4_HOLE / 2, SLOT_Z + M4_HOLE / 2)
        for x in (x0, x1):
            slot = slot.union(along_y(cyl_z(x, -SLOT_Z, -1, BAR_T + 1, M4_HOLE)))
        part = part.cut(slot)
    # 對準記號（在背面）：鏡頭要在橫桿刻線的正上方、與俯仰座刻線等高
    part = part.cut(box(-0.5, 0.5, BAR_T - 0.6, BAR_T + 1, BAR_Z0 - 1, BAR_Z1 + 1))
    part = part.cut(box(POST_X0 - 1, POST_X1 + 1, POST_Y1 - 0.6, POST_Y1 + 1, -0.5, 0.5))
    return part


def jaw(x, side):
    """夾爪。x 是手機邊緣的位置，side 為 +1（靠俯仰座的一側）或 −1。

    朝手機的一面上半部是斜面：夾緊時把手機往橫桿壓，厚薄不同的手機都夾得住。
    """
    overhang = (JAW_H - 1 - JAW_WALL) * math.tan(math.radians(30))
    profile = [(0, 0), (JAW_W, 0), (JAW_W, -JAW_H), (-overhang, -JAW_H), (-overhang, -JAW_H + 1), (0, -JAW_WALL)]
    part = cq.Workplane("XY").polyline(profile).close().extrude(BAR_Z1 + JAW_LIP - BAR_Z0).translate((0, 0, BAR_Z0))
    part = part.union(box(0, JAW_W, 0, 6, BAR_Z1, BAR_Z1 + JAW_LIP))
    part = part.cut(along_y(cyl_z(JAW_W / 2, -SLOT_Z, -JAW_H - 1, 1, M4_HOLE)))
    part = part.cut(along_y(hex_z(JAW_W / 2, -SLOT_Z, -JAW_H - 1, -4, M4_HEX)))
    if side < 0:
        part = part.mirror("YZ")
    return part.translate((x, 0, 0))


def phone(to_post_edge=30.0, width=75.0, thickness=9.0, top=15.0):
    """檢查干涉用的手機外形。"""
    return box(to_post_edge - width, to_post_edge, -thickness, 0, -SWING + 5, top)


def volume(solid):
    return solid.val().Volume() if solid.vals() else 0.0


def check():
    """組裝後各零件不能互相穿透；俯仰到任何角度，手機與俯仰座都不能撞到臂與底座。"""
    a, b, c = arm(), base(), cradle()
    assert volume(a.intersect(b)) < 1e-6, "臂與底座干涉"
    assert volume(a.intersect(c)) < 1e-6, "臂與俯仰座干涉"
    moving = c.union(phone()).union(jaw(30, +1)).union(jaw(-45, -1))
    for angle in range(-90, 91, 15):
        tilted = moving.rotate((0, 0, 0), (1, 0, 0), angle)
        assert volume(tilted.intersect(a)) < 1e-6, f"俯仰 {angle}° 時撞到臂"
        assert volume(tilted.intersect(b)) < 1e-6, f"俯仰 {angle}° 時撞到底座"
    for e in (10, 40):
        assert volume(jaw(e, +1).intersect(c)) < 1e-6, "夾爪與俯仰座干涉"
    # 定位凸條在凹槽裡不能壓到底座，轉出凹槽時則要壓到（靠彈片變形）
    assert volume(a.intersect(b.rotate((0, 0, 0), (0, 0, 1), 15))) > 1, "定位凸條沒有碰到底座"


def main():
    check()
    stl = OUT / "stl"
    stl.mkdir(exist_ok=True)
    flip = ((1, 0, 0), 180)
    prints = {
        "base": (base(), flip),  # 上表面朝下列印，軸承座朝上
        "arm": (arm(), ((1, 0, 0), 90)),  # 側躺
        "cradle": (cradle(), ((1, 0, 0), -90)),  # 背面朝下
        "jaw_a": (jaw(0, +1), flip),  # 凸緣朝下
        "jaw_b": (jaw(0, -1), flip),
    }
    for name, (part, (axis, angle)) in prints.items():
        part = part.rotate((0, 0, 0), axis, angle)
        box_ = part.val().BoundingBox()
        part = part.translate((-box_.xmin, -box_.ymin, -box_.zmin))
        cq.exporters.export(part, str(stl / f"{name}.stl"), tolerance=0.02, angularTolerance=0.1)
        print(f"{name:7s} {box_.xlen:6.1f} × {box_.ylen:6.1f} × {box_.zlen:6.1f} mm")

    assembly = cq.Assembly()
    for name, part in [("base", base()), ("arm", arm()), ("cradle", cradle()), ("jaw_a", jaw(30, +1)), ("jaw_b", jaw(-45, -1))]:
        assembly.add(part, name=name)
    assembly.save(str(OUT / "rig.step"))
    print(f"已輸出 {stl} 與 {OUT / 'rig.step'}")


if __name__ == "__main__":
    main()
