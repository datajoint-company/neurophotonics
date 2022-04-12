from create_geom import ShankGroup

PM1 = ShankGroup(3, [1200, 100, 1300], [10, 10], [10, 10])

PM1.shank[0].rotate("z", 45)
PM1.shank[2].rotate("z", -45)

PM1.shank[0].translate([200, 100, 0])
PM1.shank[2].translate([-200, 100, 0])

PM1.shank[0].translate([0, -64.64466094, 0])
PM1.shank[2].translate([0, -64.64466094, 0])