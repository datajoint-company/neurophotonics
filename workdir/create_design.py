from neurophotonics.probe.probely import Probe, ProbeGroup


def design101(save=False, output="Design_v101.csv"):
    # Design 1 - 30 um separation - 75 degrees

    # Create 3 Probes at 0, 0, 0
    # 360 e-pixels
    P1 = Probe(
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=5,
        e_box_horizontal_margin=15,
        n_d_box=[0, 0],
        d_box_length=0,
        d_box_sep=0,
        d_box_vertical_margin=0,
        d_box_horizontal_margin=0,
        name="P1",
    )

    # 4036 d-pixels
    P2 = Probe(
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=5,
        e_box_horizontal_margin=15,
        n_d_box=[22, 240],
        d_box_length=5,
        d_box_sep=0,
        d_box_vertical_margin=5,
        d_box_horizontal_margin=5,
        name="P2",
    )

    P2.e_pixels = []

    # 360 e-pixels
    P3 = Probe(
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=5,
        e_box_horizontal_margin=15,
        n_d_box=[0, 0],
        d_box_length=0,
        d_box_sep=0,
        d_box_vertical_margin=0,
        d_box_horizontal_margin=0,
        name="P3",
    )

    PG = ProbeGroup([P1, P2, P3])

    # Position the Probes
    PG.probes[0].translate([-150, 0, 0])
    PG.probes[0].rotate_around(["br", "tr"], -75)

    PG.probes[2].translate([150, 0, 0])
    PG.probes[2].rotate_around(["bl", "tl"], 75)

    if save:
        df = PG.to_df()
        df.to_csv(output, index=False)

    return PG


def design102(save=False, output="Design_v102.csv"):
    # Design 1 - 60 um separation - 75 degrees

    # Create 3 Probes at 0, 0, 0
    # 360 e-pixels
    P1 = Probe(
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=5,
        e_box_horizontal_margin=15,
        n_d_box=[0, 0],
        d_box_length=0,
        d_box_sep=0,
        d_box_vertical_margin=0,
        d_box_horizontal_margin=0,
        name="P1",
    )

    # 4036 d-pixels
    P2 = Probe(
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=5,
        e_box_horizontal_margin=15,
        n_d_box=[22, 240],
        d_box_length=5,
        d_box_sep=0,
        d_box_vertical_margin=5,
        d_box_horizontal_margin=5,
        name="P2",
    )

    P2.e_pixels = []

    # 360 e-pixels
    P3 = Probe(
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=5,
        e_box_horizontal_margin=15,
        n_d_box=[0, 0],
        d_box_length=0,
        d_box_sep=0,
        d_box_vertical_margin=0,
        d_box_horizontal_margin=0,
        name="P3",
    )

    PG = ProbeGroup([P1, P2, P3])

    # Position the Probes
    PG.probes[0].translate([-180, 0, 0])
    PG.probes[0].rotate_around(["br", "tr"], -75)

    PG.probes[2].translate([180, 0, 0])
    PG.probes[2].rotate_around(["bl", "tl"], 75)

    if save:
        df = PG.to_df()
        df.to_csv(output, index=False)

    return PG


def design103(save=False, output="Design_v103.csv"):
    # Design 1 - 120 um separation - 75 degrees

    # Create 3 Probes at 0, 0, 0
    # 360 e-pixels
    P1 = Probe(
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=5,
        e_box_horizontal_margin=15,
        n_d_box=[0, 0],
        d_box_length=0,
        d_box_sep=0,
        d_box_vertical_margin=0,
        d_box_horizontal_margin=0,
        name="P1",
    )

    # 4036 d-pixels
    P2 = Probe(
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=5,
        e_box_horizontal_margin=15,
        n_d_box=[22, 240],
        d_box_length=5,
        d_box_sep=0,
        d_box_vertical_margin=5,
        d_box_horizontal_margin=5,
        name="P2",
    )

    P2.e_pixels = []

    # 360 e-pixels
    P3 = Probe(
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=5,
        e_box_horizontal_margin=15,
        n_d_box=[0, 0],
        d_box_length=0,
        d_box_sep=0,
        d_box_vertical_margin=0,
        d_box_horizontal_margin=0,
        name="P3",
    )

    PG = ProbeGroup([P1, P2, P3])

    # Position the Probes
    PG.probes[0].translate([-240, 0, 0])
    PG.probes[0].rotate_around(["br", "tr"], -75)

    PG.probes[2].translate([240, 0, 0])
    PG.probes[2].rotate_around(["bl", "tl"], 75)

    if save:
        df = PG.to_df()
        df.to_csv(output, index=False)

    return PG


def design201(save=False, output="Design_v201.csv"):
    # Design 2
    # Create 10 probes at 0, 0, 0
    PG = ProbeGroup(
        [
            Probe(
                probe_dimensions=[1200, 150, 1300],
                n_e_box=[5, 60],
                e_box_length=10,
                e_box_sep=10,
                e_box_vertical_margin=5,
                e_box_horizontal_margin=30,
                n_d_box=[22, 240],
                d_box_length=5,
                d_box_sep=0,
                d_box_vertical_margin=0,
                d_box_horizontal_margin=25,
                name="P" + str(i),
            )
            for i in range(10)
        ]
    )

    # Position the Probes
    for i, probe in enumerate(PG.probes):
        if not i % 2:
            probe.rotate("z", 180)  # Around the origin (0, 0, 0)
            probe.translate([-150.0 * len(PG.probes) / 2 + 0.5 + i * 150.0, 75, 0])
        else:
            probe.translate([-150.0 * len(PG.probes) / 2 + 0.5 + i * 150.0, -75, 0])

    if save:
        df = PG.to_df()
        df.to_csv(output, index=False)

    return PG


def design202(save=False, output="Design_v202.csv"):
    # Design 2
    # Create 10 probes at 0, 0, 0
    PG = ProbeGroup(
        [
            Probe(
                probe_dimensions=[1200, 150, 1300],
                n_e_box=[5, 60],
                e_box_length=10,
                e_box_sep=10,
                e_box_vertical_margin=5,
                e_box_horizontal_margin=30,
                n_d_box=[22, 240],
                d_box_length=5,
                d_box_sep=0,
                d_box_vertical_margin=0,
                d_box_horizontal_margin=25,
                name="P" + str(i),
            )
            for i in range(10)
        ]
    )

    def rotate_epixels(probe):
        # Rotate detectors in each column with the given angles

        for i, epixel in enumerate(probe.e_pixels):
            if i < 60:
                angle = -45.0
                epixel.rotate_normal("z", angle)
            elif i >= 60 and i < 120:
                angle = 45
                epixel.rotate_normal("z", angle)
            elif i >= 120 and i < 180:
                if i % 2:
                    angle = -45
                else:
                    angle = 45
                epixel.rotate_normal("x", angle)
            elif i >= 180 and i < 240:
                angle = -45
                epixel.rotate_normal("z", angle)
            elif i >= 240:
                angle = 45
                epixel.rotate_normal("z", angle)

    def position(PG):
        # Position the Probes
        for i, probe in enumerate(PG.probes):
            if not i % 2:  # evens
                probe.rotate("z", 180)  # Around the origin (0, 0, 0)
                # Emission beams of the end-probes will be directed to the opposite shank.
                if i == 0:
                    [e_pixel.rotate_normal("z", 45) for e_pixel in probe.e_pixels]
                else:
                    rotate_epixels(probe)
                probe.translate([-150.0 * len(PG.probes) / 2 + 0.5 + i * 150.0, 75, 0])

            else:  # odds
                # Emission beams of the end-probes will be directed to the opposite shank.
                if i == 9:
                    [e_pixel.rotate_normal("z", 45) for e_pixel in probe.e_pixels]
                else:
                    rotate_epixels(probe)
                probe.translate([-150.0 * len(PG.probes) / 2 + 0.5 + i * 150.0, -75, 0])

    position(PG)

    if save:
        df = PG.to_df()
        df.to_csv(output, index=False)

    return PG
