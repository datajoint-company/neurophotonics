from neurophotonics.probe.probely import Probe, ProbeGroup


def design1(save=False, output="Design1.csv"):
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

    PG.probes[0].translate([-120, 0, 0])
    PG.probes[0].rotate_around(["br", "tr"], -45)

    PG.probes[2].translate([120, 0, 0])
    PG.probes[2].rotate_around(["bl", "tl"], 45)

    if save:
        df = PG.to_df()
        df.to_csv(output, index=False)

    return PG
