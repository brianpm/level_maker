from pathlib import Path
import json
from level_maker import makelev

def test_example_data_exists():
    assert Path("../level_maker/cam3_levels_input.json").is_file()


def test_level_output001():
    with open("../level_maker/cam3_levels_input.json") as f:
        data = json.load(f)
    am, bm, ai, bi, lev, ilev = make_levels(data['dps'], data['purmax'], data['regions'], print_out=False)
    assert am.shape == bm.shape