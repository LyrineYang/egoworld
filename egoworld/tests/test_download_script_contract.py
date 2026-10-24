from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


def test_download_models_script_matches_defaults() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "download_models.sh"
    content = script_path.read_text(encoding="utf-8")

    expected = [
        "sam2.1_hiera_small.pt",
        "sam2.1_hiera_s.yaml",
        "groundingdino_swint_ogc.pth",
        "GroundingDINO_SwinT_OGC.py",
    ]
    for name in expected:
        assert name in content


def test_downloader_paths_match_config_from_another_directory(tmp_path):
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / "repository with spaces"
    script = root / "egoworld/scripts/download_models.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(project / "scripts/download_models.sh", script)
    commands = tmp_path / "commands"
    commands.mkdir()
    stubs = {
        "wget": '#!/bin/bash\nwhile [[ "$1" != "-O" ]]; do shift; done\n"$TEST_REAL_PYTHON" -c "import sys; open(sys.argv[1], \'wb\').truncate(100000001)" "$2"\n',
        "python": '#!/bin/bash\ncat > /dev/null\npwd >> "$TEST_PYTHON_CWD"\n',
        "git": '#!/bin/bash\nfor destination; do :; done\nmkdir -p "$destination"\nprintf "exit 0\\n" > "$destination/fetch_demo_data.sh"\n',
    }
    for name, content in stubs.items():
        command = commands / name
        command.write_text(content)
        command.chmod(0o755)
    working = tmp_path / "outside"
    working.mkdir()
    cwd_log = tmp_path / "python-cwd.log"
    environment = {**os.environ, "PATH": str(commands) + os.pathsep + os.defpath,
                   "TEST_PYTHON_CWD": str(cwd_log), "TEST_REAL_PYTHON": sys.executable}
    subprocess.run(["bash", str(script)], cwd=working, env=environment, check=True, capture_output=True)
    params = json.loads((project / "configs/example.json").read_text())["operators"]["sam2"]["params"]
    for filename in (params["checkpoint"], params["config"], params["prompting"]["gd_checkpoint"], params["prompting"]["gd_config"]):
        assert (root / filename).is_file()
    assert set(cwd_log.read_text().splitlines()) == {str(root)}
