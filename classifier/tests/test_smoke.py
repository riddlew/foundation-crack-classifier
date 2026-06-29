import os

import foundation_crack_classifier


def test_classifier_test_environment_loads():
    assert foundation_crack_classifier is not None


def test_runtime_environment_paths_are_configured():
    expected_env = {
        "FCC_DATA_DIR": "/app/training_images",
        "FCC_MODEL_DIR": "/app/models",
        "FCC_REPORT_DIR": "/app/reports",
        "FCC_INPUT_DIR": "/app/input",
    }

    for name, value in expected_env.items():
        assert os.environ.get(name) == value
