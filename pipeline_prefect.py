import argparse
import os
import subprocess
import sys
from prefect import flow, get_run_logger, task

PYTHON = sys.executable
TARGETS = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
REPO_URL = "https://github.com/YasmineMansour/ml-pipeline-prefect.git"
REPO_DIR = os.path.expanduser("~/ml_project_git")


def _run(cmd):
    """Exécute une commande et échoue si le code retour n'est pas 0."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Échec ({result.returncode}) : {' '.join(cmd)}")


@task(name="sync-git-repo")
def sync_git_repo():
    if not os.path.exists(os.path.join(REPO_DIR, ".git")):
        _run(["git", "clone", REPO_URL, REPO_DIR])
    else:
        _run(["git", "-C", REPO_DIR, "pull"])


@task(name="install-dependencies")
def install_dependencies():
    _run([PYTHON, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format-code")
def format_code():
    _run([PYTHON, "-m", "black", *TARGETS])


@task(name="quality-code")
def quality_code():
    _run([PYTHON, "-m", "flake8", "--max-line-length=88", *TARGETS])


@task(name="security-code")
def security_code():
    _run([PYTHON, "-m", "bandit", "-q", *TARGETS])


@task(name="unit-tests")
def unit_tests():
    _run([PYTHON, "-m", "pytest", "tests", "-q"])


@task(name="prepare-data")
def prepare_data():
    _run([PYTHON, "main.py", "--prepare"])


@task(name="train-model")
def train_model():
    _run([PYTHON, "main.py", "--train"])


@task(name="evaluate-model")
def evaluate_model():
    _run([PYTHON, "main.py", "--evaluate"])


@flow(name="code", log_prints=True)
def flow_code():
    format_code()
    quality_code()
    security_code()
    unit_tests()


@flow(name="train", log_prints=True)
def flow_train():
    prepare_data()
    train_model()


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    evaluate_model()


@flow(name="all", log_prints=True)
def flow_all():
    sync_git_repo()
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()
    prepare_data()
    train_model()
    evaluate_model()


FLOWS = {
    "code": flow_code,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
    "all": flow_all,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
