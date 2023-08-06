# Create a virtual environment and install required dependencies
initialize:
    #!/usr/bin/env bash
    set -euxo pipefail
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip setuptools wheel pip-tools

# Generate requirements.txt, requirements.workflows.txt, and requirements.dev.txt
pin-deps: pin-src-deps pin-workflows-deps pin-dev-deps

# Generate requirements.txt from requirements.in
pin-src-deps:
    #!/usr/bin/env bash
    set -euxo pipefail
    source .venv/bin/activate
    pip-compile \
        --resolver=backtracking \
        --generate-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        --no-emit-index-url \
        --output-file requirements/requirements.txt \
        requirements/requirements.in

# Generate requirements.workflows.txt from requirements.workflows.in
pin-workflows-deps: pin-src-deps
    #!/usr/bin/env bash
    set -euxo pipefail
    source .venv/bin/activate
    pip-compile \
        --resolver=backtracking \
        --generate-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        --no-emit-index-url \
        --output-file requirements/requirements.workflows.txt \
        requirements/requirements.workflows.in

# Generate requirements.dev.txt from requirements.dev.in
pin-dev-deps: pin-src-deps pin-workflows-deps
    #!/usr/bin/env bash
    set -euxo pipefail
    source .venv/bin/activate
    pip-compile \
        --resolver=backtracking \
        --generate-hashes \
        --no-emit-index-url \
        --output-file requirements/requirements.dev.txt \
        requirements/requirements.dev.in

# Upgrade all dependencies in requirements.txt and requirements.dev.txt
upgrade-deps: upgrade-src-deps upgrade-workflows-deps upgrade-dev-deps

# Upgrade all dependencies in requirements.txt
upgrade-src-deps:
    #!/usr/bin/env bash
    set -euxo pipefail
    source .venv/bin/activate
    pip-compile \
        --resolver=backtracking \
        --generate-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        --no-emit-index-url \
        --upgrade \
        --output-file requirements/requirements.txt \
        requirements/requirements.in

# Upgrade all dependencies in requirements.workflows.txt
upgrade-workflows-deps: upgrade-src-deps
    #!/usr/bin/env bash
    set -euxo pipefail
    source .venv/bin/activate
    pip-compile \
        --resolver=backtracking \
        --generate-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        --no-emit-index-url \
        --upgrade \
        --output-file requirements/requirements.workflows.txt \
        requirements/requirements.workflows.in

# Upgrade all dependencies in requirements.dev.txt
upgrade-dev-deps: upgrade-src-deps upgrade-workflows-deps
    #!/usr/bin/env bash
    set -euxo pipefail
    source .venv/bin/activate
    pip-compile \
        --resolver=backtracking \
        --generate-hashes \
        --no-emit-index-url \
        --upgrade \
        --output-file requirements/requirements.dev.txt \
        requirements/requirements.dev.in

# Install all dependencies from requirements.txt and requirements.dev.txt into the virtual environment
sync-deps:
    #!/usr/bin/env bash
    set -euxo pipefail
    source .venv/bin/activate
    pip-sync \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        requirements/requirements.txt \
        requirements/requirements.workflows.txt \
        requirements/requirements.dev.txt
