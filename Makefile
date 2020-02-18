PYTHON = python3.6

# ========== Linux (Debian) ==========

.PHONY: install venv update update-dev test clean remove


# ----- Install -----

install:
	sudo apt-get install --no-install-recommends -y \
		build-essential pkg-config \
		$(PYTHON) $(PYTHON)-dev $(PYTHON)-venv


# ----- Virtualenv -----

venv:
	@if [ ! -f "venv/bin/activate" ]; then $(PYTHON) -m venv venv ; fi;


# ----- Update -----

update:
	@echo "----- Updating requirements -----"
	@pip install --upgrade wheel pip setuptools
	@pip install --upgrade --requirement requirements.txt

update-dev: update
	@pip install --upgrade --requirement requirements-dev.txt


# ----- Tests -----

test: update
	@pytest iou.py -vv tests.py --flake8
