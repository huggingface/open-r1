.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := src scripts

style:
	black --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort $(check_dirs) setup.py

quality:
	black --check --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort --check-only $(check_dirs) setup.py
	flake8 --max-line-length 119 $(check_dirs) setup.py


# Release stuff

pre-release:
	python src/open_r1/release.py

pre-patch:
	python src/open_r1/release.py --patch

post-release:
	python src/open_r1/release.py --post_release

post-patch:
	python src/open_r1/release.py --post_release --patch

wheels:
	python setup.py bdist_wheel && python setup.py sdist

wheels_clean:
	rm -rf build && rm -rf dist

pypi_upload:
	python -m pip install twine
	twine upload dist/* -r pypi

pypi_test_upload:
	python -m pip install twine
	twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
