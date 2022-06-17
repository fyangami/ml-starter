VERSION=0.0.1

build:
	python -m build -w
	pip install dist/jaxnn-$(VERSION)-py3-none-any.whl --force-reinstall

clean:
	@rm -fr build
	@rm -fr dist
	@rm -fr jaxnn.egg-info
	@rm -fr ./*/__pycache__
	@echo 'Done.'

all: build clean

.PHONY: build clean

