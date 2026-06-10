PYTHON = venv/bin/python

install:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

lint:
	venv/bin/flake8 src tests scripts

test:
	venv/bin/pytest -q

train:
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 1 \
		-save models/1sess.txt
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 10 \
		-save models/10sess.txt
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 100 \
		-save models/100sess.txt
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 1000 \
		-save models/1000sess.txt
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 10000 \
		-save models/10000sess.txt
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 50000 \
		-save models/50000sess.txt
	cp models/50000sess.txt models/best.txt

.PHONY: install lint test train
