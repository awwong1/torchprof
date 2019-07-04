build:
	python3 setup.py sdist bdist_wheel

clean:
	rm -rf build dist torchprof.egg-info

test:
	python3 -m unittest discover
