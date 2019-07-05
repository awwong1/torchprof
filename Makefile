build:
	python3 setup.py sdist bdist_wheel

clean:
	rm -rf build dist torchprof.egg-info

upload:
	twine upload dist/*

test:
	python3 -m unittest discover
