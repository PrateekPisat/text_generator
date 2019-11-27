install-deps:
	pip install -r ./deps/requirements.in


build-LSTM:
	cd src &&  python LSTM.py && cd ..

run-LSTM:
	cd src &&  python text_generator.py && cd ..

run-LM:
	pushd src
	python text_generator.py
	popd src