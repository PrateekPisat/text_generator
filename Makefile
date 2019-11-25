install-deps:
	pip install -r ./deps/requirements.in


run-LSTM:
	cd src && python LSTM.py && cd ..

run-LM:
	pushd src
	python text_generator.py
	popd src