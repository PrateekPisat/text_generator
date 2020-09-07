install-deps:
	poetry install


build-LSTM:
	cd src && \
	python text_generator.py --build_model lincoln 100 \
	&& cd ..

run-LSTM:
	cd src && \
	python text_generator.py --generate_sent \
	lincoln \
	500 \
	"as a subject for the remarks of the evening, the perpetuation of our  political institutions, is sel" \
	/home/prateek/text_generator/src/model_artifacts/lincoln/lincoln-plus-25-50-0.3981.hdf5 \
	&& cd ..

get_similarity:
	cd src && \
	python text_generator.py --similarity \
	lincoln \
	"Mr. President and Gentlemen of the Senate of the State of New-Jersey: I am very grateful to you for the honorable reception of which I have been the object. I cannot but remember the place that New-Jersey holds in our early history. In the early Revolutionary struggle, few of the States among the old Thirteen had more of the battle-fields of the country within their limits than old New-Jersey. May I be pardoned if, upon this occasion, I mention that away back in my childhood, the earliest days of my being able to read, I got hold of a small book, such a " \
	/home/prateek/text_generator/src/model_artifacts/lincoln/lincoln-plus-25-50-0.3981.hdf5 \
	&& cd ..

run-trigram-generate:
	cd src && \
	python text_generator.py --trigram-generate lincoln 10 \
	&& cd ..

run-trigram-similarity:
	cd src && \
	python text_generator.py --trigram-similarity lincoln /home/prateek/text_generator/src/test/test01.txt \
	&& cd ..