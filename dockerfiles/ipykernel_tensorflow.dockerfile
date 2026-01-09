FROM jeanollion/training_dnn:tf-2.14.0
RUN pip install matplotlib==3.8.2
RUN pip install ipython ipykernel
ENV NUMBA_NUM_THREADS=1