FROM jeanollion/training_dnn:tf-2.14.0
RUN pip install --upgrade h5py==3.9.0
RUN pip install psutil
COPY dataset_iterator /dataset_iterator
RUN pip install /dataset_iterator
COPY pix_mclass /pix_mclass
RUN pip install /pix_mclass
COPY training_scripts/training_scripts/training_core.py /training_core.py
COPY training_scripts/training_scripts/training_pixmclass.py /train.py
ENV NUMBA_NUM_THREADS=1
ENTRYPOINT ["/bin/bash"]