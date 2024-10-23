FROM jeanollion/training_dnn:tf-2.7.1
RUN pip install --upgrade h5py==3.11.0
COPY dataset_iterator /dataset_iterator
RUN pip install /dataset_iterator
COPY distnet2d /distnet2d
RUN pip install /distnet2d
COPY training_scripts/training_scripts/training_core.py /training_core.py
COPY training_scripts/training_scripts/training_distnet2d_seg.py /train.py
ENV NUMBA_NUM_THREADS=1
ENTRYPOINT ["/bin/bash"]