FROM jeanollion/training_dnn:tf-2.14.0
COPY dataset_iterator /dataset_iterator
RUN pip install -e /dataset_iterator
COPY distnet2d /distnet2d
RUN pip install -e /distnet2d
COPY training_scripts/training_scripts/training_core.py /training_core.py
COPY training_scripts/training_scripts/training_distnet2d.py /train.py
ENV HDF5_USE_FILE_LOCKING=FALSE
ENTRYPOINT ["/bin/bash"]