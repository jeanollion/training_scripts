FROM jeanollion/training_dnn:tf-2.7.1
COPY dataset_iterator /dataset_iterator
RUN pip install /dataset_iterator
COPY pix_mclass /pix_mclass
RUN pip install /pix_mclass
COPY training_core.py /training_core.py
COPY training_pixmclass.py /train.py
ENV HDF5_USE_FILE_LOCKING=FALSE
ENTRYPOINT ["/bin/bash"]