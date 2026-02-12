FROM training_dnn:tf-2.18.0
RUN pip install tf-keras==2.18.0
COPY dataset_iterator /dataset_iterator
RUN pip install /dataset_iterator
COPY distnet2d /distnet2d
RUN pip install /distnet2d
COPY training_scripts/training_scripts/training_core.py /training_core.py
COPY training_scripts/training_scripts/training_distnet2d.py /train.py
RUN chmod a+r /training_core.py
RUN chmod a+r /train.py
ENV NUMBA_NUM_THREADS=1
ENV TF_USE_LEGACY_KERAS=1
ENTRYPOINT ["/bin/bash"]