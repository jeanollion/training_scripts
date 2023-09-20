FROM jeanollion/training_dnn:tf-2.7.1
RUN pip install git+https://github.com/jeanollion/dataset_iterator.git
RUN pip install git+https://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@github.com/jeanollion/distnet2d.git
COPY training_core.py /training_core.py
COPY training_distnet2d.py /train.py
ENV HDF5_USE_FILE_LOCKING=FALSE
ENTRYPOINT ["/bin/bash"]