FROM jeanollion/training_dnn:tf-2.7.1
RUN pip install git+https://github.com/jeanollion/dataset_iterator.git
RUN pip install git+https://github.com/jeanollion/pix_mclass.git
COPY training_core.py /training_core.py
COPY training_pixmclass.py /train.py
ENTRYPOINT ["/bin/bash"]