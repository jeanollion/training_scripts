FROM training:tf-2.7.1
RUN pip install git+https://github.com/jeanollion/dataset_iterator.git
RUN pip install git+https://github.com/jeanollion/pix_mclass.git
COPY training_core.py .
COPY training_pixmclass.py .

ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["python", "training_pixmclass.py", "/data"]
