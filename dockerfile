# start with python base image
FROM python:3.9

# create directory in container for vetiver files
WORKDIR /vetiver

# copy  and install requirements
COPY vetiver_requirements.txt /vetiver/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /vetiver/requirements.txt

# copy app file
COPY app.py /vetiver/app/app.py

# expose port
EXPOSE 8080

# run vetiver API
CMD ["uvicorn", "app.app:api", "--host", "0.0.0.0", "--port", "8080"]