FROM python:3.9

ADD . /app
WORKDIR "/app"
RUN pip3 install --no-cache-dir -r requirements.txt -i https://mirror.baidu.com/pypi/simple

RUN adduser --system --no-create-home --group app
USER app

CMD [ "python3", "./main.py" ]
