FROM kevinnunu/gluoncv0.3.0:mxnet1.3.0_gpu_cu90_py3_cv

RUN mkdir /app
WORKDIR /app

#RUN pip install --upgrade pip && pip install gluoncv==0.3.0

#COPY requirements.txt /app/requirements.txt
#RUN apt-get update  && pip install --upgrade pip && pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

COPY . /app
