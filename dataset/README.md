### Download directly from [here](https://ucdcs-student.ucd.ie/~cwang/autocoder/source_code.zip)

unzip the source_code file and move it under this directory.


### Or build dataset from scratch
This allows to customize dataset building. Below is an example of the building process.

Let's use Python and Java codes from [The Algorithms project](https://github.com/TheAlgorithms) as the dataset. We want AutoCoder to help auto-complete codes at a general level. The codes of The Algorithms suits the need! Another reason is personally thinking this code from this project is well written (high-quality codes!).

##### download source code
```
git clone https://github.com/TheAlgorithms/Python
git clone https://github.com/TheAlgorithms/Java
```

##### Move the dowloaded two folders into here this `dataset/` directory and then run

```
python convert.py --segment_len 256 --stride 10 --dev_size 0.1
```

You will find a train set named train.jsonl and dev set named dev.jsonl under `source_code/json/`.

Have a look at the `convert.py` script for the specific process of dataset construction or quickly read [this blog](#). 

