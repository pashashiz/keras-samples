Create virtual env
```
python -m venv venv
```

Activate a virtual env:
```
source venv/bin/activate
```
NOTE: configure PyCharm to use this env also

Install libs:
```
pip install -r requirements.txt
```

Make PyCharm to resolve tansorflow.keras via a trick
```shell
cd venv/lib/python3.9/site-packages/tensorflow
ln -s ../keras/api/_v2/keras/ keras
```