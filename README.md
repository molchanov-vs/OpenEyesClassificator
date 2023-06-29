# OpenEyesClassificator
Classificator of opened/closed eyes on CNN

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [TODO](#todo)


## Installation
For using report.ipynb
```
conda create --name OpenEyesClassificator python=3.9
conda activate OpenEyesClassificator
pip install -r requirements.txt
```

For using class OpenEyesClassificator
```
conda create --name OpenEyesClassificator python=3.9
conda activate OpenEyesClassificator
pip install numpy
pip install Pillow
pip install tensorflow
```

## Usage
```
from model import OpenEyesClassificator

classifier = OpenEyesClassificator()
prediction = classifier.predict(inpIm)
print(prediction)
```
где inpIm - полный путь к изображению глаза, который возвращает is_open_score - float score классификации от 0.0 до 1.0 (где 1 - открыт, 0 - закрыт).

## TODO

- [ ] usage of clusimage.
- [ ] add documentation to functions.