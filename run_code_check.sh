#?1bin/sh
#running all the unit tests:
python -m unittest discover cappuccino 'test_*.py' -v

#static code checking:
#run after the unit tests so the code can be seen on the console
python setup.py flakes

#pylint
#pylint cappuccino

