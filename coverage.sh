coverage run --omit nn_test --omit */.local/* nn_test.py test
coverage report -m
rm .coverage