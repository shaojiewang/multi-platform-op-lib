set -x

python3 -m pip uninstall bfgemm_test -y
python3 setup.py bdist_wheel && cd dist && python3 -m pip install *.whl && cd -
rm -rf bfgemm_test.egg-info __pycache__

set +x
