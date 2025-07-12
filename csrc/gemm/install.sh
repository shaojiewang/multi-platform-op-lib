set -x

python3 -m pip uninstall ampere_bfgemm -y
python3 setup.py bdist_wheel && cd dist && python3 -m pip install *.whl && cd -
rm -rf ampere_bfgemm.egg-info __pycache__

set +x