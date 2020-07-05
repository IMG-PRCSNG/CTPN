#!/bin/bash

set -euxo pipefail

for req in $(cat requirements.txt ./build-requirements.txt)
do 
    pip${PYTHON_VERSION} install $req
done

make -j$(nproc)
