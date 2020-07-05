#!/bin/bash

set -euxo pipefail

for req in $(cat requirements.txt ./build-requirements.txt)
do 
    pip${python_version} install $req
done

make -j$(nproc)
