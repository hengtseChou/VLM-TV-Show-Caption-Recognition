#!bin/bash
if ! which "virtualenv" >/dev/null 2>&1; then
    echo "virtualenv not installed. exiting"
    exit 1
fi
virtualenv .data_prep
source .data_prep/bin/activate
pip install --upgrade pip
pip install pandas pyarrow pillow tqdm shortuuid
deactivate
