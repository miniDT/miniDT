echo "Activating Python environment from: $PWD"
source bin/activate
# Adding root folder to the Python PATH
export PYTHONPATH="$PWD:$PYTHONPATH"
