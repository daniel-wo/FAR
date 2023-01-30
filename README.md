# FAR - Finite Automaton Recognizer
This is a repository for my bachelor thesis, the project aims to allow the recognition of handrawn finite automata.

## How to Use

1. Download the models: https://drive.google.com/drive/folders/17uZaehJ-_TW_3oGN3rMZi77hRcoy6DNG?usp=sharing and extract the contents of the archive into a directory `/FAR/recognition/models` so that the model files are directly under this directory (`/FAR/recognition/models/{model}.h5`)

2. Navigate to the repository root: `(/FAR)`

3. Build the container: `docker build . -t far`
   (this takes a long time)
   
4. Navigate to the recognition directory: `(/FAR/recognition)`

(optional) Do a test run with one of the test images: `python3 recognize.py ../test_images/automaton1.png`

5. Add your images to the IO directory (create this directory if needed): `(/FAR/recognition/IO)`
   (sometimes depending on the OS it is necessary to add permissions on this folder, e.g. `chmod 777 IO`)

6. Execute the recognition script: `python3 recognize.py IO/{your_image}.png`

After the script is executed, your default browser should open and show the recognition result.

The IO directory gets mounted into the docker container which is the reason why we add input images here. After running the tool it contains all the intermediary results which are mentioned in the thesis.
