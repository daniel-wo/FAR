# FAR - Finite Automaton Recognizer
This is a repository for my bachelor thesis, the project aims to allow the recognition of handrawn finite automata.

## How to Use

1. Navigate to the repository directory: `(/FAR)`

2. Build the container: `docker build . -t far`
   (this takes a long time)

3. Add your images to the input directory: `(/FAR/recognition/IO)`

4. Naviagate to the recognition directory: `(/FAR/recognition)`

5. Execute the recognition script: `python3 recognize.py IO/{your_image}.png`

With this after the script is executed, your default browser should open, showing the recognition result.
