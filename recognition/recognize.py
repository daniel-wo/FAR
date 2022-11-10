import docker
import sys
import os
import webbrowser
client = docker.from_env()


os.system("docker run -v $(pwd)/output:/recognition/output far " + sys.argv[1])

webbrowser.open("../frontend/doc/examples/html_pages/01_basic.html")