import docker
import sys
import os
import webbrowser
client = docker.from_env()


os.system("docker run -v $(pwd)/IO" + ":/recognition/IO far " + sys.argv[1])
webbrowser.open("IO/0_frontend.html")
