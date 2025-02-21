 
Steps:

Step 1. Download the "project-science-lab" to laptop's Downloads folder.

Step 2. Setup the Python environment
	There are two ways to setup the Python environment. 
	You can either use the laptop's environment or create a virtual environment. 
	My recommendation is to use virtual environment (Method 2)

   Method 1: Installation to use laptop's environment
	cd ~/Downloads/project-science-lab/
	brew install python@3.13
	pip install --upgrade pip
	Note: If pip doesn't exist, recommended to follow "Method 2"
	pip install -r requirements.txt

   Method 2:
	cd ~/Downloads/project-science-lab/
	brew install python@3.13
	python3 -m venv myenv
	source myenv/bin/activate
	pip install -r requirements.txt


Step 3: Start the project
python3 app.py

Step 4: Open the website in browser
http://127.0.0.1:5000/

