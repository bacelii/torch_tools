# torch_tools

Repository for pytorch wrappers, tutorials and any useful classes/functions for implementing deep learning models in pytorch

Overall solution for local development in google colab with vs code

1. install google desktop
2. Create a folder in google drive
3. clone a github repository to that folder
4. Launch local vs code to edit repository
   (changes will be synced with google drive on a delay)
   --- running notebooks ---
5. Create google colab (in github repo applications folder)
6. add the %load_ext autoreload at top of notebook
   - so that changes in vscode local once synced with google drive
     will be reflected in the module
7. Mount the google drive folder in google colab
8. Install the package with pip or just add the path to modules
   at the beginning of the notebook
9. commit and push progress of repo to github from local machine (or even google colab interface)
