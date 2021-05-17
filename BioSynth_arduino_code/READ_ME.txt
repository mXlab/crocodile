The newer version of the crocodile code is in the CROCODILE_LAURENCE folder. 

You'll need platform.io to upload it to the teensy. 

Platform.io is an extension of visual studio code.

1. Install VSCODE ( https://code.visualstudio.com/download ) 

2. In VSCODE, go to the extension tab and search for Platform.io  and install the extension

3. Restart VSCODE

4. Go on the Platform.io tab and select open project. Select the CROCODILE_LAURENCE folder.

5. To upload the code click on the arrow pointing to the right in the bottom blue bar.

6. If on the first time uploading a library is missing, add them to the project through the Platform.io librarie manager. 
   for custom libraries like biodata library you'll have to use a custom command. 
	- open a terminal instance from the terminal icon in the bottom blue bar
	- type this command : pio lib install /PATHTOLIBRARY

	! The zip file of the library should be in a directory on your computer.