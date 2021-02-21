# ml-in-prod

## Cloud Application:

	* visti site:http://jsa4qu3jfo40nn.gradient.paperspace.com:8000

	* Upload image

	* Select gender

	* Select orginal/preprocessed image

	* Hit 'submit & predic' button

	*Note: Server may not be up running all the time.

## Installation:

	* Clone github repo

	* Install requirements from terminal-pip insall -r "requiremets.txt"

## Usages (Local Machine):

	* To run the code in the local computer, the update updates are required

		* load.py:

			'/storage/upload/model.json' to os.path.join(os.path.dirname(__file__),'model.json')

			'/storage/upload/model.h5' to os.path.join(os.path.dirname(__file__),'model.h5')

		* app.py:

			'/storage/upload/' to 'uploaded/image/'

			app.run(debug=True, port=8000, host='0.0.0.0') to app.run()

		* /model:

			Place model.json and model.h5 in the 'model'folder

	* Run python app.py

	* Go to http://127.0.0.1:5000/ for local deployment

	* Upload image

	* Select gender

	* Select orginal/preprocessed image

	* Hit 'submit & predic' button
