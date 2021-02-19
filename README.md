# ml-in-prod
#Installation:
	* Clone github repo
	* Install requirements from terminal-pip insall -r "requiremets.txt"

#Usages:
	*To run the code in the local computer, the update updates are required
		*load.py:
			'/storage/upload/model.json' to os.path.join(os.path.dirname(__file__),'model.json')
			'/storage/upload/model.h5' to os.path.join(os.path.dirname(__file__),'model.h5')
		*app.py:
			'/storage/upload/' to 'uploaded/image/'
			app.run(debug=True, port=8000, host='0.0.0.0') to app.run()
		*/model folder:
			Place model.json and model.h5
	*Run python app.py
	*Go to http://127.0.0.1:5000/ for local deloyment
	*Upload image
	*Select gender
	*Select orginal/preprocessed image
