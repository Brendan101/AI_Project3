Instructions
	
	1) This program requires scikit to be up-to-date, specifically
	   sklearn, skimage and numpy. These are all included in anaconda
	2) This program uses an SVM which is trained every time it is run,
	   therefore it requires that the given Training folder is in the
	   same directory as ProcessImage.py
	3) How to run: python ProcessImage.py filepath_to_image
	
	FYI:
	4) The code I used to generate the data in results.txt is commented
	   out in ProcessImage.py starting on line 84, but there is no need
	   to run this every time a single image is to be classified. If you
	   do want to check it out yourself, uncomment that section and comment
	   out the line directly preceding it:
	   "lin_clf.fit(shaped_data, classification)" instead.
	   That test trains on half the data instead of all of it and tests
	   accuracy on the other half.
