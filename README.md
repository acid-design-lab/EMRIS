# HOW TO INSTALL

1. Create an account in Google Colab (for instance, by using your main Google account): https://colab.research.google.com/.
2. Create a Google Drive account: https://drive.google.com.
3. Connect Google Colab to your Google Drive: https://www.marktechpost.com/2019/06/07/how-to-connect-google-colab-with-google-drive/.
4. Create a new notebook in Google Colab and then enter the following line to clone all the files to your Google Drive:
!git clone https://github.com/NikitaSSerov/EMRIS.git

   Easy as that :)

# HOW TO USE

1. Create a table of synthesis parameters, where each of the syntheses is indexed, and upload it on your Google Drive either in .xlsx or .csv format.
2. Create the folder with your images on Google Drive (.tif, .png, .jpeg files are supported), where its names represent indexes in the syntheses table created on step 1.
3. Open TOOL.py file and fill in all the information above regarding file destinations, query image, as well as the search type (image or contour).
