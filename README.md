# HOW TO INSTALL

1. Create an account in Google Colab (for instance, by using your main Google account): https://colab.research.google.com/.

2. Create a Google Drive account: https://drive.google.com.

3. Connect Google Colab to your Google Drive by typing and running the following command in a new project of Google Colab:

   from google.colab import drive
   
   drive.mount(‘/content/gdrive’)
   
   See more detailed instructions here: https://www.marktechpost.com/2019/06/07/how-to-connect-google-colab-with-google-drive/.
   
4. Next, enter the following line to clone all the files to your Google Drive:

   !git clone https://github.com/NikitaSSerov/EMRIS.git

5. Open EMRIS tool.ipynb and follow the internal instructions to work further.

   Easy as that :)

# HOW TO USE

1. Create a table of synthesis parameters, where each of the syntheses is indexed, and upload it on your Google Drive either in .xlsx or .csv format.

2. Create the folder with your images on Google Drive (.tif, .png, .jpeg files are supported), where its names represent indexes in the syntheses table created on step 1.

3. Unpack EMRIS GitHub files with the following command (if it was not done before):
!git clone https://github.com/NikitaSSerov/EMRIS.git

4. Open EMRIS tool.ipynb file and fill in all the information above regarding file destinations, query image, number of similar images to show, as well as the search type (image or contour).

5. Run the whole code (Ctrl+F9) and see the result below.

# ISSUES FACED AND SOLUTIONS

1.

2.

3.

4.

5.

If your problem is not in the list, You can contact us anytime using:
1. E-mail: serov@scamt-itmo.ru
2. WhatsApp / Telegram: +7 (977) 509-91-81 
