Create images and videos as full-fledged ASCII and UTF-8 art
---
You can assign your characters using uploaded font, as well as create a background fill.

A feature of the algorithm is the combination of foreground and background. The sketch is created based on the OpenCV adaptive threshold and expressed in base symbols selected, and then fill symbols are applied to paint the image with a gradient. You can also set the gradient layer not with a single character, but with a sentence or any set of unicode characters. If you create a video, you will see how the layer moves over time.

![preview](https://github.com/AndreiIljuhin/ascii-2L/blob/master/preview1.png)

How to use
---
Clone the repository and open it in the Java IDE, ask ide to load the dependencies in the pom file. Change the INPUT_FILE_NAME string in MainClass.java by file name and move the file to the \data_set\input&output directory, run the main.

To create a new character set, change the fontPatch string in the NewSet class to the full path to the ttf file. You can change the _filling variable, where each element represents the fill gradient. Start the class, wait for it to finish, and copy the folder name from the console. Change the SYMBOLS_FOLDER variable in MainClass to the created folder name.

If you have Maven installed, you can compile the project and run it by passing the file name to MainClass or the path to the ttf font to NewSet.
***

![preview](https://github.com/AndreiIljuhin/ascii-2L/blob/master/preview2.png)

Problems
---
Sometimes you need to manually adjust the adaptive threshold to view the images, go to the data_set\threshold folder. The algorithm works well on well-defined lines, but is not suitable for most ordinary photos and videos. It is best to apply it to anime or images in a minimalist style. You can disable the threshold option and transfer the thumbnail image from Photoshop.
***
If you create a set.

Creating character images is fraught with incorrect fields and differences between the resulting images and text files. The selection showed that monospaced fonts created with a size of 15-16 pixels best match the resulting text. The error can be caused by a change in the distance between characters in normal text and different interpolation.

The evaluation function depends on the size and type of the symbol images. You can adjust the highlighted characters manually after creating a new set by changing the flag or excluding the character from the evaluation by adding _false to the file name if you don't like it. It is best to create images with a height of 14-16 pixels.