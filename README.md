## Create images and videos as ASCII or UTF-8 art  (づ๑•ᴗ•๑)づ.
Algorithm without using neural networks.

![preview](https://github.com/AndreiIljuhin/ascii-2L/blob/master/preview0.gif)

Change the INPUT_FILE_NAME in MainClass by video or image file name and move it to the \data_set\input&output directory. You kan run now!

To create new characters change the fontPatch in NewSet class. Run and copy the folder name from the console into the SYMBOLS_FOLDER variable in MainClass.

***

![preview](https://github.com/AndreiIljuhin/ascii-2L/blob/master/preview2.png)

Compile the video yourself using ffmpeg for better quality

ffmpeg -y -framerate 23.98 -i frame-%03d.png -c:v libvpx-vp9 -b 3000k -minrate 2000k -maxrate 9000k -bufsize 1835k -vf "format=yuv420p" "sample_conv.webm"

ffmpeg -i "sample_conv.webm" -i "sample.webm" -map 0:v -map 1:a -c:v copy -c:a libopus -b:a 128k "sample_conv_with_audio.webm"
